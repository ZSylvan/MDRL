"""Focused progress telemetry tests using CPU tiny kernels and mocked scoring."""
import json
from pathlib import Path
import random
import numpy as np
import torch
import pytest

from mdrl_finetune import sampling
from mdrl_finetune.runtime import _sample_with_progress


class _Model:
    def eval(self):
        return self


def _fake_sampler_parts(monkeypatch):
    def placeholder(count, device):
        return (np.full(count, 2), torch.arange(count).repeat_interleave(2),
                torch.stack((torch.arange(count) * 2, torch.arange(count) * 2 + 1)),
                torch.arange(count), 0)
    reverse_seen = []
    def reverse(model, count, batch_node, halfedge_index, batch_halfedge, **kwargs):
        reverse_seen.append(kwargs.get("completed_attempts"))
        return {"pred_node": torch.zeros((count * 2, 8)),
                "pred_pos": torch.zeros((count * 2, 3)),
                "pred_halfedge": torch.zeros((count, 6))}
    counter = {"n": 0, "before_decode": []}
    def decode(*args):
        counter["before_decode"].append(reverse_seen[-1] if reverse_seen else None)
        counter["n"] += 1
        return {"valid": counter["n"] % 2 == 0, "trainable": True}
    monkeypatch.setattr(sampling, "placeholder", placeholder)
    monkeypatch.setattr(sampling, "reverse_predictions", reverse)
    monkeypatch.setattr(sampling, "decode_prediction", decode)
    return counter


@pytest.mark.parametrize("count,expected", [(3000, [512] * 5 + [440]), (2000, [512] * 3 + [464])])
def test_fixed_attempt_batches_and_reconstruction_counts(monkeypatch, count, expected):
    counter = _fake_sampler_parts(monkeypatch)
    events = []
    rows = sampling.sample_native(_Model(), count, batch_size=512,
                                  progress_callback=events.append)
    complete = [e for e in events if e["event"] == "batch_complete"]
    assert len(rows) == count and [e["current_batch_size"] for e in complete] == expected
    assert [e["completed_attempts"] for e in complete] == list(np.cumsum(expected))
    assert all(e["completed_valid"] + e["completed_invalid"] == e["completed_attempts"] for e in complete)
    assert events[-1]["event"] == "completed" and events[-1]["generation_fraction"] == 1.0
    assert all(value in ({0} | set(np.cumsum(expected)[:-1])) for value in counter["before_decode"])


def test_callback_does_not_change_rng_and_none_is_backward_compatible(monkeypatch):
    _fake_sampler_parts(monkeypatch)
    random.seed(8); np.random.seed(8); torch.manual_seed(8)
    before = (random.random(), np.random.rand(), torch.rand(()))
    random.seed(8); np.random.seed(8); torch.manual_seed(8)
    one = sampling.sample_native(_Model(), 8, batch_size=4, progress_callback=lambda event: None)
    after_on = (random.random(), np.random.rand(), torch.rand(()))
    random.seed(8); np.random.seed(8); torch.manual_seed(8)
    two = sampling.sample_native(_Model(), 8, batch_size=4)
    after_off = (random.random(), np.random.rand(), torch.rand(()))
    assert len(one) == len(two) == 8
    assert after_on[0] == after_off[0] and after_on[1] == after_off[1] and torch.equal(after_on[2], after_off[2])


def test_real_tiny_reverse_has_native_steps_and_denoising_events():
    from test_finetune_training import Tiny
    model = Tiny()
    batch_node = torch.zeros(2, dtype=torch.long)
    halfedge = torch.tensor([[0], [1]])
    batch_edge = torch.zeros(1, dtype=torch.long)
    events, seen_t = [], []
    original = model.forward
    def forward(*args):
        seen_t.append(int(args[-1][0]))
        return original(*args)
    model.forward = forward
    sampling.reverse_predictions(model, 1, batch_node, halfedge, batch_edge,
                                 progress_callback=events.append, total_attempts=4,
                                 completed_attempts=0)
    denoise = [e for e in events if e["event"] == "denoising"]
    assert [e["steps_completed"] for e in denoise] == list(range(100, 1001, 100))
    assert seen_t == list(reversed(range(1000)))
    assert all(e["completed_attempts"] == 0 and e["generation_fraction"] == 0 for e in denoise)


def test_real_tiny_callback_toggle_preserves_prediction_and_rng():
    from test_finetune_training import Tiny
    def run(callback):
        model = Tiny(); batch_node = torch.zeros(2, dtype=torch.long)
        halfedge = torch.tensor([[0], [1]]); batch_edge = torch.zeros(1, dtype=torch.long)
        return sampling.reverse_predictions(model, 1, batch_node, halfedge, batch_edge,
                                            progress_callback=callback)
    random.seed(19); np.random.seed(19); torch.manual_seed(19)
    on = run(lambda event: None)
    rng_on = (random.random(), np.random.rand(), torch.rand(()))
    random.seed(19); np.random.seed(19); torch.manual_seed(19)
    off = run(None)
    rng_off = (random.random(), np.random.rand(), torch.rand(()))
    for key in ("pred_pos", "pred_node", "pred_halfedge"):
        assert torch.equal(on[key], off[key])
    assert rng_on[0] == rng_off[0] and rng_on[1] == rng_off[1] and torch.equal(rng_on[2], rng_off[2])


def test_old_injected_sampler_is_not_passed_callback():
    seen = {}
    def old(model, count, *, batch_size, device):
        seen.update(count=count, batch_size=batch_size, device=device)
        return [{"valid": True}] * count
    rows = _sample_with_progress(old, object(), 3, 512, "cpu", lambda event: None)
    assert len(rows) == 3 and seen == {"count": 3, "batch_size": 512, "device": "cpu"}


def test_sft_generation_batch_configs_and_defaults():
    import yaml
    from mdrl_finetune.config import validate_config
    from test_finetune_config import base
    assert validate_config(base(), "sft")["sft"]["generation_batch_size"] == 512
    root = Path(__file__).resolve().parents[1] / "configs/sft_nft"
    for name in ("brd4-cdk2", "thrombin-trypsin", "chk1-auroraa"):
        cfg = yaml.safe_load((root / ("mdrl-sft-%s-s2024-r1.yaml" % name)).read_text())
        assert cfg["sft"]["generation_batch_size"] == 512
        assert cfg["sft"]["batch_size"] == 16


def test_runtime_publishes_phases_before_hooks_and_final_completion(tmp_path, monkeypatch):
    from test_finetune_runtime import fixture, sampler, Scorer, Tiny
    import mdrl_finetune.runtime as runtime
    cfg = fixture(tmp_path)
    phases = []
    class HookScorer(Scorer):
        def score_many(self, rows, directory):
            phases.append(("score", json.loads((Path(directory) / "sampling_progress.json").read_text())["phase"]))
            rows = super().score_many(rows, directory)
            rows[0]["failed"] = True  # scoring failure must not erase reconstruction validity
            return rows
    def hook_training(*args, **kwargs):
        phases.append(("train", json.loads((tmp_path / "sft/progress.json").read_text())["phase"]))
        return [0.0]
    monkeypatch.setattr(runtime, "sft_step", hook_training)
    original_progress = runtime._write_progress
    def checked_progress(out, directory, payload):
        original_progress(out, directory, payload)
        if payload.get("phase") == "round_complete":
            assert (directory / "checkpoint.pt").is_file() and (out / "checkpoints/last.pt").is_file()
            assert (directory / "metrics.json").is_file()
        if payload.get("phase") == "scoring_complete" and payload.get("round") == 0:
            phases.append(("counts", payload["completed_valid"], payload["completed_invalid"]))
        if payload.get("phase") == "sft_complete":
            assert (out / "final.pt").is_file() and json.loads((out / "status.json").read_text())["status"] == "COMPLETED"
    monkeypatch.setattr(runtime, "_write_progress", checked_progress)
    runtime.run_sft(cfg, tmp_path / "sft", model=Tiny(), sampler=sampler, scorer=HookScorer())
    assert phases and all(item[1] == "scoring" for item in phases if item[0] == "score")
    assert all(item[1] == "training" for item in phases if item[0] == "train")
    progress = json.loads((tmp_path / "sft/progress.json").read_text())
    assert progress["phase"] == "sft_complete" and progress["completed_rounds"] == 8
    assert json.loads((tmp_path / "sft/round_00/sampling_progress.json").read_text())["phase"] == "round_complete"
    assert ("counts", 2, 1) in phases


def test_resume_from_completed_sft_boundary_publishes_completion(tmp_path):
    from test_finetune_runtime import fixture, sampler, Scorer, Tiny
    from mdrl_finetune.runtime import run_sft
    cfg = fixture(tmp_path)
    run_sft(cfg, tmp_path / "source", model=Tiny(), sampler=sampler, scorer=Scorer())
    run_sft(cfg, tmp_path / "branch", model=Tiny(), sampler=sampler, scorer=Scorer(),
            resume=tmp_path / "source/final.pt")
    progress = json.loads((tmp_path / "branch/progress.json").read_text())
    assert progress["phase"] == "sft_complete" and progress["completed_rounds"] == 8
