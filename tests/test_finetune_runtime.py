"""Actual CPU optimizer/checkpoint loops, with sampling/docking replaced explicitly."""
import copy
from pathlib import Path
import random
import numpy as np
import pytest
import torch
import yaml
from rdkit import Chem
from mdrl_finetune.checkpoint import load_payload
from mdrl_finetune.config import validate_config
from mdrl_finetune.runtime import run_sft, run_nft
from test_finetune_config import base
from test_finetune_training import Tiny, graph


def fixture(tmp_path, stage="sft"):
    cfg = base(stage)
    native = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs/train/train_MDRL.yml").read_text())
    checkpoint = tmp_path / "pretrained.pt"
    if not checkpoint.exists():
        torch.save(dict(iteration=985000, config=native, model=Tiny().state_dict()), checkpoint)
    cfg["model"]["checkpoint"] = str(checkpoint)
    cfg["sft"] = dict(rounds=8, initial_attempts=3, round_attempts=2, top_k=2,
                      updates_per_round=1, batch_size=2, generation_batch_size=2)
    cfg["nft"].update(iterations=3, attempts=4, rollout_batch_size=2, epochs=2, micro_batch_size=1)
    return cfg


def sampler(model, count, **kwargs):
    result = []
    for index in range(count):
        count_nodes = 2 + index % 2
        clean = graph(count_nodes)
        # Consumption of all three RNGs must be reproduced after a boundary resume.
        clean["node_pos"] += random.random() + np.random.rand() + torch.rand(())
        molecule = Chem.MolFromSmiles("C" * count_nodes)
        conformer = Chem.Conformer(count_nodes)
        for atom_index, xyz in enumerate(clean["node_pos"].tolist()):
            conformer.SetAtomPosition(atom_index, xyz)
        molecule.AddConformer(conformer)
        result.append(dict(graph=clean, valid=index < 2, trainable=True,
                           rdmol=molecule if index < 2 else None,
                           smiles="C" * count_nodes if index < 2 else None, raw={}))
    return result


class Scorer:
    def score_many(self, rows, directory):
        for index, row in enumerate(rows):
            row.update(reward=.3 + .4 * index if row["valid"] else -1., failed=not row["valid"])
        return rows

    def close(self):
        pass


def equivalent(left, right):
    if torch.is_tensor(left):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            equivalent(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            equivalent(a, b)
    else:
        assert left == right


def test_sft_eight_rounds_and_exact_boundary_optimizer_resume(tmp_path):
    cfg = fixture(tmp_path)
    counts = []
    def counted(model, count, **kwargs):
        counts.append(count)
        return sampler(model, count, **kwargs)
    final = run_sft(cfg, tmp_path / "sft", model=Tiny(), sampler=counted, scorer=Scorer())
    assert counts == [3] + [2] * 7 and final["completed"] == 8
    resumed = run_sft(cfg, tmp_path / "branch", model=Tiny(), sampler=sampler, scorer=Scorer(),
                      resume=tmp_path / "sft/round_03/checkpoint.pt")
    equivalent(final["model"], resumed["model"])
    equivalent(final["optimizer"], resumed["optimizer"])
    assert torch.equal(final["rng"]["torch"], resumed["rng"]["torch"])
    assert len(final["pool"]) == 2


def test_nft_uses_sft8_saves_three_models_and_resumes(tmp_path):
    cfg = fixture(tmp_path)
    run_sft(cfg, tmp_path / "sft", model=Tiny(), sampler=sampler, scorer=Scorer())
    cfg["stage"] = "nft"
    cfg["nft"]["init_checkpoint"] = str(tmp_path / "sft/final.pt")
    final = run_nft(cfg, tmp_path / "nft", model=Tiny(), sampler=sampler, scorer=Scorer())
    assert final["completed"] == 3 and final["successful_updates"] == 3
    assert final["metrics"]["n_valid"] == 2 and final["metrics"]["n_training_invalid"] == 2
    assert final["metrics"]["reward_mean"] == pytest.approx(-.25)
    assert (tmp_path / "nft/checkpoints/best.pt").is_file()
    branch = run_nft(cfg, tmp_path / "nft_branch", model=Tiny(), sampler=sampler, scorer=Scorer(),
                     resume=tmp_path / "nft/checkpoints/iter000000.pt")
    for key in ("model", "old", "reference", "optimizer"):
        equivalent(final[key], branch[key])
    assert torch.equal(final["rng"]["torch"], branch["rng"]["torch"])


def test_resume_does_not_overwrite_source_and_rejects_config_change(tmp_path):
    cfg = fixture(tmp_path)
    run_sft(cfg, tmp_path / "sft", model=Tiny(), sampler=sampler, scorer=Scorer())
    source = tmp_path / "sft/checkpoints/last.pt"
    before = source.read_bytes()
    with pytest.raises(FileExistsError):
        run_sft(cfg, tmp_path / "sft", model=Tiny(), sampler=sampler, scorer=Scorer(), resume=source)
    cfg["sft"]["lr"] = .1
    with pytest.raises(ValueError, match="incompatible"):
        run_sft(cfg, tmp_path / "rejected", model=Tiny(), sampler=sampler, scorer=Scorer(), resume=source)
    assert not (tmp_path / "rejected").exists() and source.read_bytes() == before


def test_nonfinite_gradient_keeps_last_safe_checkpoint_and_marks_failed(tmp_path):
    cfg = fixture(tmp_path)
    model = Tiny()
    model.atom.register_hook(lambda value: value * float("nan"))
    with pytest.raises(FloatingPointError):
        run_sft(cfg, tmp_path / "failed", model=model, sampler=sampler, scorer=Scorer())
    initial = load_payload(tmp_path / "failed/checkpoints/last.pt")
    assert initial["completed"] == 0 and not (tmp_path / "failed/final.pt").exists()
    assert '"FAILED"' in (tmp_path / "failed/status.json").read_text()


def test_wrong_pretraining_iteration_fails_before_output_creation(tmp_path):
    cfg = fixture(tmp_path)
    payload = load_payload(cfg["model"]["checkpoint"])
    payload["iteration"] = 985001
    torch.save(payload, cfg["model"]["checkpoint"])
    with pytest.raises(ValueError, match="985000"):
        run_sft(cfg, tmp_path / "rejected", model=Tiny(), sampler=sampler, scorer=Scorer())
    assert not (tmp_path / "rejected").exists()
