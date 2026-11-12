"""Configuration-only tests: no server assets or model construction."""
import copy
from pathlib import Path
import pytest
from mdrl_finetune.config import load_config, validate_config


def base(stage="sft"):
    return {"stage": stage, "seed": 2024,
            "model": {"checkpoint": "trusted.pt", "expected_iteration": 985000},
            "targets": [{"id": name, "receptor": name + ".pdbqt", "center": [0, 0, 0],
                         "box_size": 24, "mu": -10., "le_model": name + ".pkl"}
                        for name in ("A", "B")],
            "scoring": {"executable": "vina", "fscores": "fpscores.pkl.gz"},
            "runtime": {"device": "cpu"}, "sft": {}, "nft": {"init_checkpoint": "sft.pt"}}


def test_schema_normalizes_without_mutating_input():
    source = base()
    cfg = validate_config(source)
    assert cfg["targets"][0]["box_size"] == [24, 24, 24]
    assert source["targets"][0]["box_size"] == 24
    assert cfg["sft"]["rounds"] == 8 and cfg["sft"]["lr"] == 1e-5


@pytest.mark.parametrize("change", [
    lambda cfg: cfg["model"].update(expected_iteration=12),
    lambda cfg: cfg["targets"][0].update(sigma=0),
    lambda cfg: cfg["targets"][0].update(center=[float("nan"), 0, 0]),
    lambda cfg: cfg["sft"].update(lr=float("nan")),
    lambda cfg: cfg["scoring"].update(workers=0),
    lambda cfg: cfg["targets"][1].update(id="A"),
])
def test_bad_schema(change):
    cfg = base()
    change(cfg)
    with pytest.raises(ValueError):
        validate_config(cfg)


def test_stage_and_init_guard():
    with pytest.raises(ValueError, match="stage"):
        validate_config(base(), "nft")
    cfg = base("nft")
    cfg["nft"].pop("init_checkpoint")
    with pytest.raises(ValueError, match="init_checkpoint"):
        validate_config(cfg)


def test_all_six_plans_match_approved_budgets():
    paths = list((Path(__file__).resolve().parents[1] / "configs/sft_nft").glob("*.yaml"))
    assert len(paths) == 6
    for path in paths:
        cfg = validate_config(load_config(path))
        assert cfg["model"]["expected_iteration"] == 985000
        assert all(t["box_size"] == [24, 24, 24] for t in cfg["targets"])
        assert cfg["scoring"]["workers"] == 16
        if cfg["stage"] == "sft":
            assert cfg["sft"]["rounds"] == 8 and cfg["sft"]["updates_per_round"] == 100
        else:
            assert cfg["nft"]["iterations"] == 100 and cfg["nft"]["attempts"] == 128
