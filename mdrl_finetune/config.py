"""Configuration assertions; the native architecture comes from the checkpoint."""
from __future__ import annotations
import copy
import math
from pathlib import Path
import yaml

ELEMENTS = ("C", "N", "O", "F", "P", "S", "Cl")


def load_config(path):
    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("configuration must be a mapping")
    return config


def _number(value, name, *, positive=False, integer=False, nonnegative=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or (integer and type(value) is not int)
            or (positive and value <= 0) or (nonnegative and value < 0)):
        raise ValueError("invalid numeric setting: " + name)


def validate_config(config, stage=None):
    cfg = copy.deepcopy(config)
    actual = cfg.get("stage", stage)
    if actual not in ("sft", "nft") or (stage is not None and actual != stage):
        raise ValueError("stage mismatch; expected sft or nft")
    cfg["stage"] = actual
    cfg.setdefault("seed", 2024)
    _number(cfg["seed"], "seed", integer=True, nonnegative=True)
    model = cfg.get("model", {})
    if not model.get("checkpoint"):
        raise ValueError("model.checkpoint is required")
    model.setdefault("expected_iteration", 985000)
    if type(model["expected_iteration"]) is not int or model["expected_iteration"] != 985000:
        raise ValueError("model.expected_iteration must be 985000")
    arch = model.setdefault("architecture", {})
    if tuple(arch.get("elements", ELEMENTS)) != ELEMENTS:
        raise ValueError("requires C,N,O,F,P,S,Cl in native order")
    if arch.get("n_timesteps", 1000) != 1000 or arch.get("conditioning", "unconditional") != "unconditional":
        raise ValueError("requires unconditional 1000-step MDRL")
    targets = cfg.get("targets")
    if not isinstance(targets, list) or len(targets) != 2:
        raise ValueError("exactly two targets are required")
    if len({target.get("id") for target in targets}) != 2:
        raise ValueError("target ids must be unique")
    for target in targets:
        for key in ("id", "receptor", "mu", "center"):
            if key not in target:
                raise ValueError("target requires " + key)
        if not isinstance(target["center"], list) or len(target["center"]) != 3:
            raise ValueError("target.center must have three coordinates")
        for value in target["center"]:
            _number(value, "center")
        box = target.setdefault("box_size", 24.0)
        if isinstance(box, (int, float)):
            box = [box] * 3
        if not isinstance(box, (list, tuple)) or len(box) != 3:
            raise ValueError("box_size must be scalar or three coordinates")
        for value in box:
            _number(value, "box_size", positive=True)
        target["box_size"] = list(box)
        target.setdefault("sigma", 1.0)
        target.setdefault("le_mu", 0.5)
        target.setdefault("le_sigma", 0.1)
        for key in ("mu", "le_mu"):
            _number(target[key], key)
        for key in ("sigma", "le_sigma"):
            _number(target[key], key, positive=True)
        if actual == "sft" and not target.get("le_model"):
            raise ValueError("SFT requires each target.le_model")
    scoring = cfg.setdefault("scoring", {})
    for key, default in dict(cpu=1, workers=16, timeout_s=300, exhaustiveness=8, num_modes=1).items():
        scoring.setdefault(key, default)
        _number(scoring[key], key, positive=True, integer=key != "timeout_s")
    if scoring.get("backend", "existing_rl_utils") != "existing_rl_utils":
        raise ValueError("unsupported scoring backend")
    for key in ("executable", "fscores"):
        if not scoring.get(key):
            raise ValueError("scoring." + key + " is required")
    runtime = cfg.setdefault("runtime", {})
    runtime.setdefault("device", "cpu")
    if not isinstance(runtime["device"], str) or not runtime["device"].startswith(("cpu", "cuda")):
        raise ValueError("runtime.device must be cpu or cuda")
    defaults = (
        dict(rounds=8, initial_attempts=3000, round_attempts=2000, top_k=4000,
             updates_per_round=100, batch_size=16, generation_batch_size=16)
        if actual == "sft" else
        dict(iterations=100, attempts=128, rollout_batch_size=16, epochs=4,
             micro_batch_size=64, min_valid=2))
    section = cfg.setdefault(actual, {})
    if actual == "nft" and not section.get("init_checkpoint"):
        raise ValueError("nft.init_checkpoint is required")
    for key, default in defaults.items():
        section.setdefault(key, default)
        _number(section[key], key, positive=True, integer=True)
    for key, default in dict(lr=1e-5, weight_decay=1e-4, grad_clip=1.0).items():
        section.setdefault(key, default)
        _number(section[key], key, positive=key != "weight_decay", nonnegative=key == "weight_decay")
    if actual == "nft":
        section.setdefault("beta", 1.0)
        section.setdefault("anchor_coef", 0.001)
        _number(section["beta"], "beta", positive=True)
        _number(section["anchor_coef"], "anchor_coef", nonnegative=True)
        if section["min_valid"] < 2:
            raise ValueError("min_valid must be at least 2")
    return cfg
