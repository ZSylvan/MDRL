"""Trusted checkpoint IO and complete round-boundary state."""
from __future__ import annotations

import hashlib
import inspect
import json
import os
import random
from pathlib import Path
import numpy as np
import torch
from .training import require_finite

FORMAT = "mdrl-sft-nft-v1"


def load_payload(path):
    kwargs = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = False
    return torch.load(Path(path), **kwargs)


def load_trusted_checkpoint(path, *, expected_iteration=985000, model=None, map_location="cpu"):
    payload = load_payload(path)
    if (not isinstance(payload, dict) or type(payload.get("iteration")) is not int
            or payload["iteration"] != expected_iteration or payload.get("diagnostic_only")):
        raise ValueError("trusted checkpoint must have iteration=%d" % expected_iteration)
    if not isinstance(payload.get("model"), dict) or not payload["model"]:
        raise ValueError("checkpoint model state is missing")
    require_finite(payload["model"], "checkpoint model")
    if model is not None:
        model.load_state_dict(payload["model"], strict=True)
    return payload


def atomic_save(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(str(temporary), str(path))


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(str(temporary), str(path))


def capture_rng():
    state = dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state())
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state:
        if not torch.cuda.is_available():
            raise ValueError("cannot restore a CUDA RNG checkpoint without CUDA")
        torch.cuda.set_rng_state_all(state["cuda"])


def fingerprint(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_boundary(path, stage, config):
    state = load_payload(path)
    if (not isinstance(state, dict) or state.get("format") != FORMAT
            or state.get("stage") != stage or state.get("diagnostic_only")
            or state.get("config_fingerprint") != fingerprint(config)):
        raise ValueError("incompatible stage/config or diagnostic resume checkpoint")
    required = {"model", "optimizer", "rng", "completed", "native_config", "source_digest"}
    required |= {"pool"} if stage == "sft" else {"old", "reference", "successful_updates", "top_k"}
    if not required <= state.keys() or type(state["completed"]) is not int or state["completed"] < 0:
        raise ValueError("incomplete round-boundary checkpoint")
    require_finite((state["model"], state["optimizer"], state.get("old"), state.get("reference")), "resume")
    return state
