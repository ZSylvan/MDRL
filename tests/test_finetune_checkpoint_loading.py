"""Exercise the real loader with a tiny stand-in for the native architecture."""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from mdrl_finetune.checkpoint import FORMAT
from mdrl_finetune.sampling import load_native_model


class SmallModel(torch.nn.Module):
    def __init__(self, config, num_node_types, num_edge_types):
        super().__init__()
        assert num_node_types == 8 and num_edge_types == 6
        assert config.diff.num_timesteps == 1000
        self.weight = torch.nn.Parameter(torch.zeros(1))


@pytest.mark.parametrize("boundary", [False, True])
def test_native_and_finetune_checkpoint_architecture(monkeypatch, tmp_path, boundary):
    native = yaml.safe_load((Path(__file__).resolve().parents[1]
                             / "configs/train/train_MDRL.yml").read_text())
    payload = dict(config=native, model={"weight": torch.tensor([3.0])})
    if boundary:
        payload.update(format=FORMAT, stage="sft", completed=8,
                       config={"stage": "sft", "sft": {"rounds": 8}}, native_config=native)
    monkeypatch.setitem(sys.modules, "models.model", SimpleNamespace(MDRL=SmallModel))
    path = tmp_path / "checkpoint.pt"
    torch.save(payload, path)
    for source in (payload, path):
        model, config = load_native_model(source, "cpu")
        assert model.weight.item() == 3.0
        assert config == native


def test_boundary_cannot_fall_back_to_experiment_config(monkeypatch):
    monkeypatch.setitem(sys.modules, "models.model", SimpleNamespace(MDRL=SmallModel))
    with pytest.raises(ValueError, match="native training config"):
        load_native_model(dict(format=FORMAT, config={}, model={}), "cpu")
