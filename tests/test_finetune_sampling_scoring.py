"""Real kernels/RDKit, with reconstruction tools and Vina explicitly mocked."""
import copy
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pytest
import torch
from rdkit import Chem
from mdrl_finetune.scoring import _load_sascorer
from mdrl_finetune import sampling, scoring
from test_finetune_config import base
from test_finetune_training import Tiny
from mdrl_finetune.config import validate_config


def reconstruct(decoded, **kwargs):
    molecule = Chem.RWMol()
    for element in decoded["element"]:
        molecule.AddAtom(Chem.Atom(int(element)))
    types = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE,
             3: Chem.BondType.TRIPLE, 4: Chem.BondType.AROMATIC}
    for (left, right), kind in zip(decoded["bond_index"].T, decoded["bond_type"]):
        if left < right:
            molecule.AddBond(int(left), int(right), types[int(kind)])
    molecule = molecule.GetMol()
    conformer = Chem.Conformer(len(decoded["element"]))
    for index, xyz in enumerate(decoded["atom_pos"]):
        conformer.SetAtomPosition(index, [float(value) for value in xyz])
    molecule.AddConformer(conformer)
    Chem.SanitizeMol(molecule)
    return molecule


def logits(classes, width):
    result = torch.full((len(classes), width), -10.)
    if classes:
        result[torch.arange(len(classes)), torch.tensor(classes)] = 10.
    return result


def test_decode_mask_map_preserves_correct_bond_not_interleaved_pairs():
    fake = SimpleNamespace(reconstruct_from_generated_with_edges=reconstruct)
    with patch.dict(sys.modules, {"utils.reconstruct": fake}):
        row = sampling.decode_prediction(logits([0, 7, 1], 8), torch.arange(9.).reshape(3, 3),
                                         logits([1, 2, 3], 6), torch.triu_indices(3, 3, 1))
    assert row["valid"] and row["trainable"]
    assert row["retained_atom_indices"].tolist() == [0, 2]
    assert row["graph"]["node_type"].tolist() == [0, 1]
    assert row["graph"]["halfedge_type"].tolist() == [2]
    assert torch.allclose(row["graph"]["node_pos"].mean(0), torch.zeros(3))


def test_reconstruction_topology_change_rejected_but_negative_graph_kept():
    def changed(decoded, **kwargs):
        altered = copy.deepcopy(decoded)
        altered["bond_type"][:] = 1
        return reconstruct(altered)
    with patch.dict(sys.modules, {"utils.reconstruct": SimpleNamespace(reconstruct_from_generated_with_edges=changed)}):
        row = sampling.decode_prediction(logits([0, 0], 8), torch.zeros(2, 3),
                                         logits([2], 6), torch.triu_indices(2, 2, 1))
    assert not row["valid"] and row["trainable"]
    assert row["graph"]["halfedge_type"].item() == 2
    assert "changed" in row["error"]


def test_empty_and_nonfinite_predictions_are_not_trainable():
    row = sampling.decode_prediction(logits([7], 8), torch.zeros(1, 3),
                                     logits([], 6), torch.empty(2, 0, dtype=torch.long))
    assert not row["trainable"]
    row = sampling.decode_prediction(logits([0], 8), torch.full((1, 3), float("nan")),
                                     logits([], 6), torch.empty(2, 0, dtype=torch.long))
    assert not row["trainable"]


def test_sampler_runs_native_thousand_steps_and_never_replenishes_failed_graphs():
    model = Tiny()
    with torch.no_grad():
        model.atom.zero_()
        model.atom[0] = 3.
    seen = []
    original = model.forward
    def forward(*args):
        seen.append(int(args[-1][0]))
        return original(*args)
    model.forward = forward
    def holder(count, device):
        sizes = np.full(count, 2)
        nodes = torch.arange(count).repeat_interleave(2)
        halves = torch.stack([torch.arange(count) * 2, torch.arange(count) * 2 + 1])
        return sizes, nodes, halves, torch.arange(count), 0
    with patch.object(sampling, "placeholder", side_effect=holder), patch.dict(
            sys.modules, {"utils.reconstruct": SimpleNamespace(reconstruct_from_generated_with_edges=reconstruct)}):
        rows = sampling.sample_native(model, 3, batch_size=2)
    assert len(rows) == 3 and len(seen) == 2000
    assert seen[:1000] == list(reversed(range(1000)))
    assert all(not row["valid"] and row["trainable"] for row in rows)


class Predictor:
    def predict(self, values):
        assert np.asarray(values).shape == (1, 2048)
        return np.asarray([2.0])


def test_sft_four_property_components_and_nft_docking_only(tmp_path):
    cfg = base()
    sascorer = _load_sascorer()
    cfg["scoring"]["fscores"] = str(Path(sascorer.__file__).parent / "fpscores.pkl.gz")
    model_path = tmp_path / "model.pkl"
    with model_path.open("wb") as handle:
        pickle.dump(Predictor(), handle)
    for target in cfg["targets"]:
        target["le_model"] = str(model_path)
    cfg = validate_config(cfg)
    ethane = Chem.MolFromSmiles("CC")
    conformer = Chem.Conformer(2)
    conformer.SetAtomPosition(1, (1.5, 0, 0))
    ethane.AddConformer(conformer)
    with patch("RL_utils.docking.sdf_to_pdbqt"), patch.object(scoring, "_run_vina", return_value=-10.):
        result = scoring._ScorerCore(cfg, "sft").score(ethane, tmp_path / "sft")
        assert not result["failed"] and len(result["property_components"]) == 4
        assert result["le"] == pytest.approx([1.4 * 2 / 6] * 2)
        assert result["reward"] == pytest.approx(.5 + .5 * sum(result["property_components"]) / 4)
        for target in cfg["targets"]:
            target["le_model"] = "not-needed-in-nft.pkl"
        result = scoring._ScorerCore(cfg, "nft").score(ethane, tmp_path / "nft")
        assert not result["failed"] and result["reward"] == 1.


def test_seeded_vina_command_and_timeout_are_forwarded(tmp_path):
    cfg = validate_config(base())
    finished = SimpleNamespace(returncode=0, stdout="   1   -9.5  0  0\n", stderr="")
    with patch.object(scoring.subprocess, "run", return_value=finished) as run:
        affinity = scoring._run_vina(tmp_path / "ligand.pdbqt", cfg["targets"][0],
                                    cfg["scoring"], 2024, tmp_path / "vina.log")
    command = run.call_args.args[0]
    assert command[command.index("--seed") + 1] == "2024"
    assert command[command.index("--cpu") + 1] == "1"
    assert run.call_args.kwargs["timeout"] == 300 and affinity == -9.5
