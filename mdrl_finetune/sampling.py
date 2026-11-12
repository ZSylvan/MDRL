"""Isolated native-DDPM sampler; fixed attempts and no trajectory allocation."""
from __future__ import annotations

import numpy as np
import torch
from .checkpoint import load_payload
from .training import require_finite

ATOMIC_NUMBERS = (6, 7, 8, 9, 15, 16, 17)


class _Attributes(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


def _attributes(value):
    if isinstance(value, dict):
        return _Attributes({key: _attributes(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_attributes(item) for item in value]
    return value


def check_native_config(config):
    if not isinstance(config, dict):
        raise ValueError("checkpoint must contain its native training config")
    try:
        model, chem, transform = config["model"], config["chem"], config["transform"]
        diff = model["diff"]
        correct = (model["name"] == "diffusion"
                   and diff["num_timesteps"] == 1000
                   and diff.get("categorical_space", "discrete") == "discrete"
                   and tuple(chem["atomic_numbers"]) == ATOMIC_NUMBERS
                   and tuple(chem["mol_bond_types"]) == (1, 2, 3, 4)
                   and transform["use_mask_node"] is True
                   and transform["use_mask_edge"] is True)
    except (KeyError, TypeError) as exc:
        raise ValueError("incomplete native checkpoint architecture") from exc
    if not correct:
        raise ValueError("requires the native discrete 1000-step seven-element masked MDRL")
    if any("condition" in key.lower() for key in model):
        raise ValueError("conditional model configs are not supported")
    return config


def load_native_model(checkpoint, device="cpu"):
    from models.model import MDRL

    payload = load_payload(checkpoint) if not isinstance(checkpoint, dict) else checkpoint
    config = check_native_config(payload["config"])
    model = MDRL(_attributes(config["model"]), num_node_types=8, num_edge_types=6).to(device)
    model.load_state_dict(payload["model"], strict=True)
    require_finite(model.state_dict(), "loaded model")
    return model, config


def placeholder(count, device):
    sizes = np.random.normal(24.923464980477522, 5.516291901819105, count).astype(np.int64)
    resampled = 0
    for index in range(count):
        while sizes[index] <= 0:
            sizes[index] = int(np.random.normal(24.923464980477522, 5.516291901819105))
            resampled += 1
    nodes, halves, edge_batches = [], [], []
    offset = 0
    for index, size in enumerate(sizes):
        edge = torch.triu_indices(int(size), int(size), offset=1, device=device)
        halves.append(edge + offset)
        nodes.append(torch.full((int(size),), index, dtype=torch.long, device=device))
        edge_batches.append(torch.full((edge.shape[1],), index, dtype=torch.long, device=device))
        offset += int(size)
    return sizes, torch.cat(nodes), torch.cat(halves, dim=1), torch.cat(edge_batches), resampled


@torch.no_grad()
def reverse_predictions(model, count, batch_node, halfedge_index, batch_halfedge):
    """Same native adjacent-step kernels, including the final t=0 draw."""
    from models.diffusion import log_sample_categorical

    _, node, log_node = model.node_transition.sample_init(len(batch_node))
    position = model.pos_transition.sample_init([len(batch_node), 3])
    _, edge, log_edge = model.edge_transition.sample_init(len(batch_halfedge))
    full_edge = torch.cat([halfedge_index, halfedge_index.flip(0)], dim=1)
    full_batch = torch.cat([batch_halfedge, batch_halfedge])
    if model.num_timesteps != 1000 or model.categorical_space != "discrete":
        raise ValueError("only native discrete 1000-step sampling is supported")
    for step in reversed(range(model.num_timesteps)):
        time = torch.full((count,), step, dtype=torch.long, device=batch_node.device)
        prediction = model(node, position, batch_node, torch.cat([edge, edge]),
                           full_edge, full_batch, time)
        require_finite(prediction, "sampling prediction")
        position = model.pos_transition.get_prev_from_recon(
            position, prediction["pred_pos"], time, batch_node)
        log_node = model.node_transition.q_v_posterior(
            torch.log_softmax(prediction["pred_node"], -1), log_node,
            time, batch_node, v0_prob=True)
        node = model.node_transition.onehot_encode(log_sample_categorical(log_node))
        log_edge = model.edge_transition.q_v_posterior(
            torch.log_softmax(prediction["pred_halfedge"], -1), log_edge,
            time, batch_halfedge, v0_prob=True)
        edge = model.edge_transition.onehot_encode(log_sample_categorical(log_edge))
    # Match the native sampling script's decode convention, NOT traj[-1].
    return prediction


def decode_prediction(node_logits, position, edge_logits, halfedge_index):
    """Retain the exact projected graph used for molecule reconstruction."""
    raw = dict(node_logits=node_logits.cpu(), position=position.cpu(),
               edge_logits=edge_logits.cpu(), halfedge_index=halfedge_index.cpu())
    row = dict(raw=raw, graph=None, rdmol=None, smiles=None, trainable=False,
               valid=False, error=None, requested_atoms=len(position))
    try:
        require_finite(raw, "terminal prediction")
    except FloatingPointError as exc:
        row["error"] = str(exc)
        return row
    node_type = raw["node_logits"].argmax(-1)
    keep = (node_type < 7).nonzero(as_tuple=False).flatten()
    row["retained_atom_indices"] = keep
    if not len(keep):
        row["error"] = "all atoms masked"
        return row
    count = len(keep)
    upper = torch.triu_indices(count, count, offset=1)
    adjacency = torch.zeros((len(node_type), len(node_type)), dtype=torch.long)
    chosen_edges = raw["edge_logits"].argmax(-1)
    chosen_edges = torch.where(chosen_edges <= 4, chosen_edges, torch.zeros_like(chosen_edges))
    adjacency[raw["halfedge_index"][0], raw["halfedge_index"][1]] = chosen_edges
    adjacency = adjacency + adjacency.T
    kinds = adjacency[keep[upper[0]], keep[upper[1]]]
    positions = raw["position"][keep].float()
    positions = positions - positions.mean(0, keepdim=True)
    graph = dict(node_type=node_type[keep], node_pos=positions,
                 halfedge_index=upper, halfedge_type=kinds)
    row.update(graph=graph, trainable=True, decoded_atoms=count)
    try:
        from rdkit import Chem
        from utils.reconstruct import reconstruct_from_generated_with_edges

        real = kinds > 0
        bonds = upper[:, real]
        decoded = dict(element=np.asarray(ATOMIC_NUMBERS)[graph["node_type"].numpy()],
                       atom_pos=positions.numpy(),
                       bond_index=torch.cat([bonds, bonds.flip(0)], dim=1).numpy(),
                       bond_type=torch.cat([kinds[real], kinds[real]]).numpy())
        molecule = reconstruct_from_generated_with_edges(decoded, check_validity=True, add_edge=None)
        if molecule.GetNumAtoms() != count:
            raise ValueError("reconstruction changed the atom count")
        for index, atom in enumerate(molecule.GetAtoms()):
            if atom.GetAtomicNum() != int(decoded["element"][index]):
                raise ValueError("reconstruction changed atom identity/order")
        reconstructed = torch.zeros_like(kinds)
        bond_classes = {1.0: 1, 2.0: 2, 3.0: 3, 1.5: 4}
        for index, (left, right) in enumerate(upper.T.tolist()):
            bond = molecule.GetBondBetweenAtoms(left, right)
            if bond is not None:
                reconstructed[index] = bond_classes.get(bond.GetBondTypeAsDouble(), -1)
        if not torch.equal(reconstructed, kinds):
            raise ValueError("reconstruction changed predicted bond classes/topology")
        if not np.allclose(molecule.GetConformer().GetPositions(), positions.numpy(), atol=1e-5):
            raise ValueError("reconstruction changed coordinates")
        row["formal_charges"] = [atom.GetFormalCharge() for atom in molecule.GetAtoms()]
        row["explicit_hydrogens"] = [atom.GetNumExplicitHs() for atom in molecule.GetAtoms()]
        row["smiles"] = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
        row["rdmol"] = molecule
        if len(Chem.GetMolFrags(molecule)) != 1:
            raise ValueError("disconnected molecule")
        row["valid"] = True
    except Exception as exc:
        row["error"] = type(exc).__name__ + ": " + str(exc)
    return row


@torch.no_grad()
def sample_native(model, count, *, batch_size=16, device="cpu"):
    model.eval()
    result = []
    for start in range(0, count, batch_size):
        number = min(batch_size, count - start)
        sizes, batch_node, halfedge_index, batch_halfedge, retries = placeholder(number, device)
        prediction = reverse_predictions(model, number, batch_node, halfedge_index, batch_halfedge)
        offset = 0
        for index, size in enumerate(sizes):
            edge_mask = batch_halfedge == index
            row = decode_prediction(prediction["pred_node"][batch_node == index],
                                    prediction["pred_pos"][batch_node == index],
                                    prediction["pred_halfedge"][edge_mask],
                                    halfedge_index[:, edge_mask] - offset)
            row["atom_count_resamples_in_batch"] = retries
            result.append(row)
            offset += int(size)
    if len(result) != count:
        raise RuntimeError("fixed-attempt sampler returned the wrong count")
    return result
