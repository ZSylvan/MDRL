"""Valid-only reward normalization and balanced invalid-example retention."""
from __future__ import annotations
import torch
from .training import require_finite, ema_update


def population_advantage(reward, eps=1e-4):
    reward = torch.as_tensor(reward, dtype=torch.float32)
    require_finite(reward, "reward")
    return (reward - reward.mean()) / (reward.std(unbiased=False) + eps)


def select_examples(rows, min_valid=2):
    rewards = torch.tensor([row["reward"] for row in rows], dtype=torch.float32)
    require_finite(rewards, "rollout reward")
    valid = torch.tensor([bool(row.get("valid")) and not row.get("failed", False) for row in rows])
    trainable = torch.tensor([bool(row.get("trainable")) and row.get("graph") is not None for row in rows])
    if bool((valid & ~trainable).any()):
        raise ValueError("a valid scored molecule has no trainable graph")
    nvalid = int(valid.sum())
    if nvalid < min_valid:
        return [], torch.empty(0), nvalid
    optimality = torch.zeros_like(rewards)
    optimality[valid] = 0.5 + 0.5 * population_advantage(rewards[valid]).clamp(-1, 1)
    invalid = ((~valid) & trainable).nonzero(as_tuple=False).flatten()[:nvalid]
    keep = valid.clone()
    keep[invalid] = True
    indices = keep.nonzero(as_tuple=False).flatten().tolist()
    return indices, optimality[keep], nvalid


def coordinate_mse(pred, target):
    return (pred - target).square().mean()


def coordinate_adaptive_weight(delta, floor=1e-5):
    return delta.detach().abs().mean(dim=-1).clamp_min(floor).reciprocal()


def anchor_loss(coords, ref_coords, atom_logits, ref_atom_logits, edge_logits, ref_edge_logits):
    result = coordinate_mse(coords, ref_coords) + coordinate_mse(atom_logits, ref_atom_logits)
    if edge_logits.numel():
        result = result + coordinate_mse(edge_logits, ref_edge_logits)
    return 0.001 * result
