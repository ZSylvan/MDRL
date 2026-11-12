"""Independent CPU contract checks for the additive MDRL fine-tune runtime.

These tests deliberately avoid torch-geometric, Vina, native sampling, and
real checkpoints.  They exercise only tensor-level scientific contracts.
"""
import torch

from models.transition import GeneralCategoricalTransition
from mdrl_finetune.training import (
    freeze_reference,
    ema_update,
    nft_loss,
)
from mdrl_finetune.objectives import select_examples


def test_tomask_prior_retains_nonmask_support():
    betas = torch.tensor([0.2, 0.3]).numpy()
    transition = GeneralCategoricalTransition(betas, num_classes=4, init_prob="tomask")
    prior = transition.init_prob
    assert prior[-1] > 0.9
    assert (prior[:-1] > 0).all()


def test_reference_freeze_and_ema_leave_frozen_parameter_unchanged():
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(2.0)
    ref = freeze_reference(model)
    assert all(not p.requires_grad for p in ref.parameters())
    old = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        old.weight.fill_(0.0)
    model.weight.requires_grad_(False)
    ema_update(old, model, 0.5)
    assert torch.equal(old.weight, model.weight)


def test_nft_keeps_finite_failed_graphs_as_negative_examples():
    rows = [
        {"reward": 1.0, "valid": True, "trainable": True, "graph": {}},
        {"reward": -1.0, "valid": False, "trainable": True, "graph": {}},
    ]
    selected, optimality, nvalid = select_examples(rows, min_valid=1)
    assert len(selected) == 2 and optimality.numel() == 2 and nvalid == 1


def test_nft_update_depends_on_reward_ordering():
    batch = {
        "node_type": torch.tensor([0, 0]),
        "node_pos": torch.zeros(2, 3),
        "halfedge_type": torch.tensor([0]),
        "batch_node": torch.tensor([0, 1]),
        "batch_halfedge": torch.tensor([0]),
    }
    old = {"pred_pos": torch.zeros(2, 3), "pred_node": torch.zeros(2, 2),
           "pred_halfedge": torch.zeros(1, 2)}
    ref = {key: value.clone() for key, value in old.items()}
    pred = {"pred_pos": torch.ones(2, 3, requires_grad=True),
            "pred_node": torch.tensor([[1., 0.], [1., 0.]], requires_grad=True),
            "pred_halfedge": torch.tensor([[1., 0.]], requires_grad=True)}
    plus, _ = nft_loss(pred, old, ref, batch, torch.tensor([1.0, 0.0]))
    minus, _ = nft_loss(pred, old, ref, batch, torch.tensor([0.0, 1.0]))
    assert not torch.equal(plus.detach(), minus.detach())


def test_nft_loss_matches_explicit_endpoint_formula_and_has_gradient():
    """Pin the +/- endpoint objective, adaptive coordinate scale, and anchor."""
    batch = {
        "node_type": torch.tensor([0, 1]),
        "node_pos": torch.tensor([[0.0, 0.0, 0.0], [1.0, -1.0, 0.5]]),
        "halfedge_type": torch.tensor([1]),
        "batch_node": torch.tensor([0, 0]),
        "batch_halfedge": torch.tensor([0]),
    }
    old = {"pred_pos": torch.tensor([[0.2, 0.0, 0.0], [1.2, -1.0, 0.5]]),
           "pred_node": torch.tensor([[0.4, -0.2], [0.1, 0.3]]),
           "pred_halfedge": torch.tensor([[0.2, -0.1]])}
    ref = {key: value + 0.1 for key, value in old.items()}
    pred = {key: value.clone().requires_grad_() for key, value in {
        "pred_pos": torch.tensor([[0.8, 0.0, 0.0], [1.4, -1.0, 0.5]]),
        "pred_node": torch.tensor([[0.7, -0.4], [0.2, 0.8]]),
        "pred_halfedge": torch.tensor([[0.5, -0.3]]),
    }.items()}
    r = torch.tensor([0.75, 0.25])
    total, parts = nft_loss(pred, old, ref, batch, r, beta=0.5, anchor_coef=0.01)

    positive = {key: 0.5 * old[key] + 0.5 * pred[key] for key in pred}
    negative = {key: 1.5 * old[key] - 0.5 * pred[key] for key in pred}
    rn = r[batch["batch_node"]]
    scale = (positive["pred_pos"] - batch["node_pos"]).abs().mean(-1).clamp_min(1e-5)
    neg_error = negative["pred_pos"] - batch["node_pos"]
    neg_scale = neg_error.abs().mean(-1).clamp_min(1e-5)
    coord = (rn * (positive["pred_pos"] - batch["node_pos"]).square().sum(-1) / scale
             + (1-rn) * neg_error.square().sum(-1) / neg_scale).sum() / 2
    atom = (rn * torch.nn.functional.cross_entropy(positive["pred_node"], batch["node_type"], reduction="none")
            + (1-rn) * torch.nn.functional.cross_entropy(negative["pred_node"], batch["node_type"], reduction="none")).sum() / 2
    bond = (r[0] * torch.nn.functional.cross_entropy(positive["pred_halfedge"], batch["halfedge_type"], reduction="none")
            + (1-r[0]) * torch.nn.functional.cross_entropy(negative["pred_halfedge"], batch["halfedge_type"], reduction="none")).sum()
    anchor = (pred["pred_pos"] - ref["pred_pos"]).square().sum() / 6
    anchor += (pred["pred_node"] - ref["pred_node"]).square().sum() / 4
    anchor += (pred["pred_halfedge"] - ref["pred_halfedge"]).square().sum() / 2
    expected = 3 * coord + .4 * atom + 2 * bond + .01 * anchor
    assert torch.allclose(total, expected, atol=1e-6, rtol=1e-6)
    assert torch.allclose(parts["coord"], coord) and torch.allclose(parts["atom"], atom)
    total.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in pred.values())


def test_nft_loss_microbatch_sum_matches_global_denominators():
    """Per-microbatch losses add to the same normalized objective as one batch."""
    batch = {
        "node_type": torch.tensor([0, 1, 2]),
        "node_pos": torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 2., 0.]]),
        "halfedge_type": torch.tensor([1, 2]),
        "batch_node": torch.tensor([0, 0, 1]),
        "batch_halfedge": torch.tensor([0, 1]),
    }
    old = {"pred_pos": torch.zeros(3, 3), "pred_node": torch.zeros(3, 3), "pred_halfedge": torch.zeros(2, 5)}
    ref = {key: value.clone() for key, value in old.items()}
    pred = {"pred_pos": torch.tensor([[.2, 0., 0.], [1.2, 0., 0.], [0., 2.3, 0.]]),
            "pred_node": torch.tensor([[.1, .2, -.1], [.3, -.2, .4], [.2, .1, -.3]]),
            "pred_halfedge": torch.tensor([[.1, .2, .3, .4, -.1], [.2, -.1, .4, .1, .3]])}
    r = torch.tensor([.8, .2])
    full, _ = nft_loss(pred, old, ref, batch, r, anchor_coef=0.0)
    # Split on molecule boundary; use the same global node/edge denominators.
    def sliced(indexes, edge_indexes):
        return ({key: value[indexes if key != "pred_halfedge" else edge_indexes] for key, value in pred.items()},
                {key: value[indexes if key != "pred_halfedge" else edge_indexes] for key, value in old.items()},
                {key: value[indexes if key != "pred_halfedge" else edge_indexes] for key, value in ref.items()},
                {"node_type": batch["node_type"][indexes], "node_pos": batch["node_pos"][indexes],
                 "halfedge_type": batch["halfedge_type"][edge_indexes],
                 "batch_node": torch.zeros(len(indexes), dtype=torch.long),
                 "batch_halfedge": torch.zeros(len(edge_indexes), dtype=torch.long)},
                r[[0 if i < 2 else 1 for i in indexes]])
    p0, o0, f0, b0, r0 = sliced([0, 1], [0])
    p1, o1, f1, b1, r1 = sliced([2], [1])
    norm = {"nodes": 3, "edges": 2}
    part0, _ = nft_loss(p0, o0, f0, b0, r0, anchor_coef=0.0, normalization=norm)
    part1, _ = nft_loss(p1, o1, f1, b1, r1, anchor_coef=0.0, normalization=norm)
    assert torch.allclose(full, part0 + part1, atol=1e-6, rtol=1e-6)
