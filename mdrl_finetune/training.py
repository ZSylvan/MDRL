"""Native SFT and endpoint-NFT updates; no native training code is changed."""
from __future__ import annotations

import copy
import math
import torch
import torch.nn.functional as F


def require_finite(value, stage):
    if torch.is_tensor(value):
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError("non-finite " + stage)
    elif isinstance(value, dict):
        for key, item in value.items():
            require_finite(item, stage + "." + str(key))
    elif isinstance(value, (list, tuple)):
        for item in value:
            require_finite(item, stage)
    elif isinstance(value, float) and not math.isfinite(value):
        raise FloatingPointError("non-finite " + stage)


def _graph_batch(graphs, device):
    """Collate complete upper halfedges and center clean coordinates per graph."""
    if not graphs:
        raise ValueError("empty graph batch")
    nodes, positions, edges, types, node_batch, edge_batch = [], [], [], [], [], []
    offset = 0
    for index, graph in enumerate(graphs):
        node = torch.as_tensor(graph["node_type"], dtype=torch.long)
        pos = torch.as_tensor(graph["node_pos"], dtype=torch.float32)
        halfedge = torch.as_tensor(graph["halfedge_index"], dtype=torch.long)
        kind = torch.as_tensor(graph["halfedge_type"], dtype=torch.long)
        count = node.numel()
        expected = torch.triu_indices(count, count, offset=1, device=halfedge.device)
        if (count == 0 or node.ndim != 1 or pos.shape != (count, 3)
                or not torch.equal(halfedge, expected) or kind.shape != (expected.shape[1],)
                or bool(((node < 0) | (node > 6)).any())
                or bool(((kind < 0) | (kind > 4)).any())):
            raise ValueError("invalid seven-element/complete-halfedge clean graph")
        require_finite(pos, "clean coordinates")
        nodes.append(node)
        positions.append(pos - pos.mean(dim=0, keepdim=True))
        edges.append(halfedge + offset)
        types.append(kind)
        node_batch.append(torch.full((count,), index, dtype=torch.long, device=node.device))
        edge_batch.append(torch.full((kind.numel(),), index, dtype=torch.long, device=kind.device))
        offset += count
    return {
        "node_type": torch.cat(nodes).to(device), "node_pos": torch.cat(positions).to(device),
        "batch_node": torch.cat(node_batch).to(device),
        "halfedge_type": torch.cat(types).to(device),
        "halfedge_index": torch.cat(edges, dim=1).to(device),
        "batch_halfedge": torch.cat(edge_batch).to(device), "num_mol": len(graphs),
    }


def _finish_step(model, optimizer, grad_clip):
    parameters = [p for p in model.parameters() if p.requires_grad]
    for parameter in parameters:
        if parameter.grad is not None:
            require_finite(parameter.grad, "gradient before clipping")
    norm = torch.nn.utils.clip_grad_norm_(parameters, grad_clip)
    require_finite(norm, "clipped gradient norm")
    optimizer.step()
    require_finite(model.state_dict(), "updated model")
    require_finite(optimizer.state_dict(), "optimizer state")
    return float(norm)


def sft_step(model, optimizer, graphs, *, updates=100, batch_size=16, grad_clip=1.0):
    if not graphs:
        raise ValueError("SFT requires retained, successfully scored graphs")
    model.train()
    losses = []
    for step in range(updates):
        start = (step * batch_size) % len(graphs)
        selected = [graphs[(start + i) % len(graphs)]
                    for i in range(min(batch_size, len(graphs)))]
        batch = _graph_batch(selected, next(model.parameters()).device)
        optimizer.zero_grad(set_to_none=True)
        # Includes native posterior KL/NLL multipliers and optional bond length.
        result = model.get_loss(**batch)
        require_finite(result, "native SFT loss")
        result["loss"].backward()
        _finish_step(model, optimizer, grad_clip)
        losses.append(float(result["loss"].detach()))
    return losses


def freeze_reference(model):
    reference = copy.deepcopy(model).eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    return reference


@torch.no_grad()
def ema_update(old, theta, eta):
    old_parameters, theta_parameters = dict(old.named_parameters()), dict(theta.named_parameters())
    if old_parameters.keys() != theta_parameters.keys():
        raise ValueError("EMA parameter mismatch")
    for name, parameter in theta_parameters.items():
        if parameter.requires_grad:
            old_parameters[name].mul_(eta).add_(parameter, alpha=1.0 - eta)
        else:
            old_parameters[name].copy_(parameter)
    old_buffers = dict(old.named_buffers())
    for name, buffer in theta.named_buffers():
        old_buffers[name].copy_(buffer)
    require_finite(old.state_dict(), "EMA model")


def prepare_noisy(model, batch):
    """One native q draw, reused unchanged for theta, old and reference."""
    time, _ = model.sample_time(batch["num_mol"], batch["node_pos"].device)
    position = model.pos_transition.add_noise(batch["node_pos"], time, batch["batch_node"])
    node = model.node_transition.add_noise(batch["node_type"], time, batch["batch_node"])[0]
    edge = model.edge_transition.add_noise(batch["halfedge_type"], time, batch["batch_halfedge"])[0]
    return dict(
        h_node_pert=node, pos_pert=position, batch_node=batch["batch_node"],
        h_edge_pert=torch.cat([edge, edge]),
        edge_index=torch.cat([batch["halfedge_index"], batch["halfedge_index"].flip(0)], dim=1),
        batch_edge=torch.cat([batch["batch_halfedge"], batch["batch_halfedge"]]), t=time,
    )


def nft_loss(pred, old, reference, batch, optimality, *, beta=1.0,
             anchor_coef=0.001, normalization=None):
    """Implicit +/- endpoint regression; the anchor is output MSE, not KL."""
    nodes, edges = batch["node_type"].numel(), batch["halfedge_type"].numel()
    denominators = normalization or {"nodes": nodes, "edges": edges}
    nden, eden = max(int(denominators["nodes"]), 1), max(int(denominators["edges"]), 1)
    rnode, redge = optimality[batch["batch_node"]], optimality[batch["batch_halfedge"]]
    positive = {key: (1 - beta) * old[key].detach() + beta * pred[key] for key in pred}
    negative = {key: (1 + beta) * old[key].detach() - beta * pred[key] for key in pred}

    def coordinate_term(value):
        error = value - batch["node_pos"]
        scale = error.abs().mean(dim=-1).clamp_min(1e-5).detach()
        return error.square().sum(dim=-1) / scale

    coord = (rnode * coordinate_term(positive["pred_pos"])
             + (1 - rnode) * coordinate_term(negative["pred_pos"])).sum() / nden
    atom = (rnode * F.cross_entropy(positive["pred_node"], batch["node_type"], reduction="none")
            + (1 - rnode) * F.cross_entropy(negative["pred_node"], batch["node_type"], reduction="none")).sum() / nden
    bond = pred["pred_halfedge"].sum() * 0
    if edges:
        bond = (redge * F.cross_entropy(positive["pred_halfedge"], batch["halfedge_type"], reduction="none")
                + (1 - redge) * F.cross_entropy(negative["pred_halfedge"], batch["halfedge_type"], reduction="none")).sum() / eden
    anchor = (pred["pred_pos"] - reference["pred_pos"].detach()).square().sum() / (nden * 3)
    anchor = anchor + (pred["pred_node"] - reference["pred_node"].detach()).square().sum() / (nden * pred["pred_node"].shape[-1])
    if edges:
        anchor = anchor + (pred["pred_halfedge"] - reference["pred_halfedge"].detach()).square().sum() / (eden * pred["pred_halfedge"].shape[-1])
    total = 3.0 * coord + 0.4 * atom + 2.0 * bond + anchor_coef * anchor
    return total, {"coord": coord, "atom": atom, "bond": bond, "anchor": anchor}


def nft_step(model, old, reference, optimizer, graphs, optimality, *, epochs=4,
             micro_batch_size=64, grad_clip=1.0, beta=1.0, anchor_coef=0.001):
    if not graphs or len(graphs) != len(optimality):
        raise ValueError("NFT graphs and optimality must be nonempty and aligned")
    normalization = {
        "nodes": sum(len(graph["node_type"]) for graph in graphs),
        "edges": sum(len(graph["halfedge_type"]) for graph in graphs),
    }
    device = next(model.parameters()).device
    optimality = torch.as_tensor(optimality, dtype=torch.float32, device=device)
    require_finite(optimality, "optimality")
    if bool(((optimality < 0) | (optimality > 1)).any()):
        raise ValueError("optimality must lie in [0, 1]")
    model.train()
    old.eval()
    reference.eval()
    last = {}
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        sums = {key: 0.0 for key in ("loss", "coord", "atom", "bond", "anchor")}
        for start in range(0, len(graphs), micro_batch_size):
            chunk = graphs[start:start + micro_batch_size]
            batch = _graph_batch(chunk, device)
            noisy = prepare_noisy(model, batch)
            require_finite(noisy, "noisy training batch")
            prediction = model(**noisy)
            with torch.no_grad():
                previous, fixed = old(**noisy), reference(**noisy)
            require_finite((prediction, previous, fixed), "NFT predictions")
            loss, components = nft_loss(
                prediction, previous, fixed, batch, optimality[start:start + len(chunk)],
                beta=beta, anchor_coef=anchor_coef, normalization=normalization)
            require_finite(loss, "NFT loss")
            loss.backward()
            sums["loss"] += float(loss.detach())
            for key, value in components.items():
                sums[key] += float(value.detach())
        _finish_step(model, optimizer, grad_clip)
        require_finite(reference.state_dict(), "frozen reference")
        last = sums
    return dict(last, skipped=False)
