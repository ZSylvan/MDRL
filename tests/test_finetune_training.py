"""CPU heads with real native diffusion kernels and the actual native loss body.

This does NOT instantiate the six-block KAN model or validate a real checkpoint.
"""
import ast
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from models.transition import ContigousTransition, GeneralCategoricalTransition
from mdrl_finetune.training import freeze_reference, ema_update, nft_step, sft_step

torch.set_num_threads(1)


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.position = torch.nn.Parameter(torch.tensor([0.2, 0.3, 0.4]))
        self.atom = torch.nn.Parameter(torch.arange(8).float() * 0.03)
        self.bond = torch.nn.Parameter(torch.arange(6).float() * 0.04)
        betas = np.full(1000, 0.01)
        self.pos_transition = ContigousTransition(betas)
        self.node_transition = GeneralCategoricalTransition(betas, 8, init_prob="tomask")
        self.edge_transition = GeneralCategoricalTransition(betas, 6, init_prob="absorb")
        self.num_timesteps, self.categorical_space, self.bond_len_loss = 1000, "discrete", False

    def sample_time(self, count, device):
        return torch.randint(0, 1000, (count,), device=device), None

    def forward(self, h_node_pert, pos_pert, batch_node, h_edge_pert, edge_index, batch_edge, t):
        return {"pred_pos": pos_pert * 0 + self.position,
                "pred_node": self.atom.expand(len(pos_pert), -1),
                "pred_halfedge": self.bond.expand(h_edge_pert.shape[0] // 2, -1)}


tree = ast.parse((Path(__file__).resolve().parents[1] / "models/model.py").read_text())
native = next(x for x in next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == "MDRL").body
              if isinstance(x, ast.FunctionDef) and x.name == "get_loss")
scope = {"torch": torch, "F": F}
exec(compile(ast.Module(body=[native], type_ignores=[]), "<native MDRL get_loss>", "exec"), scope)
Tiny.get_loss = scope["get_loss"]


def graph(count=2):
    return {"node_type": torch.zeros(count, dtype=torch.long),
            "node_pos": torch.arange(count).float()[:, None].expand(-1, 3).clone(),
            "halfedge_type": torch.ones(count * (count - 1) // 2, dtype=torch.long),
            "halfedge_index": torch.triu_indices(count, count, offset=1)}


def test_native_loss_body_updates_heads_not_diffusion_constants():
    model = Tiny()
    constants = {n: p.clone() for n, p in model.named_parameters() if not p.requires_grad}
    before = model.atom.detach().clone()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.01)
    losses = sft_step(model, optimizer, [graph()], updates=2, batch_size=1)
    assert all(np.isfinite(losses)) and not torch.equal(before, model.atom)
    assert all(torch.equal(value, dict(model.named_parameters())[name]) for name, value in constants.items())


def test_nft_uses_real_native_noise_and_preserves_reference():
    model = Tiny()
    old, ref = freeze_reference(model), freeze_reference(model)
    frozen = {key: value.clone() for key, value in ref.state_dict().items()}
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.01)
    before = model.atom.detach().clone()
    result = nft_step(model, old, ref, optimizer, [graph(), graph(3)], torch.tensor([1., 0.]),
                      epochs=2, micro_batch_size=1)
    assert not result["skipped"] and np.isfinite(result["loss"])
    assert not torch.equal(before, model.atom)
    assert all(p.grad is None for p in old.parameters())
    assert all(p.grad is None for p in ref.parameters())
    assert all(torch.equal(value, ref.state_dict()[name]) for name, value in frozen.items())
    ema_update(old, model, .001)
    assert all(torch.equal(value, old.state_dict()[name]) for name, value in frozen.items()
               if not dict(model.named_parameters())[name].requires_grad)
