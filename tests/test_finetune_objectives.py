import pytest
import torch
from mdrl_finetune.objectives import population_advantage, anchor_loss, select_examples
from mdrl_finetune.pool import update_cumulative_pool


def test_population_std_and_raw_xyz_mean_anchor():
    out = population_advantage(torch.tensor([1., 2., 3.]))
    assert out.mean() == pytest.approx(0, abs=1e-6)
    zero = torch.zeros(2, 3)
    # Constant translations must be penalized: predictions are NOT recentered.
    assert anchor_loss(zero + 1, zero, zero[:, :1], zero[:, :1],
                       zero[:0, :1], zero[:0, :1]) == pytest.approx(.001)


def test_only_valid_rewards_define_advantage_and_invalids_are_balanced():
    rows = [dict(reward=r, valid=v, trainable=True, graph={})
            for r, v in ((.2, True), (-1, False), (.8, True), (-1, False), (-1, False))]
    selected, weights, valid = select_examples(rows)
    assert selected == [0, 1, 2, 3] and valid == 2
    assert weights.tolist() == pytest.approx([0., 0., 1., 0.], abs=.001)
    rows[0]["valid"] = False
    assert select_examples(rows)[0] == []


def test_pool_preserves_full_history_and_latest_duplicate():
    pool = update_cumulative_pool([], [dict(smiles="A", reward=1.), dict(smiles="B", reward=2.)])
    pool = update_cumulative_pool(pool, [dict(smiles="C", reward=3.)])
    assert len(pool) == 3
    pool = update_cumulative_pool(pool, [dict(smiles="C", reward=0.)])
    assert [row["smiles"] for row in pool] == ["B", "A", "C"]


def test_pool_ties_use_original_insertion_order_not_previous_ranking():
    pool = update_cumulative_pool([], [dict(smiles="A", reward=1.), dict(smiles="B", reward=2.)])
    pool = update_cumulative_pool(pool, [dict(smiles="A", reward=2.)])
    assert [row["smiles"] for row in pool] == ["A", "B"]
