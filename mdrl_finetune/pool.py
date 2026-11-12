from __future__ import annotations
from collections import OrderedDict


def update_cumulative_pool(pool, scored_rows):
    """Deduplicate by isomeric SMILES, latest row wins, then stable score sort."""
    merged = OrderedDict((row["smiles"], row) for row in pool if row.get("smiles") and row.get("reward") is not None)
    next_order = max((row.get("_pool_order", -1) for row in pool), default=-1) + 1
    for row in scored_rows:
        if row.get("smiles") and row.get("reward") is not None:
            previous = merged.get(row["smiles"])
            order = previous["_pool_order"] if previous is not None else next_order
            if previous is None:
                next_order += 1
            merged[row["smiles"]] = dict(row, _pool_order=order)
    return sorted(merged.values(), key=lambda row: (-float(row["reward"]), row["_pool_order"]))
