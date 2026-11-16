"""Independent fixed-attempt evaluation of a completed MDRL SFT8 checkpoint."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pair", choices=("brd4-cdk2", "thrombin-trypsin"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if min(args.count, args.batch_size, args.workers) <= 0 or args.seed < 0:
        parser.error("count, batch-size, workers must be positive; seed must be nonnegative")
    if args.output_dir.exists() or args.output_dir.is_symlink():
        parser.error("output-dir must be a new directory (existing results are never overwritten)")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = "1"

    import numpy as np
    import torch
    from rdkit import Chem
    from mdrl_finetune.checkpoint import FORMAT, file_digest, load_payload, write_json
    from mdrl_finetune.config import validate_config
    from mdrl_finetune.sampling import load_native_model, sample_native
    from mdrl_finetune.scoring import ScoreEngine

    torch.set_num_threads(1)
    state = load_payload(args.checkpoint)
    if (not isinstance(state, dict) or state.get("format") != FORMAT
            or state.get("stage") != "sft" or state.get("completed") != 8
            or state.get("diagnostic_only")
            or state.get("config", {}).get("sft", {}).get("rounds") != 8):
        raise ValueError("requires a non-diagnostic checkpoint after all 8 SFT rounds")
    cfg = validate_config(state["config"], "sft")
    expected = {"brd4-cdk2": ["BRD4", "CDK2"],
                "thrombin-trypsin": ["Thrombin", "Trypsin"]}[args.pair]
    targets = [target["id"] for target in cfg["targets"]]
    if targets != expected:
        raise ValueError("checkpoint targets do not match --pair: " + repr(targets))
    cfg["seed"] = args.seed
    cfg["scoring"].update(workers=args.workers, cpu=1)
    cfg["runtime"]["device"] = args.device
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    # Boundary checkpoints store native architecture separately from fine-tune config.
    model, _ = load_native_model(
        {"config": state["native_config"], "model": state["model"]}, args.device)
    engine = ScoreEngine(cfg, "sft")
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    try:
        write_json(out / "status.json", {"status": "RUNNING", "phase": "sampling"})
        write_json(out / "evaluation.json", dict(
            checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=file_digest(args.checkpoint),
            completed_sft_rounds=8, pair=args.pair, count=args.count,
            batch_size=args.batch_size, seed=args.seed, workers=args.workers,
            vina_cpu_per_worker=1, config=cfg,
            device=args.device, cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
            torch_version=torch.__version__))

        def progress(event):
            write_json(out / "progress.json", dict(event, phase="sampling"))

        started = time.perf_counter()
        rows = sample_native(model, args.count, batch_size=args.batch_size,
                             device=args.device, progress_callback=progress)
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        sampling_seconds = time.perf_counter() - started
        # Persist generated attempts before scoring, so a scoring failure loses no samples.
        torch.save(rows, out / "samples.pt")
        writer = Chem.SDWriter(str(out / "valid.sdf"))
        try:
            for index, row in enumerate(rows):
                if row.get("valid"):
                    molecule = Chem.Mol(row["rdmol"])
                    molecule.SetIntProp("attempt_index", index)
                    writer.write(molecule)
        finally:
            writer.close()
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
        write_json(out / "status.json", {"status": "RUNNING", "phase": "scoring"})
        started = time.perf_counter()
        rows = engine.score_many(rows, out)
        scoring_seconds = time.perf_counter() - started
        excluded = {"graph", "raw", "rdmol", "mol", "retained_atom_indices"}
        records = [dict(attempt_index=i, **{k: v for k, v in row.items() if k not in excluded})
                   for i, row in enumerate(rows)]
        write_json(out / "scores.json", records)

        def affinities(row):
            return {item["target"]: item["affinity"] for item in row.get("docking", [])}

        successful = [row for row in rows if row.get("valid") and not row.get("failed")
                      and all(isinstance(affinities(row).get(t), (int, float))
                              and math.isfinite(affinities(row)[t]) for t in targets)]
        means = {t: (sum(affinities(r)[t] for r in successful) / len(successful)
                     if successful else None) for t in targets}
        valid = sum(bool(row.get("valid")) for row in rows)
        summary = dict(
            pair=args.pair, attempts=len(rows), valid=valid, valid_fraction=valid / len(rows),
            dual_target_scoring_success=len(successful),
            valid_but_scoring_failed=valid - len(successful),
            affinity_mean_kcal_mol=means,
            joint_affinity_mean_kcal_mol=(sum(means.values()) / 2 if successful else None),
            sampling_seconds=sampling_seconds, scoring_seconds=scoring_seconds,
            validity_definition="native sample_native valid: reconstruction/topology checks and connected molecule",
            mean_population="same samples successfully scored on both targets; failures excluded, not zero-filled",
            replacement_sampling=False)
        with (out / "scores.csv").open("w", newline="", encoding="utf-8") as handle:
            fields = ["attempt_index", "smiles", "valid", "failed", "error", "reward"] + targets
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for index, row in enumerate(rows):
                record = {key: row.get(key) for key in fields if key not in targets}
                record.update(attempt_index=index, **affinities(row))
                writer.writerow(record)
        write_json(out / "summary.json", summary)
        write_json(out / "status.json", {"status": "COMPLETED"})
        print(json.dumps(summary, indent=2, allow_nan=False), flush=True)
    except Exception:
        write_json(out / "status.json", {"status": "FAILED", "error": traceback.format_exc()})
        raise
    finally:
        engine.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
