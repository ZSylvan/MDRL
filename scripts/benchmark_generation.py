"""Independent MDRL/FlowMol generation benchmark.

Run one backend per process/environment, sequentially on the same GPU when a
comparison is desired.  This script performs generation only: no scoring,
docking, training, PBS submission, or checkpoint writes.  The timed quantity
is the existing sampler call (including graph preparation and reconstruction),
not an isolated denoiser kernel.  MDRL's native sampler also performs its
existing finite-value checks, so this is not a claim of inherent architecture
or hardware speed.

Example (separate environments/processes)::

    python scripts/benchmark_generation.py --backend mdrl --output-dir out/mdrl
    python scripts/benchmark_generation.py --backend flowmol --output-dir out/flowmol
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch


DEFAULT_MDRL_CHECKPOINT = "/home/zhuenqiang_1/Student_Project/zms/experiment_data/mdrl/pretrain/mdrl-b16-amp-resume-20261030T141531Z/training/config_20261030_224031/checkpoints/last.pt"
DEFAULT_FLOWMOL_ROOT = "/home/zhuenqiang_1/Student_Project/zms/project/flow-mol-condition"
DEFAULT_FLOWMOL_CONFIG = "/home/zhuenqiang_1/Student_Project/zms/data/flowmol/checkpoints/flowmol_condition/config.yaml"
DEFAULT_FLOWMOL_CHECKPOINT = "/home/zhuenqiang_1/Student_Project/zms/data/flowmol/checkpoints/flowmol_condition/checkpoints/epoch=epoch=1687-step=step=3305104-val=val_total_loss=0.2335.ckpt"
DEFAULT_TARGET_POOL = "/home/zhuenqiang_1/Student_Project/zms/data/flowmol/condition_assets/target_pool_multi.pt"
DEFAULT_HISTOGRAM = "/home/zhuenqiang_1/Student_Project/zms/data/flowmol/datasets/geom_full_kekulized/train_data_n_atoms_histogram.pt"
TARGET_IDS = ("O60885", "P24941")


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _sync(device: str) -> None:
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(torch.device(device))


def _memory_stats(device: str) -> Dict[str, int]:
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        index = torch.device(device).index or 0
        return {"allocated_bytes": int(torch.cuda.memory_allocated(index)),
                "reserved_bytes": int(torch.cuda.memory_reserved(index)),
                "max_allocated_bytes": int(torch.cuda.max_memory_allocated(index)),
                "max_reserved_bytes": int(torch.cuda.max_memory_reserved(index))}
    return {"allocated_bytes": 0, "reserved_bytes": 0, "max_allocated_bytes": 0, "max_reserved_bytes": 0}


def _reset_memory(device: str) -> None:
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        index = torch.device(device).index or 0
        torch.cuda.reset_peak_memory_stats(index)


def _dtype_names(model: Any) -> List[str]:
    return sorted({str(parameter.dtype) for parameter in model.parameters()})


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hardware(device: str) -> Dict[str, Any]:
    result = {"device": device, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        index = torch.device(device).index or 0
        result.update(cuda_name=torch.cuda.get_device_name(index),
                      cuda_capability=list(torch.cuda.get_device_capability(index)))
        try:
            result["cuda_uuid"] = str(torch.cuda.get_device_properties(index).uuid)
        except AttributeError:
            result["cuda_uuid"] = None
    return result


def _rdkit_row(molecule: Any, requested_atoms: int, native_valid: Optional[bool] = True) -> Dict[str, Any]:
    row: Dict[str, Any] = {"native_valid": (None if native_valid is None else bool(native_valid)), "requested_atoms": int(requested_atoms),
                           "valid": False, "connected": False, "error": None, "native_error": None, "smiles": None}
    try:
        from rdkit import Chem
        row["heavy_atoms"] = int(molecule.GetNumHeavyAtoms())
        row["total_atoms"] = int(molecule.GetNumAtoms())
        checked = Chem.Mol(molecule)
        Chem.SanitizeMol(checked)
        row["valid"] = True
        row["connected"] = len(Chem.GetMolFrags(checked)) == 1
        row["smiles"] = Chem.MolToSmiles(checked, canonical=True, isomericSmiles=True)
    except Exception as exc:
        row["error"] = type(exc).__name__ + ": " + str(exc)
        try:
            row["heavy_atoms"] = int(molecule.GetNumHeavyAtoms())
            row["total_atoms"] = int(molecule.GetNumAtoms())
        except Exception:
            row["heavy_atoms"] = None
            row["total_atoms"] = None
    return row


def _write_batch(handle: Any, rows: Iterable[Dict[str, Any]]) -> None:
    for row in rows:
        handle.write(json.dumps(row, allow_nan=False) + "\n")
    handle.flush()


def _prepare_output(path: Path) -> None:
    if path.exists():
        raise FileExistsError("output-dir must be a new, non-existing directory: " + str(path))
    path.mkdir(parents=True)


def _timed(callable_, device: str):
    _sync(device)
    started = time.perf_counter()
    value = callable_()
    _sync(device)
    return value, time.perf_counter() - started


def _run_mdrl(args: argparse.Namespace, output: Path) -> Dict[str, Any]:
    repo_root = str(Path(__file__).resolve().parents[1])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    setup_started = time.perf_counter()
    from mdrl_finetune.checkpoint import load_trusted_checkpoint
    from mdrl_finetune.sampling import load_native_model, sample_native

    payload = load_trusted_checkpoint(args.mdrl_checkpoint)
    if payload.get("iteration") != 985000:
        raise ValueError("MDRL checkpoint must have native iteration 985000")
    model, _ = load_native_model(payload, args.device)
    model.eval()
    _sync(args.device)
    setup_seconds = time.perf_counter() - setup_started
    _seed(args.seed)
    # Warmup is deliberately excluded, then all RNG streams are reset.
    _, warmup_seconds = _timed(lambda: sample_native(model, args.warmup_count, batch_size=args.batch_size, device=args.device), args.device)
    _seed(args.seed)
    _reset_memory(args.device)
    rows: List[Dict[str, Any]] = []
    batches = []
    from rdkit import Chem
    sdf = Chem.SDWriter(str(output / "molecules.sdf"))
    with (output / "attempts.jsonl").open("w", encoding="utf-8") as handle, (output / "batches.jsonl").open("w", encoding="utf-8") as batch_handle:
        for batch_index, start in enumerate(range(0, args.count, args.batch_size), 1):
            number = min(args.batch_size, args.count - start)
            native, elapsed = _timed(lambda: sample_native(model, number, batch_size=number, device=args.device), args.device)
            if len(native) != number:
                raise RuntimeError("MDRL sampler returned an incorrect batch length")
            batch_rows = []
            for offset, native_row in enumerate(native):
                requested = native_row.get("requested_atoms", native_row.get("decoded_atoms", 0))
                row = _rdkit_row(native_row["rdmol"], requested, native_row.get("valid", False)) if native_row.get("rdmol") is not None else {
                    "native_valid": bool(native_row.get("valid")), "requested_atoms": requested,
                    "valid": False, "connected": False, "error": native_row.get("error"),
                    "native_error": native_row.get("error"), "smiles": None,
                    "heavy_atoms": None, "total_atoms": None}
                row["native_error"] = native_row.get("error")
                row.update(attempt_index=start + offset, batch_index=batch_index)
                batch_rows.append(row)
                if native_row.get("rdmol") is not None:
                    try:
                        molecule = Chem.Mol(native_row["rdmol"])
                        molecule.SetIntProp("attempt_index", start + offset)
                        sdf.write(molecule)
                    except Exception as exc:
                        row["sdf_error"] = type(exc).__name__ + ": " + str(exc)
            _write_batch(handle, batch_rows)
            rows.extend(batch_rows)
            batches.append({"batch_index": batch_index, "count": number, "wall_seconds": elapsed})
            batch_handle.write(json.dumps(batches[-1], allow_nan=False) + "\n")
            batch_handle.flush()
            print(json.dumps(batches[-1], separators=(",", ":"), allow_nan=False), flush=True)
    sdf.close()
    if len(rows) != args.count:
        raise RuntimeError("MDRL sampler returned a count different from requested attempts")
    return {"rows": rows, "batches": batches, "parameter_dtypes": _dtype_names(model),
            "sampler": "mdrl_finetune.sampling.sample_native", "native_timesteps": 1000,
            "setup_wall_seconds": setup_seconds, "warmup_wall_seconds": warmup_seconds,
            "generation_wall_seconds": sum(item["wall_seconds"] for item in batches),
            "memory": _memory_stats(args.device)}


def _run_flowmol(args: argparse.Namespace, output: Path) -> Dict[str, Any]:
    setup_started = time.perf_counter()
    if args.flowmol_root:
        sys.path.insert(0, str(Path(args.flowmol_root)))
    from flowmol.atom_count import sample_n_atoms_from_flowmol3_histogram
    from flowmol.condition_bundle import load_condition_from_target_pool
    from flowmol.model_utils.condition_checkpoint import load_conditional_from_config

    config = {"model": {"config": args.flowmol_config, "initial_checkpoint": args.flowmol_checkpoint,
                         "statistics_dir": args.flowmol_data_dir},
              "condition": {"config": args.flowmol_config}}
    model = load_conditional_from_config(config, map_location=args.device, evaluation_only=True)
    model.to(args.device).eval()
    if not hasattr(model, "max_targets_num"):
        raise AttributeError("FlowMol model must expose max_targets_num")
    max_targets = int(model.max_targets_num)
    condition = load_condition_from_target_pool(args.flowmol_target_pool, TARGET_IDS,
                                                max_targets_num=max_targets, require_n_atoms=False)
    _sync(args.device)
    setup_seconds = time.perf_counter() - setup_started
    _seed(args.seed)
    warm_atoms, _ = sample_n_atoms_from_flowmol3_histogram(args.flowmol_histogram, args.warmup_count, device=args.device)
    warm_targets, warm_mask = condition.batched(args.warmup_count, args.device)
    _, warmup_seconds = _timed(lambda: model.sample(warm_atoms, targets=warm_targets, target_mask=warm_mask,
                 n_timesteps=args.flowmol_steps, device=args.device,
                 xt_traj=False, ep_traj=False), args.device)
    _seed(args.seed)
    _reset_memory(args.device)
    rows, batches = [], []
    input_prep_seconds = 0.0
    from rdkit import Chem
    sdf = Chem.SDWriter(str(output / "molecules.sdf"))
    with (output / "attempts.jsonl").open("w", encoding="utf-8") as handle, (output / "batches.jsonl").open("w", encoding="utf-8") as batch_handle:
        for batch_index, start in enumerate(range(0, args.count, args.batch_size), 1):
            number = min(args.batch_size, args.count - start)
            prep_started = time.perf_counter()
            n_atoms, _ = sample_n_atoms_from_flowmol3_histogram(args.flowmol_histogram, number, device=args.device)
            targets, mask = condition.batched(number, args.device)
            _sync(args.device)
            input_prep_seconds += time.perf_counter() - prep_started
            molecules, elapsed = _timed(lambda: model.sample(n_atoms, targets=targets, target_mask=mask,
                                                             n_timesteps=args.flowmol_steps, device=args.device,
                                                             xt_traj=False, ep_traj=False), args.device)
            if len(molecules) != number:
                raise RuntimeError("FlowMol sampler returned an incorrect batch length")
            batch_rows = []
            for offset, molecule in enumerate(molecules):
                rdkit_mol = getattr(molecule, "rdkit_mol", None)
                if rdkit_mol is None:
                    row = {"native_valid": None, "requested_atoms": int(n_atoms[offset].item()),
                           "valid": False, "connected": False, "error": "missing rdkit_mol",
                           "native_error": "missing rdkit_mol", "smiles": None,
                           "heavy_atoms": None, "total_atoms": None}
                else:
                    row = _rdkit_row(rdkit_mol, int(n_atoms[offset].item()), None)
                row.update(attempt_index=start + offset, batch_index=batch_index)
                batch_rows.append(row)
                if rdkit_mol is not None:
                    try:
                        molecule_copy = Chem.Mol(rdkit_mol)
                        molecule_copy.SetIntProp("attempt_index", start + offset)
                        sdf.write(molecule_copy)
                    except Exception as exc:
                        row["sdf_error"] = type(exc).__name__ + ": " + str(exc)
            _write_batch(handle, batch_rows)
            rows.extend(batch_rows)
            batches.append({"batch_index": batch_index, "count": number, "wall_seconds": elapsed})
            batch_handle.write(json.dumps(batches[-1], allow_nan=False) + "\n")
            batch_handle.flush()
            print(json.dumps(batches[-1], separators=(",", ":"), allow_nan=False), flush=True)
    sdf.close()
    if len(rows) != args.count:
        raise RuntimeError("FlowMol sampler returned a count different from requested attempts")
    return {"rows": rows, "batches": batches, "parameter_dtypes": _dtype_names(model),
            "sampler": "FlowMol.conditioned_model.sample", "flowmol_steps": args.flowmol_steps,
            "condition_provenance": condition.provenance(),
            "setup_wall_seconds": setup_seconds, "warmup_wall_seconds": warmup_seconds,
            "generation_wall_seconds": sum(item["wall_seconds"] for item in batches),
            "input_preparation_seconds": input_prep_seconds,
            "memory": _memory_stats(args.device)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("mdrl", "flowmol"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--warmup-count", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--mdrl-checkpoint", default=DEFAULT_MDRL_CHECKPOINT)
    parser.add_argument("--flowmol-root", default=DEFAULT_FLOWMOL_ROOT)
    parser.add_argument("--flowmol-config", default=DEFAULT_FLOWMOL_CONFIG)
    parser.add_argument("--flowmol-checkpoint", default=DEFAULT_FLOWMOL_CHECKPOINT)
    parser.add_argument("--flowmol-target-pool", default=DEFAULT_TARGET_POOL)
    parser.add_argument("--flowmol-histogram", default=DEFAULT_HISTOGRAM)
    parser.add_argument("--flowmol-data-dir", default=None)
    parser.add_argument("--flowmol-steps", type=int, default=250)
    args = parser.parse_args(argv)
    if args.count <= 0 or args.batch_size <= 0 or args.warmup_count <= 0:
        parser.error("count, batch-size, and warmup-count must be positive")
    if args.flowmol_steps < 2:
        parser.error("flowmol-steps must be at least 2")
    output = args.output_dir
    _prepare_output(output)
    started = time.time()
    result = _run_mdrl(args, output) if args.backend == "mdrl" else _run_flowmol(args, output)
    rows, batches = result.pop("rows"), result.pop("batches")
    summary = dict(backend=args.backend, count=args.count, batch_size=args.batch_size, seed=args.seed,
                   warmup_count=args.warmup_count, hardware=_hardware(args.device),
                   torch_version=torch.__version__, elapsed_wall_seconds=time.time() - started,
                   sampling_seconds=result.get("generation_wall_seconds"),
                   attempts_per_second=(args.count / result["generation_wall_seconds"]
                                        if result.get("generation_wall_seconds", 0) > 0 else None),
                   valid=sum(bool(row["valid"]) for row in rows),
                   connected=sum(bool(row["connected"]) for row in rows), **result)
    summary["valid_connected_per_second"] = (summary["connected"] / summary["sampling_seconds"]
                                             if summary.get("sampling_seconds", 0) > 0 else None)
    if args.backend == "mdrl":
        summary.update(checkpoint_path=args.mdrl_checkpoint, checkpoint_sha256=_sha256(args.mdrl_checkpoint),
                       atom_count_semantics="native sample_native draws positive atom counts internally",
                       precision="float32/no autocast requested; actual parameter dtypes recorded",
                       sampler_caveat="MDRL native sample_native includes per-step finite checks and native topology/reconstruction validation; timing is the online SFT sampler, not a denoiser-only benchmark.")
    else:
        summary.update(checkpoint_path=args.flowmol_checkpoint, checkpoint_sha256=_sha256(args.flowmol_checkpoint),
                       config_path=args.flowmol_config, histogram_path=args.flowmol_histogram,
                       config_sha256=_sha256(args.flowmol_config),
                       histogram_sha256=_sha256(args.flowmol_histogram), target_pool_path=args.flowmol_target_pool,
                       target_pool_sha256=_sha256(args.flowmol_target_pool),
                       atom_count_semantics="FlowMol3 GEOM full histogram sampled before each timed model.sample call; histogram I/O/hash excluded from sampler timing",
                       precision="float32/no autocast requested; actual parameter dtypes recorded",
                       sampler_caveat="FlowMol timing includes graph preparation and SampledMolecule/RDKit reconstruction; no claim of isolated denoiser speed.")
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"backend": args.backend, "sampling_seconds": summary.get("sampling_seconds"),
                      "attempts_per_second": summary.get("attempts_per_second")},
                     separators=(",", ":"), allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
