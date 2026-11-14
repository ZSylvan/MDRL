"""Independent SFT/NFT runners with immutable-source, boundary-only resume."""
from __future__ import annotations

import json
from pathlib import Path
import random
import traceback
from datetime import datetime, timezone
import inspect
import numpy as np
import torch
import yaml
from .checkpoint import (FORMAT, atomic_save, capture_rng, file_digest, fingerprint,
                         load_boundary, load_payload, load_trusted_checkpoint,
                         restore_rng, write_json)
from .config import validate_config
from .objectives import select_examples
from .pool import update_cumulative_pool
from .training import ema_update, freeze_reference, nft_step, require_finite, sft_step


def _seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _optimizer(model, section):
    parameters = [p for p in model.parameters() if p.requires_grad]
    if not parameters:
        raise ValueError("model has no trainable parameters")
    return torch.optim.AdamW(parameters, lr=section["lr"], weight_decay=section["weight_decay"])


def _new_output(output):
    out = Path(output)
    if out.exists() or out.is_symlink():
        raise FileExistsError("output must be a NEW directory, including for --resume: " + str(out))
    return out


def preflight(config, stage, resume=None):
    """The real input checks, without creating an output or changing RNG state."""
    from .sampling import check_native_config, load_native_model
    from .scoring import ScoreEngine

    rng = capture_rng()
    try:
        cfg = validate_config(config, stage)
        payload = load_trusted_checkpoint(cfg["model"]["checkpoint"])
        native = check_native_config(payload["config"])
        digest = file_digest(cfg["model"]["checkpoint"])
        if resume:
            saved = load_boundary(resume, stage, cfg)
            if saved["source_digest"] != digest:
                raise ValueError("pretrained checkpoint content changed")
        if stage == "nft":
            source = load_payload(cfg["nft"]["init_checkpoint"])
            if (source.get("format") != FORMAT or source.get("stage") != "sft"
                    or source.get("completed") != 8 or source.get("diagnostic_only")
                    or source.get("config", {}).get("sft", {}).get("rounds") != 8
                    or source.get("config", {}).get("targets") != cfg["targets"]
                    or source.get("source_digest") != digest
                    or fingerprint(source.get("native_config")) != fingerprint(native)):
                raise ValueError("NFT requires a matching completed SFT8 checkpoint")
            payload = source
        model, _ = load_native_model(payload, "cpu")
        if resume:
            model.load_state_dict(saved["model"], strict=True)
        engine = ScoreEngine(cfg, stage)
        engine.close()
        return cfg
    finally:
        restore_rng(rng)


def _prepare(config, stage, output, model, scorer, resume):
    from .sampling import check_native_config, load_native_model
    from .scoring import ScoreEngine

    cfg = validate_config(config, stage)
    out = _new_output(output)
    _seed(cfg["seed"])
    pretrained = load_trusted_checkpoint(cfg["model"]["checkpoint"])
    native_config = check_native_config(pretrained["config"])
    source_digest = file_digest(cfg["model"]["checkpoint"])
    boundary = load_boundary(resume, stage, cfg) if resume else None
    if boundary and (boundary["source_digest"] != source_digest
                     or fingerprint(boundary["native_config"]) != fingerprint(native_config)):
        raise ValueError("pretrained checkpoint content/architecture changed")
    initial = pretrained
    if stage == "nft":
        initial = load_payload(cfg["nft"]["init_checkpoint"])
        if (initial.get("format") != FORMAT or initial.get("stage") != "sft"
                or initial.get("completed") != 8 or initial.get("diagnostic_only")
                or initial.get("config", {}).get("sft", {}).get("rounds") != 8
                or initial.get("config", {}).get("targets") != cfg["targets"]
                or initial.get("source_digest") != source_digest
                or fingerprint(initial.get("native_config")) != fingerprint(native_config)):
            raise ValueError("NFT requires a matching completed SFT8 checkpoint, not pretrained/RL weights")
        require_finite(initial["model"], "SFT8 initialization")
    if model is None:
        model, _ = load_native_model(initial, cfg["runtime"]["device"])
    else:
        model.load_state_dict(initial["model"], strict=True)
        model.to(cfg["runtime"]["device"])
    optimizer = _optimizer(model, cfg[stage])
    if boundary:
        model.load_state_dict(boundary["model"], strict=True)
        optimizer.load_state_dict(boundary["optimizer"])
    require_finite((model.state_dict(), optimizer.state_dict()), "initial state")
    if scorer is None:
        scorer = ScoreEngine(cfg, stage)
    # All model/asset checks precede creation of any run directory.
    out.mkdir(parents=True, exist_ok=False)
    (out / "config.resolved.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    write_json(out / "status.json", dict(status="RUNNING", stage=stage, resumed_from=str(resume) if resume else None))
    return cfg, out, model, optimizer, scorer, boundary, native_config, source_digest


def _checkpoint(cfg, model, optimizer, completed, native_config, source_digest, **extra):
    require_finite((model.state_dict(), optimizer.state_dict()), "checkpoint")
    return dict(format=FORMAT, stage=cfg["stage"], config=cfg,
                config_fingerprint=fingerprint(cfg), native_config=native_config,
                source_digest=source_digest, model=model.state_dict(), optimizer=optimizer.state_dict(),
                completed=completed, rng=capture_rng(), **extra)


def _round_directory(out, prefix, index):
    path = out / ("%s_%02d" % (prefix, index))
    path.mkdir(exist_ok=False)
    return path


def _save_rows(directory, rows):
    # Includes exact training graphs, failures, masks/maps and raw endpoint logits.
    atomic_save(directory / "samples.pt", rows)
    excluded = {"graph", "raw", "rdmol", "mol", "retained_atom_indices"}
    summary = [{key: value for key, value in row.items() if key not in excluded} for row in rows]
    write_json(directory / "scores.json", summary)
    from rdkit import Chem
    writer = Chem.SDWriter(str(directory / "generated.sdf"))
    try:
        for index, row in enumerate(rows):
            if row.get("rdmol") is not None:
                molecule = Chem.Mol(row["rdmol"])
                molecule.SetIntProp("attempt_index", index)
                writer.write(molecule)
    finally:
        writer.close()


def _record(out, metrics):
    with (out / "metrics.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(metrics, allow_nan=False) + "\n")
    print(json.dumps(metrics, allow_nan=False), flush=True)


def _progress_payload(stage, round_idx, phase, event=None, **values):
    payload = dict(timestamp=datetime.now(timezone.utc).isoformat(), stage=stage,
                   round=round_idx, human_round=round_idx + 1, phase=phase)
    if event is not None:
        payload["event"] = event
    payload.update(values)
    return payload


def _write_progress(out, round_dir, payload):
    """Atomically publish authoritative run and current-round progress."""
    write_json(out / "progress.json", payload)
    if round_dir is not None:
        write_json(round_dir / "sampling_progress.json", payload)


def _sample_with_progress(sample, model, attempts, batch_size, device, callback):
    # Injected samplers from older callers remain valid when they lack the new kwarg.
    try:
        accepts = "progress_callback" in inspect.signature(sample).parameters
    except (TypeError, ValueError):
        accepts = False
    kwargs = dict(batch_size=batch_size, device=device)
    if accepts:
        kwargs["progress_callback"] = callback
    return sample(model, attempts, **kwargs)


def _quality_metrics(rows):
    successful = [row for row in rows if row.get("valid") and not row.get("failed")]
    affinities = {}
    for row in successful:
        for result in row.get("docking", []):
            affinities.setdefault(result["target"], []).append(result["affinity"])
    return dict(n_chem_valid=sum(bool(row.get("valid")) for row in rows),
                scoring_success_fraction=len(successful) / len(rows),
                docking_means={name: sum(values) / len(values) for name, values in affinities.items()},
                qed_mean=sum(row["qed"] for row in successful) / len(successful)
                if successful and all("qed" in row for row in successful) else None,
                sa_mean=sum(row["sa"] for row in successful) / len(successful)
                if successful and all("sa" in row for row in successful) else None)


def _fail(out, stage, exc):
    write_json(out / "failure.json", dict(stage=stage, error=repr(exc), traceback=traceback.format_exc(),
                                        diagnostic_only=True))
    write_json(out / "status.json", dict(status="FAILED", stage=stage, error=repr(exc)))


def run_sft(config, output, *, model=None, sampler=None, scorer=None, resume=None):
    from .sampling import sample_native

    cfg, out, model, optimizer, scorer, saved, native_config, digest = _prepare(
        config, "sft", output, model, scorer, resume)
    section = cfg["sft"]
    pool = saved["pool"] if saved else []
    start = saved["completed"] if saved else 0
    if saved:
        restore_rng(saved["rng"])
    sample = sampler or sample_native
    state = None
    try:
        if start > section["rounds"]:
            raise ValueError("resume checkpoint exceeds requested SFT rounds")
        state = _checkpoint(cfg, model, optimizer, start, native_config, digest, pool=pool, round=start - 1)
        atomic_save(out / "checkpoints" / "last.pt", state)
        for round_idx in range(start, section["rounds"]):
            directory = _round_directory(out, "round", round_idx)
            attempts = section["initial_attempts"] if round_idx == 0 else section["round_attempts"]
            progress = {}
            def on_progress(event):
                progress.update(event)
                completed_valid = event.get("completed_valid")
                completed_invalid = event.get("completed_invalid")
                if completed_valid is None:
                    completed_valid = progress.get("completed_valid", 0)
                if completed_invalid is None:
                    completed_invalid = progress.get("completed_invalid", 0)
                progress.update(_progress_payload("sft", round_idx, "sampling",
                                                  event=event.get("event"),
                                                  total_attempts=event.get("total_attempts"),
                                                  completed_attempts=event.get("completed_attempts", 0),
                                                  completed_valid=completed_valid,
                                                  completed_invalid=completed_invalid,
                                                  batch_index=event.get("batch_index"),
                                                  total_batches=event.get("total_batches"),
                                                  current_batch_size=event.get("current_batch_size"),
                                                  steps_completed=event.get("steps_completed", 0),
                                                  steps_total=event.get("steps_total", 1000),
                                                  generation_fraction=(event.get("completed_attempts", 0) /
                                                                       event.get("total_attempts", 1)
                                                                       if event.get("total_attempts") else 0.0),
                                                  batch_denoising_fraction=event.get("batch_denoising_fraction", 0.0)))
                _write_progress(out, directory, progress)
            on_progress({"event": "start", "total_attempts": attempts,
                         "completed_attempts": 0, "completed_valid": 0,
                         "completed_invalid": 0, "batch_index": 0,
                         "total_batches": (attempts + section["generation_batch_size"] - 1) // section["generation_batch_size"],
                         "current_batch_size": 0, "steps_completed": 0,
                         "steps_total": 1000, "generation_fraction": 0.0})
            rows = _sample_with_progress(sample, model, attempts,
                                         section["generation_batch_size"],
                                         cfg["runtime"]["device"], on_progress)
            if len(rows) != attempts:
                raise ValueError("sampler must return exactly the requested ATTEMPTS")
            reconstruction_valid = sum(bool(row.get("valid")) for row in rows)
            reconstruction_invalid = attempts - reconstruction_valid
            _write_progress(out, directory, _progress_payload(
                "sft", round_idx, "scoring", total_attempts=attempts,
                completed_attempts=attempts,
                completed_valid=reconstruction_valid,
                completed_invalid=reconstruction_invalid,
                generation_fraction=1.0))
            rows = scorer.score_many(rows, directory)
            if len(rows) != attempts:
                raise ValueError("scorer changed attempt count")
            require_finite([row["reward"] for row in rows], "SFT reward")
            _write_progress(out, directory, _progress_payload(
                "sft", round_idx, "scoring_complete", total_attempts=attempts,
                completed_attempts=attempts, completed_valid=reconstruction_valid,
                completed_invalid=reconstruction_invalid,
                generation_fraction=1.0))
            _save_rows(directory, rows)
            successful = [row for row in rows if row.get("valid") and row.get("trainable")
                          and not row.get("failed") and row.get("smiles")]
            pool = update_cumulative_pool(pool, successful)
            selected = pool[:section["top_k"]]
            _write_progress(out, directory, _progress_payload(
                "sft", round_idx, "training", total_attempts=attempts,
                completed_attempts=attempts,
                completed_valid=reconstruction_valid,
                completed_invalid=reconstruction_invalid,
                generation_fraction=1.0))
            losses = sft_step(model, optimizer, [row["graph"] for row in selected],
                              updates=section["updates_per_round"], batch_size=section["batch_size"],
                              grad_clip=section["grad_clip"])
            atomic_save(directory / "retained.pt", selected)
            write_json(directory / "retained.json", [
                dict(smiles=row["smiles"], reward=row["reward"]) for row in selected])
            metrics = dict(round=round_idx, human_round=round_idx + 1, generated=attempts,
                           scoring_success=len(successful), history=len(pool), retained=len(selected),
                           loss_last=losses[-1], **_quality_metrics(rows))
            write_json(directory / "metrics.json", metrics)
            state = _checkpoint(cfg, model, optimizer, round_idx + 1, native_config, digest,
                                pool=pool, round=round_idx)
            atomic_save(directory / "checkpoint.pt", state)
            atomic_save(out / "checkpoints" / "last.pt", state)
            _record(out, metrics)
            _write_progress(out, directory, _progress_payload(
                "sft", round_idx, "round_complete", total_attempts=attempts,
                completed_attempts=attempts,
                completed_valid=reconstruction_valid,
                completed_invalid=reconstruction_invalid, generation_fraction=1.0))
        atomic_save(out / "final.pt", state)
        write_json(out / "status.json", dict(status="COMPLETED", stage="sft", completed=state["completed"]))
        _write_progress(out, None, _progress_payload(
            "sft", section["rounds"] - 1, "sft_complete",
            completed_rounds=section["rounds"], generation_fraction=1.0))
        return state
    except Exception as exc:
        _fail(out, "sft", exc)
        raise
    finally:
        if hasattr(scorer, "close"):
            scorer.close()


def run_nft(config, output, *, model=None, sampler=None, scorer=None, resume=None):
    from .sampling import sample_native

    cfg, out, model, optimizer, scorer, saved, native_config, digest = _prepare(
        config, "nft", output, model, scorer, resume)
    section = cfg["nft"]
    old, reference = freeze_reference(model), freeze_reference(model)
    completed = saved["completed"] if saved else 0
    successful_updates = saved["successful_updates"] if saved else 0
    top_k = saved["top_k"] if saved else []
    if saved:
        old.load_state_dict(saved["old"], strict=True)
        reference.load_state_dict(saved["reference"], strict=True)
        restore_rng(saved["rng"])
    sample = sampler or sample_native
    state = None
    try:
        if completed > section["iterations"]:
            raise ValueError("resume checkpoint exceeds requested NFT iterations")
        state = _checkpoint(cfg, model, optimizer, completed, native_config, digest,
                            old=old.state_dict(), reference=reference.state_dict(),
                            successful_updates=successful_updates, top_k=top_k)
        atomic_save(out / "checkpoints" / "last.pt", state)
        for iteration in range(completed, section["iterations"]):
            directory = _round_directory(out, "iteration", iteration)
            rows = sample(old, section["attempts"], batch_size=section["rollout_batch_size"],
                          device=cfg["runtime"]["device"])
            if len(rows) != section["attempts"]:
                raise ValueError("sampler must return exactly the requested ATTEMPTS")
            rows = scorer.score_many(rows, directory)
            if len(rows) != section["attempts"]:
                raise ValueError("scorer changed attempt count")
            _save_rows(directory, rows)
            indices, optimality, nvalid = select_examples(rows, section["min_valid"])
            reward = sum(row["reward"] for row in rows) / len(rows)
            training = dict(skipped=True)
            eta = None
            if indices:
                training = nft_step(
                    model, old, reference, optimizer, [rows[index]["graph"] for index in indices],
                    optimality, epochs=section["epochs"], micro_batch_size=section["micro_batch_size"],
                    grad_clip=section["grad_clip"], beta=section["beta"], anchor_coef=section["anchor_coef"])
                successful_updates += 1
                eta = min(0.001 * successful_updates, 0.5)
                ema_update(old, model, eta)
            require_finite((model.state_dict(), old.state_dict(), reference.state_dict(),
                            optimizer.state_dict()), "end of NFT iteration")
            valid_rewards = [row["reward"] for row in rows if row.get("valid") and not row.get("failed")]
            metrics = dict(iteration=iteration, reward_mean=reward, n_valid=nvalid,
                           reward_valid_mean=sum(valid_rewards) / len(valid_rewards) if valid_rewards else -1.0,
                           n_training=len(indices), n_training_invalid=len(indices) - nvalid if indices else 0,
                           successful_updates=successful_updates, ema_eta=eta,
                           metric_model="old_pre_update", checkpoint_models="post_update",
                           **_quality_metrics(rows), **training)
            write_json(directory / "metrics.json", metrics)
            path = out / "checkpoints" / ("iter%06d.pt" % iteration)
            candidates = top_k + [dict(iteration=iteration, reward_mean=reward, path=str(path.resolve()))]
            # Stable sorting retains earlier iterations on a tie.
            candidates = sorted(candidates, key=lambda item: item["reward_mean"], reverse=True)
            top_k = candidates[:10]
            state = _checkpoint(cfg, model, optimizer, iteration + 1, native_config, digest,
                                old=old.state_dict(), reference=reference.state_dict(),
                                successful_updates=successful_updates, top_k=top_k,
                                iteration=iteration, metrics=metrics)
            if any(item["path"] == str(path.resolve()) for item in top_k):
                atomic_save(path, state)
            atomic_save(out / "checkpoints" / "last.pt", state)
            for item in candidates[10:]:
                expired = Path(item["path"])
                # Never delete a parent run's retained checkpoint during a branch resume.
                if expired.parent.resolve() == (out / "checkpoints").resolve() and expired.is_file():
                    expired.unlink()
            if top_k and Path(top_k[0]["path"]).is_file():
                atomic_save(out / "checkpoints" / "best.pt", load_payload(top_k[0]["path"]))
            _record(out, metrics)
        atomic_save(out / "final.pt", state)
        write_json(out / "status.json", dict(status="COMPLETED", stage="nft", completed=state["completed"]))
        return state
    except Exception as exc:
        _fail(out, "nft", exc)
        raise
    finally:
        if hasattr(scorer, "close"):
            scorer.close()
