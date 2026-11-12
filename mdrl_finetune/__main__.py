"""Explicit independent entry; never dispatches through native train/RL scripts."""
from __future__ import annotations
import argparse
from pathlib import Path
from .config import load_config, validate_config


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m mdrl_finetune")
    sub = parser.add_subparsers(dest="stage", required=True)
    for stage in ("sft", "nft"):
        command = sub.add_parser(stage)
        command.add_argument("--config", required=True, type=Path)
        command.add_argument("--output", type=Path)
        command.add_argument("--init-checkpoint", type=Path)
        command.add_argument("--resume", type=Path,
                             help="continue a complete boundary checkpoint into a NEW output directory")
        mode = command.add_mutually_exclusive_group()
        mode.add_argument("--schema-only", action="store_true",
                          help="validate YAML only; does not check server files or load a model")
        mode.add_argument("--validate-only", action="store_true",
                          help="load/check input models and scoring assets, without creating a run")
    args = parser.parse_args(argv)
    raw = load_config(args.config)
    if args.init_checkpoint:
        if args.stage != "nft":
            parser.error("--init-checkpoint is only valid for nft")
        raw.setdefault("nft", {})["init_checkpoint"] = str(args.init_checkpoint)
    cfg = validate_config(raw, args.stage)
    if args.schema_only:
        print("schema valid: " + args.stage)
        return 0
    if args.validate_only:
        from .runtime import preflight
        preflight(cfg, args.stage, resume=args.resume)
        print("inputs/model valid: " + args.stage)
        return 0
    if args.output is None:
        parser.error("--output is required for training")
    from .runtime import run_sft, run_nft
    runner = run_sft if args.stage == "sft" else run_nft
    runner(cfg, args.output, resume=args.resume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
