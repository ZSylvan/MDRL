# Config-driven MDRL dual-target fine-tuning

## Tasks

- `RL_utils/config_RL/tasks/chk1_auroraa.yml`
- `RL_utils/config_RL/tasks/brd4_cdk2.yml`
- `RL_utils/config_RL/tasks/thrombin_trypsin.yml`

Each task initializes independently from the checked 985000-step checkpoint.
The templates point to the verified original `last.pt`. Set `initial_checkpoint`
to a fixed server archive if one is available. The entry rejects an unexpected
checkpoint iteration or mismatched model/chemistry/transform settings. Only model
parameters initialize fine-tuning; the pretraining optimizer is not resumed.
After validation, the loaded checkpoint is pinned as `pretrained_985000.pt` in
the new output directory. Sampling and the first update use this snapshot, so a
later overwrite of the source `last.pt` cannot change the initialized model.

Relative input paths are relative to the task YAML. Target receptors, ligand
models, docking centers, box dimensions and score normalization are supplied by
that YAML. Resources under `data/flowmol` are reused as files only; no FlowMol,
SFT or DiffusionNFT training code is used. Vina must be installed in MDRL, not
borrowed from the `mol` environment.

## Preserved method

The entry first requests 3000 connected molecules and scores them. Each update
round fine-tunes on the previous population, requests 2000 new molecules,
rescoring both old and new molecules, and retains the top 4000. As in the original
`range(1, n_epochs)`, `n_epochs: 50` means an initial population and 49 update
rounds. The original generated-pool 70/30 split, fresh optimizer each round,
10000 updates per round, and validation every 5000 updates are retained.

The templates retain training batch 24 and sampling batch 512. These are NOT
memory-tested settings for the available cards. Edit `train_overrides.batch_size`,
`train_overrides.num_workers` and `sample_overrides.batch_size` explicitly before
running. All other training settings remain in `train_MDRL_RL.yml`, including
the original learning rate and scheduler settings. Sampling retains replacement
sampling and the original output-path-dependent seed rule; it can overshoot the
requested population size. It is not the fixed-attempt inference probe.

Each molecule's property score is the arithmetic mean of normalized QED, SA and
the two ligand-efficiency predictions. The docking score is the mean of the two
normalized affinities. The final reward is their equal-weight average. The
original ligand-efficiency formula, including its hydrogen-count denominator,
is deliberately unchanged; changing that formula requires a separate decision.

## Parallel scoring and CPU control

Properties, ligand-efficiency prediction and docking are parallelized across
molecules using `docking.workers` independent processes. Each worker loads the
CPU scoring models once. The two targets for one molecule are docked sequentially.
Processes use `spawn` and do not inherit the training process's CUDA state.
Generation and training stay sequential in the main process on one selected GPU.

The templates use 4 workers, `docking.cpu: 1` and
`scoring.predictor_threads: 1`. Choose these yourself for the CPU allocation;
budget approximately `workers * max(cpu, predictor_threads)` active scoring
threads. Training data-loader workers are a separate setting. No hostname,
GPU-occupancy or scheduler-resource checks are imposed. `device` is a logical
Torch device; physical GPU selection remains under your control.

Vina's executable, exhaustiveness, mode count and optional timeout are configured
under `docking`. `timeout: null` retains no time limit; a positive number is in
seconds. A failed conversion, prediction, docking command or score stops the
task and records the molecule, rather than inventing a reward. Queued work is
cancelled on failure; already-running docking calls finish before the worker
pool shuts down. A finite timeout can bound that wait.

## Entry and checks

From the repository root, with the MDRL environment active:

```bash
python -B RL_utils/01_RL.py --config RL_utils/config_RL/tasks/chk1_auroraa.yml --check-config
```

This checks configuration values and input existence, but does not load models,
create output directories, generate molecules or dock. It does not establish
that the selected batch sizes fit GPU memory or that a receptor is biologically
correct. Replace the task filename for the other pairs.

The training entry uses the same command without `--check-config`. Optional
`--device` and `--output` override those two task settings. The output directory
must not already exist; choose a separate directory for each independent run.
No experiment is started by importing the entry.

## PBS preparation on the server

After committing and pulling the code, use `scripts/prepare_rl_pbs.py` from the
repository root. Repeat `--config` for the three tasks. Supply your own queue,
`--resources` and `--gpu`; the latter sets the physical GPU mask while the task
uses logical `cuda:0`. The generated PBS contains no hostname, GPU-occupancy or
repository-cleanliness gate. It does not reserve a GPU by itself.

Optional `--train-batch-size`, `--sample-batch-size`, `--data-workers`, `--workers`,
`--vina-cpu` and `--predictor-threads` change only the generated task copies.
Without these options, the task template values are preserved. A sampling batch
of 1 matches the tested inference batch but is slow; a training batch of 16
matches the selected pretraining batch. Neither choice guarantees that every
future fine-tuning batch fits GPU memory.

The generator checks input paths without loading models or starting sampling,
training or docking. Each new run ID gets its own directory under `finetune`,
with one `task.yml`, `job.pbs` and `launcher.log` location per task. Actual RL
output goes into each task's `results` subdirectory. Existing run IDs are
refused. A `protocol.json` records the preparation commit and resource request;
the launcher also logs its actual commit. Do not edit/pull the repository while
jobs are running, since the code itself is not copied into the run directory.

PBS files are prepared only; no job is submitted. Inspect the generated task
and resource settings before using `qsub`. Jobs configured for the same GPU
should be submitted sequentially, or edited to use separately chosen resources.

## Saved results and numerical failures

The output contains the source/resolved task configs, effective training and
sampling configs, generated scoring CSV, and one `round_000`, `round_001`, etc.
directory. Each round retains all raw and normalized per-molecule scores in
`scores.jsonl`, target-specific Vina output under `docking/mol_*/`, and selected
SDFs plus `population/mol_summary.csv`. Indices in scores refer to that round's
pre-selection population; the SMILES identifies the molecule after re-ranking.
`rounds.jsonl` records each completed round's source/result checkpoint and pool.

Training logs contain `train.jsonl`, `validation.jsonl`, AMP skipped updates and
dataset split indices. Checkpoints retain model/config plus optimizer, scheduler,
scaler, local iteration and round identity; their filenames include round and
iteration. The next round still uses the last checkpoint, not a new "best" rule.

Nonfinite forward losses stop training. AMP gradient overflow retains the existing
native skipped-update behavior and is recorded. Invalid AMP scale, unsafe gradient
updates or invalid checkpoint parameters stop the task. The first fatal training
failure saves `diagnostics/first_failure.pt` with the batch, losses, RNG state,
model/optimizer/scaler and gradients. It is diagnostic-only, not a resume input.

Whole-RL restart/resume is NOT implemented: an existing output directory is
refused. Checkpoints can be independently evaluated, but this must not be confused
with restoration of the population, scoring stage and overall RL round. Adding
round-level restart is a separate follow-up; copying a pretraining resume flag is
not sufficient.
