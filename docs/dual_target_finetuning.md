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
All accepted samples are retained, including duplicates and any overshoot. The
first update uses the initial population (approximately 3000), not a padded
4000-molecule set. Later updates use the previous round's selected top 4000.
The saved training population is the full pool before its 70/30 training and
validation split; both split indices are recorded in training checkpoints.

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

The templates use 16 workers, `docking.cpu: 1` and
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
must not already exist for a new run; choose a separate directory for each
independent run. Use `--resume` to continue a supported existing run, or
`--resume-if-exists` to continue when saved state exists and start otherwise.
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
Newly generated PBS files use `--resume-if-exists` and append to `launcher.log`.
After a job has stopped, resubmitting the same PBS continues its saved run.
Previously generated PBS files do not gain this behavior automatically. Never
run two jobs against the same output directory at the same time.

## Saved results and numerical failures

The output contains the source/resolved task configs, effective training and
sampling configs, generated scoring CSV, and one `round_000`, `round_001`, etc.
directory. Samples and model weights are kept separately:

```text
pretrained_985000.pt
round_000/
  generated/                  # Initial generated population (target 3000)
  population/                 # Scored initial population
round_001/                    # Same layout for every later update round
  training_population/        # Full input pool used by this round's training
  generated/                  # Only this round's newly generated samples (target 2000)
  population/                 # Selected top 4000 for the next update
  checkpoints/final.pt        # This round's final complete training checkpoint
  training/*/checkpoints/     # All validation checkpoints, round/iteration in names
  sampling/batches/           # Completed sampling batches for recovery
```

Every population directory contains numbered SDF files, `mol_summary.csv`,
`population.pt` (full molecule dictionaries for recovery) and a `manifest.json`
recording the actual count. Duplicate molecules have separate numbered files;
no deduplication or count padding is performed. Each training population is an
independent saved copy, not just a pointer to the previous selected population.
All round weights and intermediate checkpoints are retained without pruning.

Each round retains all raw and normalized per-molecule scores in `scores.jsonl`
and target-specific Vina output under `docking/mol_*/attempt_*/`. A retry uses a
new attempt directory rather than overwriting the previous docking files.
Indices in scores refer to that round's pre-selection population (old pool
followed by newly generated samples); the SMILES identifies the molecule after
re-ranking. Duplicate SMILES remain distinguishable by their molecule index.
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

## Resume behavior

Run the same task with `--resume`, for example:

```bash
python -B RL_utils/01_RL.py --config RL_utils/config_RL/tasks/chk1_auroraa.yml --resume
```

Use the task file belonging to the original run, including its original output
directory. Completed rounds and completed stages are reused. Recovery uses:

- Sampling: completed batch files and their random state; only an interrupted
  in-flight batch needs to be generated again.
- Scoring: completed per-molecule `scores.jsonl` records; only unfinished
  molecules are scored again. An unfinished trailing record is discarded.
- Training: the latest complete checkpoint in that round, including model,
  optimizer, scheduler, AMP scaler, iteration, split indices and random state.
  With the defaults, checkpoints are saved at iterations 5000 and 10000; any
  later unsaved updates are repeated. If no checkpoint exists, that round's
  training restarts, but its earlier completed stages remain saved.

The data-loader shuffle/prefetch position is not restored, so the continued
training is not guaranteed to be bit-for-bit identical to an uninterrupted run.
Normal round-to-round transitions still initialize only model parameters and
start fresh optimizers; full optimizer restoration is only for an interrupted
training round. The pinned `pretrained_985000.pt` is reused even if the original
source `last.pt` has since moved or changed.

Resume requires the saved `run_state.json` format and unchanged task/training/
sampling settings. Device and scoring CPU concurrency settings may change.
Legacy output directories without saved run state are not automatically
migrated. Diagnostic-only failure snapshots are never used for recovery.
