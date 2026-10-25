# Pretraining reproduction

This stage trains the original GEOM diffusion model from scratch. It does not
run pair-specific reinforcement learning or independent molecule evaluation.

## Configuration

`configs/train/train_MDRL_reproduction.yml` differs from the original configuration
in its external dataset root, output root (`logdir`) and the user-approved batch
size **8 instead of 24**. The original model, seed (2023), worker count (16),
2,500,000 iterations, validation frequency (5,000), optimizer, learning rate,
scheduler, AMP, transforms and split are retained. No gradient accumulation is
introduced. At the same iteration target, the nominal number of training molecules
is one-third of the original batch-24 configuration; this is not an equal-budget
reproduction. Validation still uses the original unweighted mean of batch losses,
now with batch 8 as well. The original YAML is unchanged.

Batch 8 reached about 99.1% estimated GPU usage in the returned experiment-001
stress test on a single V100 32 GB. The user selected 8 despite the OOM risk;
these short tests do not guarantee that long training will fit. Neither training
RuntimeError handling nor validation failure behavior is changed.

The existing `--logdir` argument overrides the configuration's output root. With
neither specified, the original `logs` default is retained. Each invocation creates
a new timestamped directory; a name collision raises an error rather than reusing
an existing directory. Run committed code from the repository root. `run.json`
records the Git commit and dirty status, not a full snapshot of uncommitted code.

The TensorBoard fix explicitly imports `distutils.version` before TensorBoard;
the server's existing Python 3.8 / PyTorch 1.10.1 environment passed this import
test. The optional `torchsummary` import/call is removed; the parameter-count print
is retained. No dependency upgrade is required for these two changes.

## Artifacts

Each run retains its copied YAML, arguments, `log.txt`, TensorBoard events and:

- `run.json`: run ID, code/environment metadata, status, counters and best/last references.
- `validation.jsonl`: all original batch-averaged validation losses, iteration,
  best-selection flag and checkpoint reference. Nonfinite metrics are stored as
  `null` with their names listed in `nonfinite_metrics`.
- `failures.jsonl`: skipped training RuntimeErrors and fatal/interrupted failures,
  with iteration, phase, error and traceback. It is empty when none occur.
- `checkpoints/<iteration>.pt`: the original periodic checkpoint schedule, retaining
  only the latest 50 saves. Older periodic files from this run are deleted.
- `checkpoints/best.pt`: lowest **finite validation total loss**, first occurrence
  on ties; no best checkpoint is selected if all validation losses are nonfinite.
- `checkpoints/last.pt`: latest completely saved validation checkpoint. Normal
  completion includes the final configured iteration. On interruption/failure it
  is the previous saved validation state, not partially updated in-memory weights;
  before the first validation it does not exist.

There are at most 52 completed `.pt` files per run: 50 periodic files plus
`best.pt` and `last.pt`. Best/last are independent copies, so pruning a periodic
file never deletes the selected best model. The best reference points to
`best.pt`; `source_checkpoint` records its original periodic filename even if
that file has been pruned. All validation metrics and failure logs are retained.
`run.json` lists the retained periodic files and the limit; `validation.jsonl`
records pruned filenames at each successful save. The checkpoint/validation
frequency and the best-selection rule are unchanged.

Checkpoints retain the original `config`, `model`, `optimizer`, `scheduler` and
`iteration` fields, plus AMP scaler state, selection evidence, skip counters and
Python/NumPy/Torch CPU/CUDA random-number-generator (RNG) states.
The new periodic file is fully staged in a temporary file before the oldest
periodic file is removed to free its slot, then atomically renamed. Best/last
are also replaced only after a complete temporary copy. Temporary files require
extra disk space (about one additional checkpoint during a normal save) and are
not included in the 52 completed-file limit. A write failure can leave `.tmp`
files or a periodic checkpoint without updated aliases; these are retained for
diagnosis, not treated as a completed save.

The original RuntimeError skip policy is unchanged. `skipped_iterations` counts
caught training RuntimeErrors; `completed_train_calls` counts training calls that
returned normally, not proven optimizer updates (AMP can skip an update). Normal
loop termination with skips is `completed_with_skips`, not `completed`. Fatal
errors and keyboard interrupts are recorded and re-raised for nonzero exit status.
Forced kills, node loss or power loss cannot be caught; `running` then means the
last persisted state, not proof that a job is still alive. Preserve PBS stdout,
stderr and scheduler exit information as well. Existing per-iteration training
metrics remain in `log.txt` and TensorBoard.

## Lightweight local checks

From the repository root, with the existing local Python (PyTorch and PyYAML):

```powershell
python -B -m unittest discover -s tests -p 'test_*.py' -v
```

These checks exercise checkpoint serialization/selection/retention, failure records, the
training loop with simulated calls, Python 3.8 syntax, import order and configuration
differences limited to approved adaptations. They do **not** execute the MDRL model, verify CUDA/PyG compatibility
or establish that a full training job will succeed. No local formal experiment,
server submission, new dependency installation or automatic data processing is
part of these checks. The server-side model/data smoke test and PBS resources
still need confirmation before formal pretraining.

## Resume

Always pass `--config configs/train/train_MDRL_reproduction.yml` for this
reproduction; the script's original default YAML still uses batch 24. A fresh
run has no `--resume` argument. To resume, add the following argument to the
training command **inside a server PBS job**, replacing the example path with
the actual trusted checkpoint:

```bash
--resume /absolute/path/to/previous-run/checkpoints/last.pt
```

This restores model, optimizer (including its learning rate and moments), scheduler,
AMP scaler, iteration, cumulative train/skip counters and the historical best.
Training starts at saved iteration + 1. `max_iters` remains the **total target**,
not an additional number of iterations, and must exceed the saved iteration.
The original validation cadence is retained, including validation at the final
target. Checkpoints are still saved only after validation every 5,000 iterations
and at the final target, not on every step or immediately on interruption. Work
since the last complete save must be repeated; no resume is possible before the
first save.

Each resume creates a **new timestamped output directory** under `--logdir` (or
the configured root), never appending to or pruning the parent run. `run.json`
records the source path/run/iteration and whether RNG state was available.
The source checkpoint is copied to the new `last.pt` before training, and the
actual historical best is copied to the new `best.pt`. New improvements replace
that best only on a strictly lower finite validation loss. The best metadata's
`run_id` and `source_checkpoint` identify its origin, even across multiple resumes.
The 50-periodic-plus-best/last limit applies **per output directory**; preserved
parent runs consume additional disk space. New validation/failure logs contain
only this segment's events; keep the entire source lineage for earlier events.

Use complete recorded checkpoints from this reproduction, not weights-only files
or untrusted downloads. The original upstream format lacks required AMP/selection
state and is rejected. Older recorded checkpoints without RNG state may resume
with an explicit warning. Configuration must match, including batch size, model,
optimizer, validation frequency, seed and split. Only `dataset.root`, `logdir` and
the total `train.max_iters` may differ. Relocated data must be the same dataset
and split; path compatibility checks do not prove data identity. A batch-24
checkpoint cannot silently continue as batch 8.

Keep the source `best.pt` (or the matching numbered checkpoint) with `last.pt`.
When resuming an older numbered checkpoint, its best may differ from today's
best alias. The code verifies historical best iteration, source run and loss,
and rejects a missing/mismatched best rather than inventing a replacement.

New checkpoints restore Python, NumPy and Torch RNG states after model/optimizer
setup; saved CUDA RNG state requires the same number of visible CUDA devices.
The shuffled multiworker DataLoader's sampler position and prefetched batches
are **not** saved. Resume starts a new shuffled iterator and is therefore not
bitwise equivalent to uninterrupted training, even with restored RNG state.
The seed is not a substitute for the missing iterator position.

The local resume tests use a small CPU linear model and simulated training calls.
They cover optimizer/scheduler/AMP-state restoration, RNG round trips, iteration
continuity, historical best selection, parent preservation, configuration checks
and the checkpoint limit. They do not run a real MDRL CUDA forward/backward pass
or establish production-training reliability.
