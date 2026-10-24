# Pretraining reproduction

This stage trains the original GEOM diffusion model from scratch. It does not
run pair-specific reinforcement learning or independent molecule evaluation.

## Configuration

`configs/train/train_MDRL_reproduction.yml` differs from the original configuration
only in its external dataset root and output root (`logdir`). The original model,
seed (2023), batch size (24), worker count (16), 2,500,000 iterations, validation
frequency (5,000), optimizer, scheduler, AMP, transforms and split are retained.
The original YAML is unchanged.

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
`iteration` fields, plus AMP scaler state, selection evidence and skip counters.
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
python -B -m unittest discover -s tests -p 'test_pretrain_record.py' -v
```

These checks exercise checkpoint serialization/selection/retention, failure records, the
training loop with simulated calls, Python 3.8 syntax, import order and configuration
equivalence. They do **not** execute the MDRL model, verify CUDA/PyG compatibility
or establish that a full training job will succeed. No local formal experiment,
server submission, new dependency installation or automatic data processing is
part of these checks. The server-side model/data smoke test and PBS resources
still need confirmation before formal pretraining.
