# Node5 batch-16 formal pretraining

This is an independent from-scratch pretraining run, not continuation of the
batch-8 baseline or the node4 one-epoch diagnostic. Do not stop those jobs or
modify their files as part of this preparation. The generator never submits a
job, installs dependencies, downloads data, or preprocesses missing inputs.

## Approved recipe

- Node5, its single physical GPU 0, 15 PBS CPU slots, 14 training loader workers.
- Original model, optimizer, learning rate 1e-4, AMP, seed 2023 and data split.
- Batch 16; total 2400000 iterations, including after resume (not additional steps).
- Full validation every 10 epochs, plus final iteration. With 231523 effective
  training samples and drop_last=False, an epoch is 14471 iterations and the
  validation interval is 144710. The generator counts index-matched split entries
  without loading LMDB contents or changing the split. The trainer verifies the
  resolved interval against its actual training subset before training.
- Explicitly approved changes from upstream: batch 24 to 16, max_iters 2500000
  in the repository YAML to 2400000, validation cadence, CPU workers, paths,
  PBS deployment and artifact retention. The paper reports batch 24 and 1600000
  iterations; 16 x 2400000 matches nominal sample presentations, not the same
  optimization trajectory. Do not describe this as identical paper settings.
- The original scheduler is still called only after actual validation. Its
  patience/factor/minimum LR are unchanged. Less frequent calls and less frequent
  opportunities to select best are a documented behavior change, not merely
  reduced logging. No automatic LR scaling, gradient accumulation or seed sweep.

## Checkpoints and records

`last.pt` is atomically refreshed every 5000 iterations without validation.
It contains model, optimizer, scheduler, AMP scaler, Python/NumPy/Torch/CUDA RNG,
iteration counters and historical best metadata. It does not create a numbered
file on these recovery-only saves and never updates best or the scheduler.
Before the first completed recovery save there is no resumable checkpoint.

Each actual validation retains its numbered `<iteration>.pt` and can update
`best.pt`, using lowest finite validation total loss, first occurrence on ties.
Before the first finite validation, best.pt does not exist. Validation loss is
the original mean of per-batch losses, not a new molecule-weighted metric.

`1600000.pt` is saved exactly at iteration 1600000 without extra validation if
it is not a scheduled validation. It is not automatically eligible for best.
`2400000.pt` is the final validated checkpoint. These two numbered files are
pinned against automatic retention pruning, even if best occurs elsewhere.

With the current split, the full schedule has 17 validation checkpoints and
one extra numbered milestone, plus best and last: 20 completed .pt files.
All validation checkpoints are retained under the limit. If the schedule ever
requires more, the oldest unpinned numbered files rotate; 48 numbered files plus
best/last is a maximum of 50 per training segment. Atomic temporary files may
briefly consume additional disk space. Checkpoint write/copy failures are fatal
and recorded, rather than silently treated as saved. A recovery-only save keeps
the previous last until the replacement has been fully written.

Resume writes a new independent output segment and never deletes or overwrites
the parent. Existing milestone and validation files remain in their owning
parent directory; retain the whole lineage when archiving. The 50-file limit is
per segment, not a license to delete other segments or existing experiments.

Preserve config.yml, protocol.json, job.pbs, launcher.log, pbs.out, the training
directory's config copy, run.json, log.txt, TensorBoard events, validation.jsonl,
checkpoints.jsonl, failures.jsonl and checkpoints. `checkpoints.jsonl` separates
validation, milestone and recovery-only saves. Validation entries alone select
best. Other original training RuntimeErrors are still recorded/skipped; the wrapper
reports a nonzero job exit for a completed run with any skips. AMP may separately
skip optimizer updates; completed_train_calls is not a guarantee of updates.

## Local verification and publishing

Run the unit tests locally; these cover schedule, retention, atomic recovery,
resume state and Bash syntax, not actual MDRL CUDA execution on node5:

```powershell
Set-Location 'D:\zhou\Research\project\MDRL'
& 'D:\anaconda3\python.exe' -B -m unittest discover -s tests -p 'test_*.py' -v
git add -- configs/train/train_MDRL_node5_b16.yml scripts/prepare_pretrain_pbs.py scripts/train_drug3d.py utils/pretrain_record.py utils/pretrain_resume.py tests/test_node5_pretrain.py tests/test_pretrain_record.py tests/test_pretrain_resume.py docs/node5_pretraining.md
git diff --cached --stat
git commit -m "Add resumable node5 batch16 formal pretraining"
git push gitee mdrl-reproduction
```

The original batch-8 YAML and original upstream YAML are unchanged. Old calls
retain their original validation frequency and default 50-periodic-plus-best/last
policy. The node4 probe still defaults to its prior validation behavior.

## Server preparation and submission

Use the clean Gitee checkout. The queue/resource syntax follows the previous
gentai examples. Node5 and 15 slots are defaults, not launch restrictions:
choose another available node/CPU count via `--resources`, or edit `#PBS -l`
in the generated PBS before submission. The recipe file's node5 name is historical;
it does not restrict execution to node5. Changing placement does not change the
batch, loader workers, learning rate, validation schedule or checkpoint source.
Physical GPU selection is not an atomic GPU reservation. The launcher rejects
a conflicting scheduler GPU mask and, by default, active compute processes; it never kills
another job. It checks Python, committed Git revision and clean state.
It records the actual hostname, PBS node-file slots/nodes when available, GPU
and Torch/CUDA details. It does not reject a different hostname. Node4 success is not proof of node5
runtime compatibility. Missing environment/data stops the launch. A busy GPU
also stops it unless `--allow-shared-gpu` was explicitly provided when preparing
the PBS. That option records existing compute PIDs and the nvidia-smi snapshot
before proceeding; it does not reserve memory or guarantee freedom from OOM.
`protocol.json` preserves the resources supplied at generation. If the PBS is
edited later, retain that final PBS and the actual-placement launcher log too;
the protocol's original resource request is not evidence of the execution host.

```bash
(
set -euo pipefail
BASE=/home/zhuenqiang_1/Student_Project/zms
cd "$BASE/project/MDRL"
git pull --ff-only origin mdrl-reproduction
PY=/home/zhuenqiang_1/.conda/envs/MDRL/bin/python
RUN_ID="mdrl-node5-b16-$(date -u +%Y%m%dT%H%M%SZ)"
RUN="$BASE/experiment_data/mdrl/pretrain/$RUN_ID"
"$PY" -B scripts/prepare_pretrain_pbs.py \
    --queue gentai \
    --resources 'nodes=node5:ppn=15,walltime=240:00:00' \
    --run-id "$RUN_ID"
bash -n "$RUN/job.pbs"
cat "$RUN/protocol.json"
printf '\nSubmission command (not executed):\nqsub %q\n' "$RUN/job.pbs"
)
```

Execute the printed qsub command manually after inspecting the protocol.
Never rerun the same job.pbs: launcher.log refuses replacement. Generate a new
run ID for a retry or continuation. Keep the checkout at the prepared revision
until the job has started; later code changes are not applied to a running
process, and should not be assumed to update that run's implementation.

## Resume after a stopped job

First confirm the old job is no longer training. Do not use an active parent's
changing last/best aliases as a new concurrent training source. Do not remove
the old segment, even if PBS left run.json at status running after termination.

```bash
(
set -euo pipefail
BASE=/home/zhuenqiang_1/Student_Project/zms
cd "$BASE/project/MDRL"
PY=/home/zhuenqiang_1/.conda/envs/MDRL/bin/python
# Set this to the actual stopped node5 formal training segment, not a probe or batch8 run.
SOURCE_LAST=/absolute/path/to/stopped/training/config_TIMESTAMP/checkpoints/last.pt
RUN_ID="mdrl-node5-b16-resume-$(date -u +%Y%m%dT%H%M%SZ)"
"$PY" -B scripts/prepare_pretrain_pbs.py \
    --queue gentai \
    --resources 'nodes=node5:ppn=15,walltime=240:00:00' \
    --run-id "$RUN_ID" --resume "$SOURCE_LAST"
PBS="$BASE/experiment_data/mdrl/pretrain/$RUN_ID/job.pbs"
bash -n "$PBS"
printf '\nSubmission command (not executed):\nqsub %q\n' "$PBS"
)
```

Generation validates the trusted full checkpoint, historical best and recipe
before creating output. The job repeats this validation at startup. Restore
starts at saved iteration + 1 and ends at total iteration 2400000. Batch,
validation/recovery schedule, optimizer and other recipe settings must match;
only dataset root, output root and total max_iters have the existing permitted
flexibility. The batch8 or node4 diagnostic checkpoint is not a valid resume
source for this recipe.

Multiworker shuffled loader position/prefetched samples are not restored.
Resume preserves training state but is not bitwise-equivalent uninterrupted
training or exactly continuous epoch coverage. The global validation/milestone
iteration schedule does not restart at each segment. With a 240-hour PBS limit,
use the latest completed last checkpoint if continuation is needed; no automatic
qsub, qdel, signal handler emergency checkpoint or retry is implemented.

All these files are pretraining artifacts, not dual-target fine-tuning results
or independent generated-molecule evaluation. The latter remains separate work.

## Guarded continuation from the healthy best

The October 30 continuation uses the stopped batch16 run's `best.pt` at
iteration 723550, not its contaminated last checkpoint or the 868260 checkpoint
whose validation losses were nonfinite. The source directory is
`mdrl-node5-b16-20261026T041227Z/training/config_20261026_185804/checkpoints`.
The source is preserved. Training resumes at 723551 in a new directory, with
the full optimizer, scheduler, AMP scaler, RNG, counters and historical best.
The total target remains 2400000. The next scheduled validation is 868260;
validation is still tied to global steps, not ten fresh epochs after restart.

No learning rate, AMP, model, loss or optimizer setting is changed. Added checks
stop on NaN or infinity in any loss component before backward, or in the
unscaled total gradient norm before the optimizer update. Validation also stops
at the first batch with a nonfinite loss, before averaging, scheduling or saving
best. The original gradient-clipping call is retained. Its returned norm is
measured before clipping; the diagnostic gradient tensors are captured after
clipping, and are labeled accordingly. Other RuntimeErrors retain the existing
record-and-skip behavior, including OOM.

On a numerical stop, the training directory retains:

- `diagnostics/first_failure.json`: step, phase, failing fields, molecule IDs
  when available, validation batch index, and best/last references.
- `diagnostics/first_failure.pt`: batch tensors, position noise when applicable,
  RNG immediately before the loss call, loss values, model/optimizer/scheduler/
  scaler state, gradients when applicable, and counters.
- `run.json` with failed status and `failures.jsonl` with the error traceback.

The diagnostic PT is **not resumable** and cannot replace best/last or enter
checkpoint selection. It is a separate one-time failure artifact beyond the
50 regular checkpoint limit. A second diagnostic write refuses replacement;
write failures are fatal and recorded, not skipped as training errors.
Resume preparation rejects nonfinite model/optimizer/scaler state and a
nonpositive AMP scale. Legitimate scheduler infinity sentinels remain allowed.

An observed overflow is evidence of that event, not proof that all weights
were permanently corrupted or that all official-default runs necessarily fail.
AMP may normally skip an overflowing update. This continuation intentionally
stops earlier to preserve the first observed event; the checks do not fix its
cause. Batch16 and the changed validation schedule are already documented
departures from the upstream recipe.

The inherited checkpoint has one historical skipped iteration. Counters are
not reset: even if the continuation reaches the target without a new skip,
the existing completion policy reports `completed_with_skips` and PBS exit 2.
Inspect the recorded status and failure list rather than treating that final
exit code alone as a new numerical failure.

GPU sharing was explicitly approved for this continuation. It can increase OOM
risk and change timing. It does not permit stopping another user's process.
The 300-hour walltime below follows the prior run's resource setting; no job
is submitted by the generator.

```bash
(
set -euo pipefail
BASE=/home/zhuenqiang_1/Student_Project/zms
cd "$BASE/project/MDRL"
git pull --ff-only origin mdrl-reproduction
PY=/home/zhuenqiang_1/.conda/envs/MDRL/bin/python
SOURCE="$BASE/experiment_data/mdrl/pretrain/mdrl-node5-b16-20261026T041227Z/training/config_20261026_185804/checkpoints/best.pt"
RUN_ID="mdrl-node5-b16-guarded-$(date -u +%Y%m%dT%H%M%SZ)"
RUN="$BASE/experiment_data/mdrl/pretrain/$RUN_ID"
"$PY" -B scripts/prepare_pretrain_pbs.py \
    --run-id "$RUN_ID" --resume "$SOURCE" --allow-shared-gpu \
    --queue gentai --resources 'nodes=node5:ppn=15,walltime=300:00:00'
bash -n "$RUN/job.pbs"
cat "$RUN/protocol.json"
printf '\nSubmission command (not executed):\nqsub %q\n' "$RUN/job.pbs"
)
```

Local CPU tests exercise the actual train/validate function bodies and outer
loop with injected bad losses/gradients, verifying stop-before-update, failure
records, preserved aliases and rejected invalid resume states. They do not
reproduce the server's original NaN cause. All 80 local tests passed, including
editable node/CPU placement, PBS sharing options and Bash syntax checks. Loading the real downloaded
checkpoint locally was blocked by missing `easydict`; no dependency was
installed. The server generator checks the real checkpoint/config before
creating the job directory, and the trainer repeats the check at startup.

Publish this continuation's scoped changes locally:

```powershell
Set-Location 'D:\zhou\Research\project\MDRL'
git add -- scripts/train_drug3d.py scripts/prepare_pretrain_pbs.py utils/pretrain_checks.py utils/pretrain_record.py utils/pretrain_resume.py tests/test_pretrain_checks.py tests/test_pretrain_record.py tests/test_pretrain_resume.py tests/test_node5_pretrain.py docs/node5_pretraining.md
git diff --cached --stat
git commit -m "Guard nonfinite training and resume batch16 with explicit GPU sharing"
git push gitee mdrl-reproduction
```
