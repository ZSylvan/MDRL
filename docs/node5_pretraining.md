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
best. Original training RuntimeErrors are still recorded/skipped; the wrapper
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
gentai examples; node5 must actually belong to that queue and permit 15 slots.
Physical GPU selection is not an atomic GPU reservation. The launcher rejects
a conflicting scheduler GPU mask and active compute processes; it never kills
another job. It checks hostname, Python, committed Git revision and clean state.
It logs actual GPU and Torch/CUDA details; node4 success is not proof of node5
runtime compatibility. Missing environment/data or a busy GPU stops the launch.

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
