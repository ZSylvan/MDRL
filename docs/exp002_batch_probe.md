# Experiment 002: smaller single-GPU batches

## Scope And Prior Evidence

Follow-up to experiment 001, not a formal pretraining run. The user returned
server outputs for run `mdrl-exp001-batch-20261024T143604Z`, PBS job `20608.mgt`,
at commit `d3ad42201cae14f752452e24d3d69e29b2751a70`:

- Batch 8 completed the stress test but estimated device usage reached 99.1%,
  exceeding the existing 80% recommendation limit. Measured training had 19
  optimizer updates and one AMP-skipped update across 20 steps.
- Batch 16 completed 20 measured optimizer updates, then ran out of GPU memory
  in the first stress-training step. Its reported peak counters predate the
  failed step and are not its OOM-time peak; use the retained exception too.
- Batch 24 and larger were not tested. No batch was recommended.

These observations come from the returned server logs, not a local GPU test.
Preserve the entire first run under the shared `batch_probe` output root.

The new config `configs/experiments/exp002_batch_probe.yml` changes only the
experiment ID and candidates to **2, 4, 6**. It reuses the same model, seed,
optimizer, AMP, dataset split, 3 warmup steps, 20 measured steps, 2 largest-
training-molecule stress steps and 1 stress loss evaluation. The 600-second
candidate limit and 80% memory limit are unchanged. Full protocol and output
definitions remain in [experiment 001](exp001_batch_probe.md).

The formal training YAML remains at batch 24. Any formal batch/LR/iteration
change requires separate review after these measurements; no accumulation,
size filtering, reward changes or new random seeds are introduced.

## Local Checks And Synchronization

Run the existing unit tests locally without model execution:

```powershell
Set-Location 'D:\zhou\Research\project\MDRL'
& 'D:\anaconda3\python.exe' -B -m unittest discover -s tests -p 'test_*.py' -v
git diff --check
git add -- configs/experiments/exp002_batch_probe.yml scripts/prepare_batch_probe_pbs.py tests/test_batch_probe.py docs/exp002_batch_probe.md
git diff --cached --check
git diff --cached --stat
git commit -m "Add smaller-batch MDRL capacity follow-up"
git push gitee HEAD:mdrl-reproduction
git rev-parse HEAD
```

Only the user synchronizes and submits the server job. On the server:

```bash
cd /home/zhuenqiang_1/Student_Project/zms/project/MDRL
git status --short --branch
git pull --ff-only origin mdrl-reproduction
git rev-parse HEAD

PY=/home/zhuenqiang_1/.conda/envs/MDRL/bin/python
RUN_ID="mdrl-exp002-batch-$(date -u +%Y%m%dT%H%M%SZ)"
JOBDIR="/home/zhuenqiang_1/Student_Project/zms/experiment_data/mdrl/batch_probe/$RUN_ID"
GPU=1
"$PY" -B scripts/prepare_batch_probe_pbs.py \
  --experiment configs/experiments/exp002_batch_probe.yml \
  --job-name mdrl-batch002 --run-id "$RUN_ID" \
  --queue gentai --resources 'nodes=node1:ppn=32,walltime=02:00:00' \
  --gpu "$GPU"
bash -n "$JOBDIR/job.pbs"
cat "$JOBDIR/job.pbs"
```

Stop if pull/generation/syntax checks fail or local and server commits differ.
Confirm GPU 1 is available and manual selection is authorized; the GPU mask
does not reserve the card. The generator preserves scheduler-provided masks.
After reviewing the generated PBS, the user may submit:

```bash
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv
qstat -u "$USER"
JOB_ID=$(qsub "$JOBDIR/job.pbs")
printf '%s\n' "$JOB_ID" > "$JOBDIR/job_id.txt"
qstat "$JOB_ID"
```

Retain this `JOBDIR`; do not select a previous experiment's directory. Read
the direct log after the job starts, and the summaries after completion:

```bash
printf 'JOBDIR=%s\n' "$JOBDIR"
tail -n 40 "$JOBDIR/launcher.log"
cat "$JOBDIR/results/summary.csv"
"$PY" -B -m json.tool "$JOBDIR/results/summary.json"
```

All-passing results identify only the largest **tested** batch satisfying the
headroom rule, not an absolute memory limit or a guarantee for long training.
Do not automatically expand the grid or change the formal training config.
