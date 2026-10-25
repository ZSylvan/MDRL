# Experiment 003: one training pass at batch 16 and 24

This is a diagnostic, not formal baseline pretraining. It does not replace,
resume, stop, or overwrite the running batch-8 job. Prepare it locally, test,
commit/push, then generate PBS on the server. No dependency installation or
data download is required.

The generator reads only the downloaded index and split to count the effective
original training subset, matching `get_dataset`'s exclusion of unavailable IDs.
It generates separate YAML files for batches 16 and 24, with
`max_iters = ceil(train_size / batch_size)` and independent output roots.
The trainer's shuffled DataLoader does not drop its final partial batch. A clean
run therefore visits every entry in that original training subset exactly once.
Validation/test samples are not added to training. No largest-molecule stress
batch, extra warmup, repeated sampling, or resume is introduced.

The existing trainer is unchanged. Its original optimizer, seed, AMP, noise,
worker count, validation cadence and final validation are retained. Each batch
starts a fresh model/optimizer in its own process. Thus this also tests normal
full validation, but makes only one training pass. Its ordinary checkpoint files
are diagnostic artifacts, not selected baseline checkpoints.

The original trainer catches/skips training RuntimeErrors. This experiment does
not change that policy: it preserves every skipped/fatal event and labels **any
skip as failure**, even if the process exits zero. A CUDA OOM during training or
validation is `oom`; neither incomplete work nor another error is called a pass.
Batch 24 still runs after batch 16 finishes or fails. A node loss, time limit,
busy GPU or other launcher failure can prevent later work; missing results are
incomplete, never passed. AMP may skip optimizer updates independently of these
counters; this is an OOM diagnostic, not a proof of updates on every batch.

Use another authorized GPU (intended physical GPU 0 on node1), not the active
formal job's GPU 1. The launcher rejects a mismatching scheduler GPU mask and
checks for active compute processes before each candidate. This is not an
atomic GPU reservation: follow the cluster's allocation rules. Concurrent jobs
share CPU/RAM/I/O, so speed is not a clean isolated throughput measurement.
The original formal job's output and reproduction YAML are untouched.

Each exclusive run directory under `experiment_data/mdrl/epoch_probe` retains
`protocol.json`, effective YAML files, PBS, launcher log, per-batch exit codes,
training logs, validation/failure records and checkpoints. `summary.json` reports
both candidates, their expected/completed iterations and recorded CUDA OOMs.
Use `--summarize /absolute/run/directory` to inspect an interrupted experiment;
it only reads its existing logs and replaces the derived summary.

Passing means no recorded errors for one shuffled training pass and the original
validation calls. It does not prove that later shuffle orders/batches, model
states or competing workloads cannot OOM. This experiment never changes the
running baseline's batch size or recommends doing so automatically.

## Node4 RTX 4090: batches 12 and 16

The generator also accepts `--batches 12 16 --num-workers 6`. Default calls
still test batches 16 and 24 and retain the original worker count. Summaries
read candidate sizes from each run's protocol, including older 16/24 runs.

For node4, request `nodes=node4:ppn=7,walltime=24:00:00` and `--gpu 1`.
The main training process uses six training DataLoader workers; the existing
validation loader uses zero workers. OMP, MKL, OpenBLAS and NumExpr thread
counts are limited to one. This is a seven-CPU request, not seven workers plus
the trainer. Physical GPU 1 is exposed as logical `cuda:0` inside the job.
Confirm that `gentai` permits node4 and follow the cluster's GPU allocation rules.

Use an independent ID such as `mdrl-exp003-node4-b12-b16-<UTC timestamp>`.
With 231523 effective training samples, batch 12 needs 19294 iterations and
batch 16 needs 14471. Each candidate starts from scratch and performs the
original validation calls, including final validation. Batch 16 is attempted
even if batch 12 fails. Neither the running batch-8 pretraining job nor the
completed node1 test directories are changed or reused.

The existing MDRL environment is reused without installation. Its actual CUDA
and compiled-extension compatibility with the RTX 4090 must be established by
the server run; an architecture/kernel/import error is a compatibility failure,
not proof of insufficient GPU memory. Preserve its logs rather than relabeling
every failure as OOM. GPU availability is checked before each candidate; a
busy GPU stops the launcher without terminating another process.
