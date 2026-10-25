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
