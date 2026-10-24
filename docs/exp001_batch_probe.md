# Experiment 001: single-GPU pretraining batch capacity

## Scope

Use the existing MDRL environment on one allocated V100 32 GB. Do not add two
cards' memory together: the original trainer is single-GPU. This diagnostic
does not launch formal pretraining, reinforcement learning, sampling or docking.
It never changes a training YAML or saves a model checkpoint. Its short-lived
model weights and losses are not baseline results.

`configs/experiments/exp001_batch_probe.yml` defines eight increasing candidates:
8, 16, 24 (the original batch), 32, 48, 64, 96 and 128. Each candidate starts in a
fresh process with seed 2023 and the original model, optimizer, noise, AMP and
gradient-clipping settings. It performs:

1. Three random training warmup steps.
2. Twenty random training steps with synchronized elapsed-time measurements.
3. Two training steps on the largest molecules from the **training split**, ranked
   by the existing summary's heavy-atom counts, followed by one no-gradient loss
   evaluation on that stress batch. This is a memory check, not full validation.

All normal batches have the requested size. The normal loader retains the
original 16 workers and pin-memory setting; the stress loader uses zero workers.
Processed inputs are checked before calling the dataset constructor, so a wrong
path fails rather than triggering automatic dataset preprocessing. No dataset
split or molecular-size filter is changed.

Each candidate has a 600-second limit. CUDA OOM stops larger candidates normally;
other errors, nonfinite losses, missing inputs, timeouts and killed workers stop
the experiment without a recommendation. A timeout/interrupt also kills the
candidate's own DataLoader process group. Each candidate's failures and outputs
are retained. Successfully tested smaller candidates remain recorded after OOM.

## Recommendation

The provisional recommendation is the **largest tested passing batch** with at
least one confirmed optimizer update during measured training and an estimated
device-memory footprint no greater than 80% of capacity. It is not a claim of
maximum throughput. AMP-skipped updates are reported separately.

The estimate is the maximum of initial device-used memory plus PyTorch peak
reserved memory, and the greatest whole-device usage observed after a step.
Peak allocated/reserved memory and per-step node/halfedge counts are retained.
Memory counters use the existing
[PyTorch 1.10.1 CUDA memory API](https://github.com/pytorch/pytorch/blob/v1.10.1/torch/cuda/memory.py).
Global usage is sampled, not continuously measured; other jobs and later data
variation can still change memory requirements. The 20% headroom and largest-
molecule stress batch are conservative diagnostics, not an OOM-free guarantee.
If every candidate passes, the result is the largest **tested** candidate, not
the GPU's absolute limit. Do not automatically extend the grid or alter learning
rate, accumulation, precision, architecture or the formal batch size.

## Prepare And Execute On Server

First review, commit and push the local changes to Gitee, then pull them in the
server's `mdrl-reproduction` branch. Do not copy local datasets into Git.

The PBS generator is `scripts/prepare_batch_probe_pbs.py`. It requires explicit
`--queue` and `--resources` values from the cluster's working PBS convention.
The user-supplied working PBS example uses `gentai`, `nodes=node1:ppn=32` and
manual GPU selection with `CUDA_VISIBLE_DEVICES=1`. Use that resource format,
but confirm the chosen physical GPU is available/authorized at submission time.
A two-hour walltime accommodates the bounded grid including startup, rather
than reusing the long walltime of the FlowMol experiment. No GPU reservation
directive has been inferred from the example's CPU/node resource request.

The generator preserves a scheduler-provided `CUDA_VISIBLE_DEVICES`. Without
one, the job refuses to run unless the user explicitly authorizes a physical
GPU using the generator's `--gpu` flag. That flag is only a fallback and never
overrides the scheduler mask. Runtime uses logical `cuda:0` inside that mask.
Setting this variable selects a card; it does **not** reserve one in PBS.

The generator prints the new `job.pbs` path but never calls `qsub`. It creates
an independent timestamped job directory under the configured output root and
refuses to reuse it. Review the generated file, submit it yourself, and preserve
the job ID and scheduler exit information. PBS uses the absolute MDRL Python,
sets its environment paths, and limits OMP/MKL threads to one. These settings
are recorded/visible in `job.pbs`; no dependency installation occurs.

For a user-managed interactive GPU allocation, the equivalent runner is:

```bash
/home/zhuenqiang_1/.conda/envs/MDRL/bin/python -u -B \
  scripts/probe_pretrain_batch.py \
  --experiment configs/experiments/exp001_batch_probe.yml
```

Run it only from a properly allocated compute node with an authorized GPU mask,
not a login node. The runner rejects Windows execution and has no CPU fallback.
This document does not authorize or submit any server job.

## Outputs And Local Checks

Under each job directory, preserve `job.pbs`, `pbs.out`, `launcher.log` and
`results/`. The launcher log is written directly on shared storage rather than
waiting for PBS output staging; it records start/end, hostname, PID and exit code.
Results
include copied experiment/base configurations, incremental `summary.json`,
`summary.csv`, code version and PBS job ID. Each `batch_NNN/` retains its effective
model/training configuration, `console.log`, final/progress `result.json` and
`steps.jsonl` with losses, molecule IDs, sizes, timings and memory statistics.
The probe's loop lengths come from the experiment YAML, not the original
`max_iters` retained in the copied model/training configuration.

Return the summary plus any failing candidate's result/console log for review.
Approve a formal batch-size change separately; increasing it changes the number
of molecules processed per iteration even when other hyperparameters are fixed.

Local checks (no MDRL model execution, no CUDA work, no dependencies installed):

```powershell
python -B -m unittest discover -s tests -p 'test_*.py' -v
```

Tests exercise selection rules, input checks, stress-split selection, simulated
worker outcomes, PBS rendering and checkpoint retention. GPU/data/model runtime
compatibility remains to be established by the actual server experiment.
