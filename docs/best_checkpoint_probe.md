# Best-checkpoint generation probe

This is exploratory unconditional inference, not a dual-target benchmark or an
activity assessment. It does not train, dock, change existing outputs, repair
training numerics, or select a checkpoint using generation results.

## Default experiment

- Source: node5 batch-16 pretraining best.pt, iteration 723550.
- Resources: node5, physical GPU 0, 15 CPU slots, gentai, 24-hour limit.
- Exactly 100 attempt slots, inference batch size 1, seed 2023 set once.
- No replacement draws for failed, invalid, disconnected, or duplicate samples.
- Reuse MDRL.sample, its full checkpoint-defined diffusion schedule, the original
  node-count distribution, decoder, and reconstruction (including its repairs).
- No autocast, bond predictor, guidance, or extra conformer optimization, matching
  the original unconditional sampling path. No LMDB or receptor inputs are needed.
- Reject nonfinite model weights, changed checkpoint content, or unexpected source
  iteration before sampling. Model-weight finiteness does not guarantee successful
  inference; sampling and reconstruction failures remain in the records.
- The existing TensorBoard-based sampler is unchanged; this entry needs no
  TensorBoard import. Training scripts and checkpoint files are unchanged.

## Outputs

Each prepared experiment has config.json and job.pbs. Its results/ directory is
created exclusively at launch. Reuse is refused; create a new run ID for retries.
There is no generation-resume or automatic PBS submission.

- metadata.json: checkpoint hash/iteration, source skip count, source training
  configuration reference, actual Git and package versions, and metric definitions.
- source_train_config.json: the checkpoint's actual training recipe.
- molecules.jsonl: durable per-attempt records flushed after each attempt.
- molecules.csv: the same records in CSV at normal completion, excluding the
  detailed metric_errors dictionary (which remains in JSONL).
- raw/mol_NNN.npz: final predicted atom logits, positions, bond logits, and edges,
  including nonfinite predictions when sampling returns them; not diffusion paths.
- sdf/mol_NNN.sdf: every reconstructed, sanitized nonempty molecule, including
  disconnected structures, with its generated coordinates retained.
- structures_NNN.png: all valid connected molecules in attempt order, 20 per page;
  no QED sorting or cherry-picking. Drawing failures are explicitly reported.
- summary.json and log.txt: progress, counts, missing metrics, and errors.

Valid% uses all 100 slots as denominator at complete execution. Validity is after
the original reconstruction/repair, not raw graph validity. Conn% is conditional
on validity; connected yield additionally uses all 100. Unique% uses canonical
RDKit SMILES among valid connected molecules, without extra stereo assignment.
QED, raw SA (1-10, lower is easier), MolWt, MolLogP, HBD and HBA are summarized
over valid connected, non-missing values, with separate denominators. Lipinski is
MolWt <= 500, MolLogP <= 5, HBD <= 5 and HBA <= 10, following the handoff protocol;
it is not MDRL's internal training reward. No novelty claim or 3D-strain assessment
is made. Interrupted runs report unattempted slots and no final /100 validity.

SA uses the installed RDKit Contrib implementation and its bundled fpscores, or
an explicitly supplied trusted .pkl.gz via --sa-scores. Missing SA resources yield
null values with a recorded reason; nothing is installed or downloaded. A completed
experiment means all slots were recorded, not that the molecules passed any quality
threshold or have biological activity.

## Server preparation

After committing locally, pushing to Gitee, and pulling on the server:

```bash
PY=/home/zhuenqiang_1/.conda/envs/MDRL/bin/python
"$PY" -B scripts/prepare_sample_probe_pbs.py --run-id UNIQUE_RUN_ID
```

Inspect config.json, check the generated script with bash -n, and manually execute
the printed qsub command. The launcher refuses an occupied GPU, a conflicting GPU
mask, or a changed/dirty checkout. Confirm the old node5 training job has stopped;
this experiment never cancels jobs. CPU slots/manual CUDA visibility alone do not
reserve the GPU against other users.

For another trusted checkpoint, explicitly supply --checkpoint and the matching
--expected-iteration. Do not point this experiment at a contaminated last.pt.

Local tests were intentionally not run for this addition at the user's request.
The first real inference and environment check are performed on the server.
