"""Prepare the node5 best-checkpoint generation probe; never submit a job."""

import argparse
import hashlib
import json
import re
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = Path('/home/zhuenqiang_1/Student_Project/zms')
PYTHON = '/home/zhuenqiang_1/.conda/envs/MDRL/bin/python'
DEFAULT_CHECKPOINT = (BASE / 'experiment_data/mdrl/pretrain/'
                      'mdrl-node5-b16-20261026T041227Z/training/'
                      'config_20261026_185804/checkpoints/best.pt')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', default='mdrl-best-probe-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--expected-iteration', type=int, default=723550)
    parser.add_argument('--sa-scores', type=Path)
    parser.add_argument('--allow-shared-gpu', action='store_true',
                        help='Allow existing GPU compute processes; log them without stopping')
    args = parser.parse_args()
    if sys.platform != 'linux':
        parser.error('Generate the PBS on the Linux server after pulling committed code')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', args.run_id):
        parser.error('Invalid run ID')
    status = subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True)
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    if status:
        parser.error('Use a clean, committed repository')
    checkpoint = args.checkpoint.resolve(strict=True)
    digest = hashlib.sha256()
    with checkpoint.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    if args.sa_scores is not None and not args.sa_scores.is_file():
        parser.error('Specified SA scores file is missing')
    directory = BASE / 'experiment_data/mdrl/inference' / args.run_id
    if not re.fullmatch(r'[A-Za-z0-9_/.-]+', str(directory)):
        parser.error('PBS output path must not contain spaces or shell metacharacters')
    config = {
        'experiment': 'exploratory_unconditional_best_probe', 'checkpoint': str(checkpoint),
        'checkpoint_sha256': digest.hexdigest(), 'expected_iteration': args.expected_iteration,
        'attempts': 100, 'batch_size': 1, 'seed': 2023, 'device': 'cuda:0',
        'output': str(directory / 'results'),
        'sa_scores': str(args.sa_scores.resolve()) if args.sa_scores is not None else None,
        'git': {'commit': revision, 'dirty': False},
        'node': 'node5', 'physical_gpu': 0, 'cpu_slots': 15,
        'allow_shared_gpu': args.allow_shared_gpu,
        'queue': 'gentai', 'walltime': '24:00:00', 'replacement_sampling': False,
    }
    quote = shlex.quote
    source = f'''#!/bin/bash
#PBS -N mdrl-best-probe
#PBS -q gentai
#PBS -l nodes=node5:ppn=15,walltime=24:00:00
#PBS -j oe
#PBS -o {directory}/pbs.out
set -euo pipefail
set -o noclobber
JOB_DIR={quote(str(directory))}
REPO={quote(str(ROOT))}
REV={quote(revision)}
PY={quote(PYTHON)}
ENV_ROOT=/home/zhuenqiang_1/.conda/envs/MDRL
ALLOW_SHARED_GPU={int(args.allow_shared_gpu)}
exec > "$JOB_DIR/launcher.log" 2>&1
echo "launcher_start=$(date -Is) host=$(hostname) job=${{PBS_JOBID:-unknown}}"
trap 'rc=$?; echo "launcher_exit_status=$rc"; echo "launcher_end=$(date -Is)"; exit "$rc"' EXIT
export PATH="$ENV_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$ENV_ROOT/lib:${{LD_LIBRARY_PATH:-}}"
export CONDA_PREFIX="$ENV_ROOT" CONDA_DEFAULT_ENV=MDRL
export PYTHONPATH="$REPO${{PYTHONPATH:+:$PYTHONPATH}}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
[[ "$(hostname -s)" == node5 ]] || {{ echo 'ERROR: expected node5'; exit 2; }}
if [[ -n "${{CUDA_VISIBLE_DEVICES:-}}" && "$CUDA_VISIBLE_DEVICES" != 0 ]]; then
    echo 'ERROR: scheduler GPU mask differs from physical GPU 0'; exit 2
fi
export CUDA_VISIBLE_DEVICES=0
cd "$REPO"
test -x "$PY"
[[ "$(git rev-parse HEAD)" == "$REV" && -z "$(git status --porcelain)" ]] || {{
    echo 'ERROR: repository changed after preparation'; exit 2;
}}
pids=$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader)
if [[ -n "$pids" ]]; then
    if [[ "$ALLOW_SHARED_GPU" != 1 ]]; then
        echo "ERROR: GPU 0 is occupied: $pids"; exit 2
    fi
    echo "WARNING: GPU sharing explicitly allowed; existing compute PIDs: $pids"
fi
nvidia-smi -i 0
"$PY" -u -B scripts/sample_best_probe.py --config "$JOB_DIR/config.json"
'''
    directory.mkdir(parents=True, exist_ok=False)
    with (directory / 'config.json').open('x', encoding='utf-8') as handle:
        json.dump(config, handle, indent=2, allow_nan=False)
        handle.write('\n')
    with (directory / 'job.pbs').open('x', encoding='utf-8', newline='\n') as handle:
        handle.write(source)
    print(json.dumps(config, indent=2))
    print('PBS=' + str(directory / 'job.pbs'))
    print('No job submitted. Manual GPU selection does not reserve a GPU in PBS.')
    print('qsub ' + quote(str(directory / 'job.pbs')))
    return 0


if __name__ == '__main__':
    sys.exit(main())
