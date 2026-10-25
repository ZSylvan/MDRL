"""Prepare, but never submit, an MDRL capacity experiment."""

import argparse
import re
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.probe_pretrain_batch import load_experiment


PYTHON = '/home/zhuenqiang_1/.conda/envs/MDRL/bin/python'


def render_pbs(repo, experiment, job_dir, queue, resources, gpu=None, python=PYTHON,
               job_name='mdrl-batch001'):
    if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]{0,14}', job_name):
        raise ValueError('PBS job name must start with a letter and use at most 15 safe characters')
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', queue):
        raise ValueError('Invalid PBS queue')
    if not re.fullmatch(r'[A-Za-z0-9_.,:=+-]+', resources):
        raise ValueError('Invalid PBS resources; pass the actual cluster resource request')
    if gpu is not None and gpu < 0:
        raise ValueError('Manual GPU index must be nonnegative')
    job_dir = PurePosixPath(str(job_dir))
    if not job_dir.is_absolute() or not re.fullmatch(r'[A-Za-z0-9_/.-]+', str(job_dir)):
        raise ValueError('PBS output needs an absolute POSIX path without spaces')
    quote = shlex.quote
    env_root = str(PurePosixPath(python).parent.parent)
    mask_fallback = ('    export CUDA_VISIBLE_DEVICES=%s' % quote(str(gpu))) if gpu is not None else (
        '    echo "ERROR: no scheduler GPU mask; authorize explicit selection with --gpu" >&2\n'
        '    exit 2')
    return '\n'.join([
        '#!/bin/bash', '#PBS -N ' + job_name, '#PBS -q ' + queue,
        '#PBS -l ' + resources, '#PBS -j oe', '#PBS -o ' + str(job_dir / 'pbs.out'),
        '', 'set -euo pipefail',
        'DIRECT_LOG=' + quote(str(job_dir / 'launcher.log')),
        'exec > "$DIRECT_LOG" 2>&1',
        'echo "launcher_start=$(date -Is)"', 'echo "launcher_host=$(hostname)"',
        'echo "launcher_pid=$$"',
        "trap 'status=$?; echo \"launcher_exit_status=$status\"; echo \"launcher_end=$(date -Is)\"; exit \"$status\"' EXIT",
        'REPO_ROOT=' + quote(str(repo)), 'PY=' + quote(python),
        'ENV_ROOT=' + quote(env_root),
        'export PATH="$ENV_ROOT/bin:$PATH"',
        'export LD_LIBRARY_PATH="$ENV_ROOT/lib:${LD_LIBRARY_PATH:-}"',
        'export CONDA_PREFIX="$ENV_ROOT"', 'export CONDA_DEFAULT_ENV=MDRL',
        'export PYTHONPATH="$REPO_ROOT"', 'export PYTHONDONTWRITEBYTECODE=1',
        'export OMP_NUM_THREADS=1', 'export MKL_NUM_THREADS=1',
        # Never override the scheduler's allocation mask with a physical index.
        'if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then', mask_fallback, 'fi',
        'echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"',
        'cd "$REPO_ROOT"', 'test -x "$PY"',
        '"$PY" -u -B scripts/probe_pretrain_batch.py --experiment ' + quote(str(experiment))
        + ' --outdir ' + quote(str(job_dir / 'results')), '',
    ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, default=ROOT / 'configs/experiments/exp001_batch_probe.yml')
    parser.add_argument('--queue', required=True)
    parser.add_argument('--resources', required=True, help='Exact PBS -l request approved for this cluster')
    parser.add_argument('--gpu', type=int, help='Physical GPU only if manual selection is permitted; preserves scheduler mask')
    parser.add_argument('--job-name', default='mdrl-batch001', help='PBS job label, at most 15 characters')
    parser.add_argument('--run-id', default='mdrl-exp001-batch-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    args = parser.parse_args()
    if sys.platform != 'linux':
        parser.error('Prepare the actual PBS file on the Linux server after pulling the committed code')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', args.run_id):
        parser.error('run-id must be a simple directory name')
    config = load_experiment(args.experiment)
    job_dir = Path(config['output_root']) / args.run_id
    source = render_pbs(ROOT, args.experiment.resolve(), job_dir, args.queue, args.resources, args.gpu,
                        job_name=args.job_name)
    job_dir.mkdir(parents=True, exist_ok=False)
    path = job_dir / 'job.pbs'
    with path.open('x', encoding='utf-8', newline='\n') as handle:
        handle.write(source)
    print(path)
    print('Review the file before submission; no job was submitted.')


if __name__ == '__main__':
    main()
