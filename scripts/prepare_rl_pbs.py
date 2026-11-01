"""Prepare independent config-driven MDRL fine-tuning jobs; never submit them."""

import argparse
import json
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml

from RL_utils.task_config import load_task_config


ENV_ROOT = '/home/zhuenqiang_1/.conda/envs/MDRL'
OUTPUT_ROOT = '/home/zhuenqiang_1/Student_Project/zms/experiment_data/mdrl/finetune'


def positive_integer(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError('Must be a positive integer')
    return value


def nonnegative_integer(value):
    value = int(value)
    if value < 0:
        raise argparse.ArgumentTypeError('Must be a nonnegative integer')
    return value


def render_pbs(repo, directory, name, queue, resources, gpu):
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', queue):
        raise ValueError('Invalid queue')
    if not re.fullmatch(r'[A-Za-z0-9_.,:=+-]+', resources):
        raise ValueError('Invalid PBS resource string')
    if (not re.fullmatch(r'[A-Za-z0-9_/.-]+', str(directory)) or
            not PurePosixPath(str(directory)).is_absolute()):
        raise ValueError('PBS directory must be an absolute POSIX path without spaces')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', name):
        raise ValueError('Invalid task name')
    if type(gpu) is not int or gpu < 0:
        raise ValueError('GPU index must be a nonnegative integer')
    quote = shlex.quote
    return f'''#!/bin/bash
#PBS -N mdrl-{name[:10]}
#PBS -q {queue}
#PBS -l {resources}
#PBS -j oe
#PBS -o {directory}/pbs.out
set -euo pipefail
set -o noclobber
JOB_DIR={quote(str(directory))}
REPO={quote(str(repo))}
ENV_ROOT={quote(ENV_ROOT)}
PY="$ENV_ROOT/bin/python"
exec >> "$JOB_DIR/launcher.log" 2>&1
echo "launcher_start=$(date -Is) host=$(hostname) job=${{PBS_JOBID:-unknown}}"
trap 'rc=$?; echo "launcher_exit_status=$rc"; echo "launcher_end=$(date -Is)"; exit "$rc"' EXIT
if [[ -r "${{PBS_NODEFILE:-}}" ]]; then
    echo "allocated_cpu_slots=$(wc -l < "$PBS_NODEFILE")"
fi
export PATH="$ENV_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$ENV_ROOT/lib:${{LD_LIBRARY_PATH:-}}"
export CONDA_PREFIX="$ENV_ROOT" CONDA_DEFAULT_ENV=MDRL
export PYTHONPATH="$REPO"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES={gpu}
cd "$REPO"
echo "git_commit=$(git rev-parse HEAD)"
"$PY" -u -B RL_utils/01_RL.py --config "$JOB_DIR/task.yml" --resume-if-exists
'''


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, action='append', required=True,
                        help='Task YAML; repeat for independent jobs')
    parser.add_argument('--run-id', default='mdrl-dual-985000-' +
                        datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    parser.add_argument('--output-root', type=Path, default=Path(OUTPUT_ROOT))
    parser.add_argument('--queue', default='gentai')
    parser.add_argument('--resources', required=True, help='User-selected PBS resource string')
    parser.add_argument('--gpu', type=nonnegative_integer, required=True,
                        help='Physical GPU index written into CUDA_VISIBLE_DEVICES')
    parser.add_argument('--workers', type=positive_integer)
    parser.add_argument('--vina-cpu', type=positive_integer)
    parser.add_argument('--predictor-threads', type=positive_integer)
    parser.add_argument('--train-batch-size', type=positive_integer)
    parser.add_argument('--sample-batch-size', type=positive_integer)
    parser.add_argument('--data-workers', type=nonnegative_integer)
    args = parser.parse_args(argv)
    if sys.platform != 'linux':
        parser.error('Generate PBS on the Linux server after pulling the code')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', args.run_id):
        parser.error('Invalid run ID')

    group = args.output_root.resolve() / args.run_id
    prepared = []
    for filename in args.config:
        config = load_task_config(filename)
        directory = group / config['name']
        config['output_dir'] = str(directory / 'results')
        config['device'] = 'cuda:0'
        for section, key, value in (
                ('docking', 'workers', args.workers),
                ('docking', 'cpu', args.vina_cpu),
                ('scoring', 'predictor_threads', args.predictor_threads),
                ('train_overrides', 'batch_size', args.train_batch_size),
                ('sample_overrides', 'batch_size', args.sample_batch_size),
                ('train_overrides', 'num_workers', args.data_workers)):
            if value is not None:
                config.setdefault(section, {})[key] = value
        source = render_pbs(ROOT, directory, config['name'], args.queue, args.resources, args.gpu)
        prepared.append((filename, config, directory, source))

    if len({item[1]['name'] for item in prepared}) != len(prepared):
        parser.error('Task names must be unique')
    initializations = {(item[1]['initial_checkpoint'], item[1]['expected_pretrain_iteration'])
                       for item in prepared}
    if len(initializations) != 1:
        parser.error('All tasks must initialize from the same pretrained checkpoint and iteration')
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()

    group.mkdir(parents=True, exist_ok=False)
    (group / 'protocol.json').write_text(json.dumps(dict(
        git_commit=revision, queue=args.queue, resources=args.resources, physical_gpu=args.gpu,
        tasks=[item[1]['name'] for item in prepared], submitted=False), indent=2) + '\n')
    for filename, config, directory, source in prepared:
        directory.mkdir()
        shutil.copyfile(filename, directory / 'task_source.yml')
        with (directory / 'task.yml').open('x', encoding='utf-8') as handle:
            yaml.safe_dump(config, handle, sort_keys=False)
        with (directory / 'job.pbs').open('x', encoding='utf-8', newline='\n') as handle:
            handle.write(source)
        print('TASK=' + config['name'])
        print('OUTPUT=' + config['output_dir'])
        print('PBS=' + str(directory / 'job.pbs'))
        print('qsub ' + shlex.quote(str(directory / 'job.pbs')))
    print('Prepared only; no jobs submitted. GPU selection does not reserve a GPU in PBS.')
    print('Resubmitting the same PBS resumes its saved run; use a new run ID for independent runs.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
