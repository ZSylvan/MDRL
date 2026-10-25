"""Prepare one-pass batch-16/24 diagnostics using the existing trainer; never submit."""

import argparse
import copy
import json
import re
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml

from scripts.probe_pretrain_batch import check_dataset_files, is_cuda_oom, write_json
from utils.pretrain_record import git_version


PYTHON = '/home/zhuenqiang_1/.conda/envs/MDRL/bin/python'
OUTPUT_ROOT = '/home/zhuenqiang_1/Student_Project/zms/experiment_data/mdrl/epoch_probe'
BATCHES = (16, 24)


def epoch_configs(base, train_size, directory):
    if train_size <= 0:
        raise ValueError('The effective original training split is empty')
    configs = {}
    for batch in BATCHES:
        config = copy.deepcopy(base)
        config['train']['batch_size'] = batch
        config['train']['max_iters'] = (train_size + batch - 1) // batch
        config['logdir'] = str(Path(directory) / ('batch_%03d' % batch))
        configs[batch] = config
    return configs


def summarize(directory):
    directory = Path(directory)
    protocol = json.loads((directory / 'protocol.json').read_text(encoding='utf-8'))
    rows = []
    for batch in BATCHES:
        output = directory / ('batch_%03d' % batch)
        states = list(output.glob('*/run.json'))
        row = {'batch_size': batch, 'status': 'incomplete',
               'expected_iterations': protocol['iterations'][str(batch)]}
        if len(states) == 1:
            state = json.loads(states[0].read_text(encoding='utf-8'))
            failures = states[0].parent / 'failures.jsonl'
            events = [json.loads(line) for line in failures.read_text(encoding='utf-8').splitlines()]
            oom = sum(is_cuda_oom(RuntimeError(event.get('error', ''))) for event in events)
            exit_path = directory / ('batch_%03d.exit' % batch)
            code = int(exit_path.read_text()) if exit_path.is_file() else None
            row.update(run_json=str(states[0]), iteration=state['iteration'],
                       completed_train_calls=state['completed_train_calls'],
                       skipped_iterations=state['skipped_iterations'],
                       recorded_status=state['status'], cuda_oom_events=oom, returncode=code)
            if oom:
                row['status'] = 'oom'
            elif (code == 0 and state['status'] == 'completed' and
                  state['iteration'] == row['expected_iterations'] and
                  state['completed_train_calls'] == row['expected_iterations'] and
                  state['skipped_iterations'] == 0):
                row['status'] = 'passed_one_train_pass_and_validation'
            elif code is not None:
                row['status'] = 'error_or_incomplete'
        rows.append(row)
    report = {'experiment': 'mdrl-exp003-one-train-pass',
              'train_size': protocol['train_size'], 'results': rows}
    write_json(directory / 'summary.json', report)
    print(json.dumps(report, indent=2))
    return 0 if all(row['status'] == 'passed_one_train_pass_and_validation' for row in rows) else 2


def render_pbs(repo, directory, revision, queue, resources, gpu):
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', queue):
        raise ValueError('Invalid queue')
    if not re.fullmatch(r'[A-Za-z0-9_.,:=+-]+', resources):
        raise ValueError('Invalid resources')
    if not re.fullmatch(r'[A-Za-z0-9_/.-]+', str(directory)) or not PurePosixPath(str(directory)).is_absolute():
        raise ValueError('Output must be an absolute POSIX path without spaces')
    if gpu < 0:
        raise ValueError('Physical GPU must be nonnegative')
    quote = shlex.quote
    return f'''#!/bin/bash
#PBS -N mdrl-epoch003
#PBS -q {queue}
#PBS -l {resources}
#PBS -j oe
#PBS -o {directory}/pbs.out
set -euo pipefail
set -o noclobber
JOB_DIR={quote(str(directory))}
REPO={quote(str(repo))}
REV={quote(revision)}
GPU={gpu}
PY={PYTHON}
ENV_ROOT=/home/zhuenqiang_1/.conda/envs/MDRL
exec > "$JOB_DIR/launcher.log" 2>&1
echo "launcher_start=$(date -Is) host=$(hostname) job=${{PBS_JOBID:-unknown}}"
trap 'rc=$?; echo "launcher_exit_status=$rc"; echo "launcher_end=$(date -Is)"; exit "$rc"' EXIT
export PATH="$ENV_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$ENV_ROOT/lib:${{LD_LIBRARY_PATH:-}}"
export CONDA_PREFIX="$ENV_ROOT" CONDA_DEFAULT_ENV=MDRL
export PYTHONPATH="$REPO${{PYTHONPATH:+:$PYTHONPATH}}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
if [[ -n "${{CUDA_VISIBLE_DEVICES:-}}" && "$CUDA_VISIBLE_DEVICES" != "$GPU" ]]; then
    echo 'ERROR: scheduler GPU mask differs from the requested physical GPU'; exit 2
fi
export CUDA_VISIBLE_DEVICES="$GPU"
cd "$REPO"
test -x "$PY"
[[ "$(git rev-parse HEAD)" == "$REV" && -z "$(git status --porcelain)" ]] || {{
    echo 'ERROR: repository changed after preparation'; exit 2;
}}
for batch in 16 24; do
    pids=$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader)
    [[ -z "$pids" ]] || {{ echo "ERROR: GPU $GPU has active compute processes: $pids"; exit 2; }}
    printf -v label 'batch_%03d' "$batch"
    echo "candidate_start batch=$batch time=$(date -Is)"
    nvidia-smi -i "$GPU"
    rc=0
    "$PY" -u -B scripts/train_drug3d.py --config "$JOB_DIR/$label.yml" --device cuda:0 || rc=$?
    printf '%s\n' "$rc" > "$JOB_DIR/$label.exit"
    echo "candidate_end batch=$batch exit=$rc time=$(date -Is)"
done
"$PY" -B scripts/prepare_epoch_probe_pbs.py --summarize "$JOB_DIR"
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summarize', type=Path, help='Only summarize an existing run')
    parser.add_argument('--queue')
    parser.add_argument('--resources')
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--run-id', default='mdrl-exp003-epoch-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    args = parser.parse_args()
    if args.summarize:
        return summarize(args.summarize)
    if sys.platform != 'linux':
        parser.error('Generate actual PBS files on the Linux server after pulling committed code')
    if args.queue is None or args.resources is None or args.gpu is None:
        parser.error('--queue, --resources and --gpu are required')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', args.run_id):
        parser.error('Invalid run ID')
    version = git_version(ROOT)
    if not version.get('commit') or version.get('dirty'):
        parser.error('Use a clean, committed repository')
    from utils.misc import load_config
    import torch
    config_path = ROOT / 'configs/train/train_MDRL_reproduction.yml'
    config = load_config(config_path)
    check_dataset_files(config.dataset)
    root = Path(config.dataset.root)
    index_path = Path(str(root / config.dataset.path_dict.processed).split('.lmdb', 1)[0] + '_molid2idx.pt')
    index = torch.load(index_path, map_location='cpu')
    split = torch.load(root / config.dataset.split, map_location='cpu')
    train_size = sum(mol_id in index for mol_id in split['train'])
    directory = Path(OUTPUT_ROOT) / args.run_id
    base = json.loads(json.dumps(config))
    configs = epoch_configs(base, train_size, directory)
    source = render_pbs(ROOT, directory, version['commit'], args.queue, args.resources, args.gpu)
    directory.mkdir(parents=True, exist_ok=False)
    for batch, effective in configs.items():
        with (directory / ('batch_%03d.yml' % batch)).open('x', encoding='utf-8') as handle:
            yaml.safe_dump(effective, handle, sort_keys=False)
    write_json(directory / 'protocol.json', {
        'experiment': 'mdrl-exp003-one-train-pass', 'git': version,
        'train_size': train_size, 'raw_train_split_entries': len(split['train']),
        'iterations': {str(batch): effective['train']['max_iters'] for batch, effective in configs.items()},
        'shuffle': True, 'drop_last': False, 'validation': 'original cadence plus final iteration',
        'oom_policy': 'original trainer logs/skips training RuntimeErrors; any skip fails this diagnostic',
        'amp_note': 'completed_train_calls are not guaranteed optimizer updates',
        'gpu': args.gpu, 'queue': args.queue, 'resources': args.resources,
    })
    with (directory / 'job.pbs').open('x', encoding='utf-8', newline='\n') as handle:
        handle.write(source)
    print('train_size=%d' % train_size)
    for batch, effective in configs.items():
        print('batch=%d iterations=%d' % (batch, effective['train']['max_iters']))
    print('PBS=' + str(directory / 'job.pbs'))
    print('No job was submitted. Physical GPU selection does not reserve a GPU in PBS.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
