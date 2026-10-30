"""Prepare an independent batch-16 pretraining job; never submit it."""

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

from scripts.probe_pretrain_batch import check_dataset_files, write_json
from utils.pretrain_record import git_version
from utils.pretrain_resume import find_resume_best, load_checkpoint, load_resume_checkpoint


PYTHON = '/home/zhuenqiang_1/.conda/envs/MDRL/bin/python'
OUTPUT_ROOT = '/home/zhuenqiang_1/Student_Project/zms/experiment_data/mdrl/pretrain'


def prepare_config(base, train_size, directory):
    config = copy.deepcopy(base)
    train = config['train']
    for value in (train_size, train['batch_size'], train['val_every_epochs'], train['checkpoint_freq']):
        if type(value) is not int or value <= 0:
            raise ValueError('Training size, batch, epoch interval and recovery interval must be positive')
    if not 3 <= train['max_checkpoint_files'] <= 50:
        raise ValueError('Total checkpoint capacity must be between 3 and 50')
    milestones = train['checkpoint_milestones']
    if (any(type(it) is not int or not 0 < it <= train['max_iters'] for it in milestones) or
            len(set(milestones)) != len(milestones) or
            len(milestones) >= train['max_checkpoint_files'] - 2):
        raise ValueError('Invalid pinned checkpoint milestones or insufficient checkpoint capacity')
    steps = (train_size + train['batch_size'] - 1) // train['batch_size']
    train['val_freq'] = steps * train['val_every_epochs']
    config['logdir'] = str(Path(directory) / 'training')
    return config


def render_pbs(repo, directory, revision, queue, resources, resume=None, allow_shared_gpu=False):
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', queue):
        raise ValueError('Invalid queue')
    if not re.fullmatch(r'[A-Za-z0-9_.,:=+-]+', resources):
        raise ValueError('Invalid resources')
    if (not re.fullmatch(r'[A-Za-z0-9_/.-]+', str(directory)) or
            not PurePosixPath(str(directory)).is_absolute()):
        raise ValueError('Output must be an absolute POSIX path without spaces')
    quote = shlex.quote
    resume_arg = ' --resume ' + quote(str(resume)) if resume is not None else ''
    return f'''#!/bin/bash
#PBS -N mdrl-pre-b16
#PBS -q {queue}
#PBS -l {resources}
#PBS -j oe
#PBS -o {directory}/pbs.out
set -euo pipefail
set -o noclobber
JOB_DIR={quote(str(directory))}
REPO={quote(str(repo))}
REV={quote(revision)}
PY={PYTHON}
ENV_ROOT=/home/zhuenqiang_1/.conda/envs/MDRL
ALLOW_SHARED_GPU={int(allow_shared_gpu)}
exec > "$JOB_DIR/launcher.log" 2>&1
echo "launcher_start=$(date -Is) host=$(hostname) job=${{PBS_JOBID:-unknown}}"
trap 'rc=$?; echo "launcher_exit_status=$rc"; echo "launcher_end=$(date -Is)"; exit "$rc"' EXIT
if [[ -r "${{PBS_NODEFILE:-}}" ]]; then
    echo "allocated_cpu_slots=$(wc -l < "$PBS_NODEFILE")"
    echo 'allocated_nodes:'
    cat "$PBS_NODEFILE"
fi
export PATH="$ENV_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$ENV_ROOT/lib:${{LD_LIBRARY_PATH:-}}"
export CONDA_PREFIX="$ENV_ROOT" CONDA_DEFAULT_ENV=MDRL
export PYTHONPATH="$REPO${{PYTHONPATH:+:$PYTHONPATH}}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
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
        echo "ERROR: GPU 0 has active compute processes: $pids"; exit 2
    fi
    echo "WARNING: GPU sharing explicitly allowed; existing compute PIDs: $pids"
fi
nvidia-smi -i 0
"$PY" -B -c "import torch; print('torch=', torch.__version__, 'cuda=', torch.version.cuda, 'gpu=', torch.cuda.get_device_name(0))"
"$PY" -u -B scripts/train_drug3d.py --config "$JOB_DIR/config.yml" --logdir "$JOB_DIR/training" --device cuda:0{resume_arg}
"$PY" -B - "$JOB_DIR" <<'PY'
import json
import sys
from pathlib import Path
paths = list((Path(sys.argv[1]) / 'training').glob('*/run.json'))
if len(paths) != 1:
    raise RuntimeError('Expected exactly one recorded training segment')
state = json.loads(paths[0].read_text())
print('training_status=', state['status'], 'iteration=', state['iteration'],
      'skipped_iterations=', state['skipped_iterations'])
sys.exit(0 if state['status'] == 'completed' and state['skipped_iterations'] == 0 else 2)
PY
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue', default='gentai')
    parser.add_argument('--resources', default='nodes=node5:ppn=15,walltime=240:00:00')
    parser.add_argument('--resume', type=Path, help='Trusted complete checkpoint from this training recipe')
    parser.add_argument('--allow-shared-gpu', action='store_true',
                        help='Allow existing GPU compute processes; log them without stopping')
    parser.add_argument('--run-id', default='mdrl-node5-b16-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    args = parser.parse_args()
    if sys.platform != 'linux':
        parser.error('Generate PBS on the Linux server after pulling committed code')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', args.run_id):
        parser.error('Invalid run ID')
    version = git_version(ROOT)
    if not version.get('commit') or version.get('dirty'):
        parser.error('Use a clean, committed repository')
    base = yaml.safe_load((ROOT / 'configs/train/train_MDRL_node5_b16.yml').read_text())
    from easydict import EasyDict
    check_dataset_files(EasyDict(base['dataset']))
    root = Path(base['dataset']['root'])
    index_path = Path(str(root / base['dataset']['path_dict']['processed']).split('.lmdb', 1)[0] + '_molid2idx.pt')
    index = load_checkpoint(index_path)
    split = load_checkpoint(root / base['dataset']['split'])
    train_size = sum(mol_id in index for mol_id in split['train'])
    directory = Path(OUTPUT_ROOT) / args.run_id
    config = prepare_config(base, train_size, directory)
    resume = args.resume.resolve() if args.resume is not None else None
    resume_info = None
    if resume is not None:
        saved = load_resume_checkpoint(str(resume), config)
        best_path = find_resume_best(str(resume), saved)
        resume_info = {'checkpoint': str(resume), 'iteration': saved['iteration'],
                       'source_run_id': saved['run_id'], 'best_checkpoint': best_path}
    source = render_pbs(ROOT, directory, version['commit'], args.queue, args.resources, resume,
                        args.allow_shared_gpu)
    train = config['train']
    validations = list(range(train['val_freq'], train['max_iters'] + 1, train['val_freq']))
    if train['max_iters'] not in validations:
        validations.append(train['max_iters'])
    retained = sorted(set(validations) | set(train['checkpoint_milestones']))
    directory.mkdir(parents=True, exist_ok=False)
    with (directory / 'config.yml').open('x', encoding='utf-8') as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    write_json(directory / 'protocol.json', {
        'experiment': 'mdrl-node5-b16-pretraining', 'run_id': args.run_id, 'git': version,
        'physical_gpu': 0, 'queue': args.queue, 'resources': args.resources,
        'placement_note': 'Resources are requested at preparation; PBS directives may be edited. '
                          'Actual host and allocated slots are recorded in launcher.log',
        'allow_shared_gpu': args.allow_shared_gpu,
        'train_size': train_size, 'raw_train_split_entries': len(split['train']),
        'steps_per_epoch': (train_size + train['batch_size'] - 1) // train['batch_size'],
        'batch_size': train['batch_size'], 'num_workers': train['num_workers'],
        'max_iters': train['max_iters'], 'val_every_epochs': train['val_every_epochs'],
        'val_freq': train['val_freq'], 'validation_iterations': validations,
        'checkpoint_freq': train['checkpoint_freq'], 'checkpoint_milestones': train['checkpoint_milestones'],
        'planned_named_checkpoint_iterations': retained,
        'planned_files_without_resume': len(retained) + 2,
        'max_checkpoint_files_per_segment': train['max_checkpoint_files'],
        'resume': resume_info, 'resume_data_iterator_restored': False,
        'best_rule': 'lowest finite validation total loss; first occurrence on ties',
        'scheduler_note': 'Original validation-driven scheduler; called only at the new validation cadence',
        'milestone_note': 'Non-validation milestone does not affect best or scheduler',
        'training_error_policy': 'Nonfinite losses or gradient norms stop before further updates; '
                                 'other original RuntimeErrors are recorded/skipped; '
                                 'any skip is not a clean completion',
        'numerical_failure_artifacts': ['diagnostics/first_failure.json', 'diagnostics/first_failure.pt'],
        'numerical_failure_note': 'A stop records an observed nonfinite value; '
                                  'it does not by itself prove permanent weight corruption or its cause',
    })
    with (directory / 'job.pbs').open('x', encoding='utf-8', newline='\n') as handle:
        handle.write(source)
    print(json.dumps({'run': str(directory), 'train_size': train_size,
                      'val_freq': train['val_freq'], 'planned_files_without_resume': len(retained) + 2,
                      'resume': resume_info}, indent=2))
    print('PBS=' + str(directory / 'job.pbs'))
    print('No job submitted. Manual GPU selection does not reserve a GPU in PBS.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
