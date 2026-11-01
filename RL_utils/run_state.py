"""Persist completed RL work without importing models or initializing CUDA."""

import json
import os
from pathlib import Path


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def save_state(path, value):
    import torch
    from utils.pretrain_checks import cpu_tree

    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('wb') as handle:
        torch.save(cpu_tree(value), handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_state(path):
    from utils.pretrain_resume import load_checkpoint

    return load_checkpoint(str(path))


def capture_rng(device):
    import random
    import numpy as np
    import torch

    state = dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state())
    state['cuda'] = torch.cuda.get_rng_state(device) if torch.device(device).type == 'cuda' else None
    return state


def restore_rng(state, device):
    import random
    import numpy as np
    import torch

    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda'] is not None and torch.device(device).type == 'cuda':
        torch.cuda.set_rng_state(state['cuda'], device)


def read_records(path):
    """Discard only an unfinished final append left by an interrupted writer."""
    path = Path(path)
    if not path.exists():
        return []
    records = []
    with path.open('r+b') as handle:
        while True:
            offset = handle.tell()
            line = handle.readline()
            if not line:
                break
            if not line.endswith(b'\n'):
                handle.truncate(offset)
                break
            records.append(json.loads(line))
    return records


def find_training_checkpoint(directory, epoch):
    candidates = []
    for path in Path(directory).glob('*/checkpoints/epoch%d_it*.pt' % epoch):
        iteration = int(path.stem.rsplit('_it', 1)[1])
        candidates.append((iteration, path))
    return str(max(candidates)[1]) if candidates else None


def validate_round_checkpoint(saved, config, root, epoch, source_checkpoint):
    from utils.pretrain_checks import validate_resume_numerics

    required = {'config', 'model', 'optimizer', 'scheduler', 'scaler', 'iteration', 'round',
                'source_checkpoint', 'dataset_root', 'dataset_split', 'rng_state', 'amp_skipped_updates'}
    if (not isinstance(saved, dict) or not required <= saved.keys() or
            saved.get('diagnostic_only') or saved['round'] != epoch or saved['config'] != config or
            saved['dataset_root'] != root or saved['source_checkpoint'] != source_checkpoint or
            type(saved['iteration']) is not int or not 0 <= saved['iteration'] <= config['train']['max_iters']):
        raise ValueError('Checkpoint is not a compatible checkpoint for this RL round')
    validate_resume_numerics(saved)


def restore_training_components(saved, optimizer, scheduler, scaler, device):
    optimizer.load_state_dict(saved['optimizer'])
    scheduler.load_state_dict(saved['scheduler'])
    scaler.load_state_dict(saved['scaler'])
    restore_rng(saved['rng_state'], device)


def read_sampling_batches(directory, identity):
    directory = Path(directory)
    metadata = directory / 'identity.json'
    if metadata.exists():
        if json.loads(metadata.read_text()) != identity:
            raise ValueError('Sampling resume settings differ from the saved batch journal')
    else:
        directory.mkdir(parents=True, exist_ok=True)
        write_json(metadata, identity)
    molecules, failed, rng = [], 0, None
    paths = sorted(directory.glob('batch_*.pt'))
    for index, path in enumerate(paths):
        if path.name != 'batch_%06d.pt' % index:
            raise ValueError('Sampling journal has a missing batch: ' + str(path))
        saved = load_state(path)
        molecules.extend(saved['molecules'])
        failed, rng = saved['failed_total'], saved['rng']
    return molecules, failed, rng, len(paths)
