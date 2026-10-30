"""Resume trusted pretraining checkpoints without changing the training recipe."""

import copy
import inspect
import math
import os
import random

import numpy as np
import torch

from utils.pretrain_record import SELECTION_RULE
from utils.pretrain_checks import validate_resume_numerics


def load_checkpoint(path):
    # These checkpoints contain config/RNG objects, not just tensors. Load only
    # trusted files; PyTorch 1.10 does not have the weights_only argument.
    kwargs = {'map_location': 'cpu'}
    if 'weights_only' in inspect.signature(torch.load).parameters:
        kwargs['weights_only'] = False
    return torch.load(path, **kwargs)


def load_resume_checkpoint(path, config):
    checkpoint = load_checkpoint(path)
    if isinstance(checkpoint, dict) and checkpoint.get('diagnostic_only'):
        raise ValueError('Failure snapshots are diagnostic only, not resumable checkpoints')
    required = {'config', 'model', 'optimizer', 'scheduler', 'scaler', 'iteration',
                'run_id', 'best', 'selection_rule', 'completed_train_calls',
                'skipped_iterations'}
    if not isinstance(checkpoint, dict) or not required <= checkpoint.keys():
        raise ValueError('Resume requires a complete recorded pretraining checkpoint, '
                         'including optimizer, scheduler, AMP and selection state')
    iteration = checkpoint['iteration']
    if type(iteration) is not int or iteration < 0:
        raise ValueError('Invalid resume iteration')
    if config['train']['max_iters'] <= iteration:
        raise ValueError('max_iters is the total target and must exceed the saved iteration')
    if checkpoint['selection_rule'] != SELECTION_RULE:
        raise ValueError('Resume checkpoint uses a different best-selection rule')
    for key in ('model', 'optimizer', 'scheduler', 'scaler'):
        if not isinstance(checkpoint[key], dict):
            raise ValueError('Invalid resume state: ' + key)
    for key in ('completed_train_calls', 'skipped_iterations'):
        if type(checkpoint[key]) is not int or checkpoint[key] < 0:
            raise ValueError('Invalid resume counter: ' + key)
    if ('amp_skipped_updates' in checkpoint and
            (type(checkpoint['amp_skipped_updates']) is not int or checkpoint['amp_skipped_updates'] < 0)):
        raise ValueError('Invalid resume counter: amp_skipped_updates')
    if ('amp_tracking_start_iteration' in checkpoint and
            (type(checkpoint['amp_tracking_start_iteration']) is not int or
             not 1 <= checkpoint['amp_tracking_start_iteration'] <= iteration + 1)):
        raise ValueError('Invalid AMP tracking start iteration')
    saved = copy.deepcopy(checkpoint['config'])
    current = copy.deepcopy(config)
    for candidate in (saved, current):
        candidate.pop('logdir', None)
        candidate['dataset'].pop('root', None)
        candidate['train'].pop('max_iters', None)
    if saved != current:
        raise ValueError('Resume configuration differs; only dataset root, logdir and '
                         'total max_iters may change (batch size must match)')
    validate_resume_numerics(checkpoint)
    return checkpoint


def find_resume_best(path, checkpoint):
    """Find the actual historical best, including when resuming an older save."""
    best = checkpoint['best']
    if best is None:
        return None
    if (type(best.get('iteration')) is not int or
            not 0 <= best['iteration'] <= checkpoint['iteration'] or
            not math.isfinite(best['loss'])):
        raise ValueError('Invalid historical best metadata')
    expected_run = best.get('run_id', checkpoint['run_id'])
    if best['iteration'] == checkpoint['iteration']:
        candidates = [path]
    else:
        directory = os.path.dirname(os.path.abspath(path))
        candidates = [os.path.join(directory, '%d.pt' % best['iteration']),
                      os.path.join(directory, 'best.pt')]
    for candidate in candidates:
        if not os.path.isfile(candidate):
            continue
        saved = checkpoint if os.path.abspath(candidate) == os.path.abspath(path) else load_checkpoint(candidate)
        if (saved.get('iteration') == best['iteration'] and
                saved.get('run_id') == expected_run and
                (saved.get('validation_losses') or {}).get('loss') == best['loss']):
            return os.path.abspath(candidate)
    raise ValueError('Matching historical best checkpoint is missing; preserve the source '
                     'best.pt (or its periodic file). Refusing to reset best selection')


def capture_rng_state():
    return {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
        'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_training_state(checkpoint, model, optimizer, scheduler, scaler):
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    scaler.load_state_dict(checkpoint['scaler'])
    rng = checkpoint.get('rng_state')
    if rng is not None:
        cuda = rng['cuda']
        if cuda and len(cuda) != torch.cuda.device_count():
            raise ValueError('Resume RNG state requires the same number of visible CUDA devices')
        random.setstate(rng['python'])
        np.random.set_state(rng['numpy'])
        torch.set_rng_state(rng['torch'])
        if cuda:
            torch.cuda.set_rng_state_all(cuda)
    return checkpoint['iteration'] + 1
