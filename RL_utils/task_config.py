"""Read a dual-target task without importing models or starting work."""

import copy
import csv
import json
import math
from pathlib import Path, PurePosixPath

import yaml


def _path(value, parent):
    path = Path(value).expanduser()
    if PurePosixPath(str(value)).is_absolute():
        return str(value)
    if path.is_absolute():
        return str(path)
    return str((parent / path).resolve())


def _modifier(value):
    if not math.isfinite(float(value['mu'])) or not math.isfinite(float(value['sigma'])) or value['sigma'] <= 0:
        raise ValueError('Normalization needs finite mu and positive sigma')
    if type(value['minimize']) is not bool:
        raise ValueError('minimize must be a boolean')


def load_task_config(filename, check_files=True):
    filename = Path(filename).resolve()
    with filename.open() as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError('Task config must be a mapping')
    config = copy.deepcopy(config)
    if type(config['seed']) is not int or not 0 <= config['seed'] < 2 ** 32:
        raise ValueError('seed must be an integer in [0, 2**32)')
    if type(config['expected_pretrain_iteration']) is not int or config['expected_pretrain_iteration'] <= 0:
        raise ValueError('expected_pretrain_iteration must be a positive integer')
    for field, positive, nonnegative in (
            ('train_overrides', ('batch_size', 'max_iters', 'val_freq'), ('num_workers',)),
            ('sample_overrides', ('batch_size',), ())):
        for key in positive + nonnegative:
            if key in config.get(field, {}):
                value = config[field][key]
                if type(value) is not int or value < (1 if key in positive else 0):
                    raise ValueError('%s.%s has an invalid value' % (field, key))
    files = []
    for key in ('initial_checkpoint', 'train_config', 'sample_config', 'fscores'):
        config[key] = _path(config[key], filename.parent)
        files.append(config[key])
    config['output_dir'] = _path(config['output_dir'], filename.parent)
    targets = config['targets']
    if len(targets) != 2 or len({target['name'] for target in targets}) != 2:
        raise ValueError('Exactly two uniquely named targets are required')
    for target in targets:
        if not target['name'] or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in target['name']):
            raise ValueError('Target names must be safe directory names')
        for key in ('receptor', 'ligand_model'):
            target[key] = _path(target[key], filename.parent)
            files.append(target[key])
        for key in ('center', 'box_size'):
            values = target[key]
            if len(values) != 3 or not all(math.isfinite(float(v)) for v in values):
                raise ValueError('%s must contain three finite numbers' % key)
        if any(float(v) <= 0 for v in target['box_size']):
            raise ValueError('Box dimensions must be positive')
        _modifier(target['normalization'])
    for key in ('qed', 'sa', 'ligand_efficiency'):
        _modifier(config['scoring'][key])
    threads = config['scoring'].get('predictor_threads', 1)
    if type(threads) is not int or threads <= 0:
        raise ValueError('scoring.predictor_threads must be a positive integer')
    for key in ('n_epochs', 'initial_num_mols', 'new_num_mols_per_epoch', 'top_k'):
        if type(config['rl'][key]) is not int or config['rl'][key] <= 0:
            raise ValueError('%s must be a positive integer' % key)
    if config['rl']['n_epochs'] < 2:
        raise ValueError('n_epochs must be at least two (original loop runs n_epochs - 1 updates)')
    docking = config['docking']
    docking['executable'] = _path(docking['executable'], filename.parent)
    files.append(docking['executable'])
    for key in ('workers', 'cpu', 'exhaustiveness', 'num_modes'):
        if type(docking[key]) is not int or docking[key] <= 0:
            raise ValueError('docking.%s must be a positive integer' % key)
    if docking.get('timeout') is not None and (not math.isfinite(float(docking['timeout'])) or docking['timeout'] <= 0):
        raise ValueError('Docking timeout must be null or a positive number')
    if check_files:
        for path in files:
            if not Path(path).is_file():
                raise FileNotFoundError(path)
    return config


def write_scoring_csv(config, filename):
    fields = ['category', 'name', 'minimize', 'mu', 'sigma', 'file', 'model', 'n_top', 'agg']
    rows = []
    for name in ('qed', 'sa'):
        rows.append(dict(category=name, name=name, **config['scoring'][name]))
    for target in config['targets']:
        rows.append(dict(category='ligand_efficiency', name=target['name'] + '_le',
                         file=target['ligand_model'], **config['scoring']['ligand_efficiency']))
    with Path(filename).open('x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def append_record(filename, record):
    with Path(filename).open('a') as handle:
        handle.write(json.dumps(record, allow_nan=False) + '\n')
