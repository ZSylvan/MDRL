"""Stop on nonfinite training numbers without changing finite updates."""

import math

import torch


class NonfiniteTrainingError(FloatingPointError):
    def __init__(self, stage, fields, values, context):
        self.stage = stage
        self.fields = fields
        self.values = values
        self.context = context
        super().__init__('Nonfinite values at %s: %s' % (stage, ', '.join(fields)))


def require_finite(stage, values, context):
    checks = torch.stack([torch.isfinite(value.detach()).all() for value in values.values()])
    if not bool(checks.all()):
        fields = [key for key, ok in zip(values, checks.cpu().tolist()) if not ok]
        raise NonfiniteTrainingError(stage, fields, values, context)


def cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    return value


def _nonfinite_paths(value, prefix):
    if torch.is_tensor(value):
        if not bool(torch.isfinite(value).all()):
            yield prefix
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _nonfinite_paths(item, '%s.%s' % (prefix, key))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _nonfinite_paths(item, '%s.%d' % (prefix, index))
    elif isinstance(value, float) and not math.isfinite(value):
        yield prefix


def validate_resume_numerics(checkpoint):
    # Scheduler min-mode sentinels may legitimately be +inf; do not reject them.
    for section in ('model', 'optimizer', 'scaler'):
        bad = next(_nonfinite_paths(checkpoint[section], section), None)
        if bad is not None:
            raise ValueError('Nonfinite resume state: ' + bad)
    if checkpoint['config']['train'].get('use_amp', False):
        scale = checkpoint['scaler'].get('scale')
        if not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
            raise ValueError('Invalid resume AMP scale; expected a finite positive value')
