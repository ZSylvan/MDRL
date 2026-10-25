"""Artifacts for one pretraining run; no model or optimization logic."""

import json
import math
import os
import shutil
import subprocess
import traceback
from datetime import datetime, timezone

import torch


SELECTION_RULE = 'lowest finite validation total loss; first occurrence on ties'
MAX_PERIODIC_CHECKPOINTS = 50


def git_version(repo_dir):
    try:
        revision = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=repo_dir, text=True).strip()
        status = subprocess.check_output(
            ['git', 'status', '--porcelain'], cwd=repo_dir, text=True)
        return {'commit': revision, 'status': status, 'dirty': bool(status)}
    except (OSError, subprocess.CalledProcessError) as exc:
        return {'commit': None, 'error': str(exc)}


def _now():
    return datetime.now(timezone.utc).isoformat()


class PretrainRecord:
    def __init__(self, log_dir, metadata):
        self.log_dir = log_dir
        self.state = {
            'run_id': os.path.basename(os.path.abspath(log_dir)),
            'status': 'running',
            'phase': 'setup',
            'started_at': _now(),
            'iteration': 0,
            'completed_train_calls': 0,
            'skipped_iterations': 0,
            'fatal_errors': 0,
            'best': None,
            'last_checkpoint': None,
            'selection_rule': SELECTION_RULE,
            'periodic_checkpoint_limit': MAX_PERIODIC_CHECKPOINTS,
            'periodic_checkpoints': [],
            'metadata': metadata,
        }
        for name in ('validation.jsonl', 'failures.jsonl'):
            with open(os.path.join(log_dir, name), 'x', encoding='utf-8'):
                pass
        self._write_state()

    def _write_state(self):
        path = os.path.join(self.log_dir, 'run.json')
        with open(path + '.tmp', 'w', encoding='utf-8') as handle:
            json.dump(self.state, handle, indent=2, allow_nan=False)
            handle.write('\n')
        os.replace(path + '.tmp', path)

    def _append(self, name, event):
        with open(os.path.join(self.log_dir, name), 'a', encoding='utf-8') as handle:
            handle.write(json.dumps(event, allow_nan=False) + '\n')

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc is None:
            self.state['status'] = (
                'completed_with_skips' if self.state['skipped_iterations'] else 'completed')
            self.state['phase'] = 'finished'
        else:
            self.state['status'] = 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed'
            self.state['fatal_errors'] += int(not isinstance(exc, KeyboardInterrupt))
            self._append('failures.jsonl', {
                'time': _now(), 'iteration': self.state['iteration'],
                'phase': self.state['phase'], 'outcome': self.state['status'],
                'error_type': exc_type.__name__, 'error': str(exc),
                'traceback': ''.join(traceback.format_exception(exc_type, exc, tb)),
            })
        self.state['finished_at'] = _now()
        self._write_state()
        return False

    def record_skip(self, exc):
        self.state['skipped_iterations'] += 1
        self._append('failures.jsonl', {
            'time': _now(), 'iteration': self.state['iteration'],
            'phase': 'train', 'outcome': 'skipped',
            'error_type': type(exc).__name__, 'error': str(exc),
            'traceback': ''.join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        })
        self._write_state()

    def resume_from(self, checkpoint, source_path, best_path):
        # A new segment owns only its new periodic files; never prune the parent.
        self.state.update({
            'iteration': checkpoint['iteration'],
            'completed_train_calls': checkpoint['completed_train_calls'],
            'skipped_iterations': checkpoint['skipped_iterations'],
            'resume': {
                'checkpoint': os.path.abspath(source_path),
                'source_run_id': checkpoint['run_id'],
                'iteration': checkpoint['iteration'],
                'rng_state_available': checkpoint.get('rng_state') is not None,
                'data_iterator_restored': False,
            },
        })
        if checkpoint['best'] is not None:
            if best_path is None:
                raise ValueError('Historical best checkpoint is required')
            best = dict(checkpoint['best'])
            best.setdefault('run_id', checkpoint['run_id'])
            best['checkpoint'] = 'checkpoints/best.pt'
            self._copy_checkpoint(best_path, 'best.pt')
            self.state['best'] = best
        self._copy_checkpoint(source_path, 'last.pt')
        self.state['last_checkpoint'] = {
            'iteration': checkpoint['iteration'], 'checkpoint': 'checkpoints/last.pt',
        }
        self._write_state()

    def _copy_checkpoint(self, source, name):
        alias = os.path.join(self.log_dir, 'checkpoints', name)
        shutil.copyfile(source, alias + '.tmp')
        os.replace(alias + '.tmp', alias)

    def save_checkpoint(self, checkpoint, losses):
        iteration = checkpoint['iteration']
        loss = float(losses['loss'])
        best = self.state['best']
        improved = math.isfinite(loss) and (best is None or loss < best['loss'])
        relative_path = 'checkpoints/%d.pt' % iteration
        if improved:
            best = {
                'iteration': iteration, 'loss': loss,
                'checkpoint': 'checkpoints/best.pt', 'source_checkpoint': relative_path,
                'run_id': self.state['run_id'],
            }
        payload = dict(checkpoint)
        payload.update({
            'validation_losses': losses,
            'run_id': self.state['run_id'],
            'completed_train_calls': self.state['completed_train_calls'],
            'skipped_iterations': self.state['skipped_iterations'],
            'selection_rule': SELECTION_RULE,
            'periodic_checkpoint_limit': MAX_PERIODIC_CHECKPOINTS,
            'best': best,
        })
        path = os.path.join(self.log_dir, relative_path)
        torch.save(payload, path + '.tmp')
        # Stage the new file fully before retiring an owned periodic checkpoint.
        # Free its slot before rename, so at most 50 periodic .pt files exist.
        pruned = []
        retained = self.state['periodic_checkpoints']
        if len(retained) == MAX_PERIODIC_CHECKPOINTS:
            oldest = retained[0]
            os.remove(os.path.join(self.log_dir, oldest))
            pruned.append(retained.pop(0))
        os.replace(path + '.tmp', path)
        retained.append(relative_path)
        # Best/last copies remain independent of periodic file retention.
        for name in (['last.pt', 'best.pt'] if improved else ['last.pt']):
            self._copy_checkpoint(path, name)
            if name == 'last.pt':
                self.state['last_checkpoint'] = {'iteration': iteration, 'checkpoint': relative_path}
            else:
                self.state['best'] = best
        self._append('validation.jsonl', {
            'time': _now(), 'iteration': iteration,
            'losses': {key: float(value) if math.isfinite(value) else None
                       for key, value in losses.items()},
            'nonfinite_metrics': [key for key, value in losses.items() if not math.isfinite(value)],
            'is_best': improved, 'checkpoint': relative_path,
            'pruned_checkpoints': pruned,
            'completed_train_calls': self.state['completed_train_calls'],
            'skipped_iterations': self.state['skipped_iterations'],
        })
        self._write_state()
        return improved
