"""Artifacts for one pretraining run; no model or optimization logic."""

import json
import math
import os
import shutil
import subprocess
import traceback
from datetime import datetime, timezone

import torch

from utils.pretrain_checks import cpu_tree


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
    def __init__(self, log_dir, metadata, periodic_checkpoint_limit=MAX_PERIODIC_CHECKPOINTS,
                 pinned_iterations=()):
        pinned_iterations = sorted(set(pinned_iterations))
        if (type(periodic_checkpoint_limit) is not int or periodic_checkpoint_limit < 1 or
                any(type(it) is not int or it <= 0 for it in pinned_iterations) or
                len(pinned_iterations) >= periodic_checkpoint_limit):
            raise ValueError('Checkpoint capacity must leave room beyond the pinned iterations')
        self.log_dir = log_dir
        self.state = {
            'run_id': os.path.basename(os.path.abspath(log_dir)),
            'status': 'running',
            'phase': 'setup',
            'started_at': _now(),
            'iteration': 0,
            'completed_train_calls': 0,
            'skipped_iterations': 0,
            'amp_skipped_updates': 0,
            'amp_tracking_start_iteration': 1,
            'fatal_errors': 0,
            'best': None,
            'last_checkpoint': None,
            'selection_rule': SELECTION_RULE,
            'periodic_checkpoint_limit': periodic_checkpoint_limit,
            'max_checkpoint_files': periodic_checkpoint_limit + 2,
            'pinned_iterations': pinned_iterations,
            'periodic_checkpoints': [],
            'metadata': metadata,
        }
        for name in ('validation.jsonl', 'failures.jsonl', 'checkpoints.jsonl', 'amp_events.jsonl'):
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

    def record_numerical_failure(self, exc, checkpoint, gradients):
        self._save_numerical_snapshot(exc, checkpoint, gradients, 'first_failure', 'numerical_failure')

    def record_amp_skip(self, event, checkpoint, gradients):
        self.state['amp_skipped_updates'] += 1
        first = 'first_amp_skip' not in self.state
        details = {'optimizer_step_skipped': True, 'scale_before': event.scale_before,
                   'scale_after': event.scale_after}
        if first:
            self._save_numerical_snapshot(event.error, checkpoint, gradients,
                                          'first_amp_skip', 'first_amp_skip', details)
        self._append('amp_events.jsonl', {
            'time': _now(), 'iteration': self.state['iteration'],
            'stage': event.error.stage, 'nonfinite_fields': event.error.fields,
            'losses': {key: value.item() for key, value in event.error.context['losses'].items()},
            'amp_skipped_updates': self.state['amp_skipped_updates'],
            'snapshot': 'diagnostics/first_amp_skip.pt' if first else None,
            **details,
        })
        self._write_state()

    def _save_numerical_snapshot(self, exc, checkpoint, gradients, stem, state_key, details=None):
        directory = os.path.join(self.log_dir, 'diagnostics')
        os.makedirs(directory, exist_ok=True)
        event = {
            'time': _now(), 'iteration': self.state['iteration'],
            'phase': self.state['phase'], 'stage': exc.stage,
            'nonfinite_fields': exc.fields,
            'validation_batch_index': exc.context.get('validation_batch_index'),
            'snapshot': 'diagnostics/%s.pt' % stem,
            'gradient_snapshot_stage': ('after_clipping_before_optimizer_step'
                                        if exc.stage == 'train_gradient_norm' else None),
            'last_checkpoint': self.state['last_checkpoint'],
            'best': self.state['best'], 'snapshot_saved': False,
        }
        event.update(details or {})
        self.state[state_key] = event
        context = dict(exc.context)
        batch = cpu_tree(context.pop('batch').to_dict())
        ids = batch.get('mol_id')
        event['molecule_ids'] = ids.reshape(-1).tolist() if torch.is_tensor(ids) else ids
        payload = cpu_tree({
            'diagnostic_only': True, 'iteration': self.state['iteration'],
            'stage': exc.stage, 'nonfinite_fields': exc.fields,
            'batch': batch, 'context': context, 'values': exc.values,
            'training_state': checkpoint, 'gradients': gradients,
            'run_id': self.state['run_id'], 'metadata': self.state['metadata'],
            'completed_train_calls': self.state['completed_train_calls'],
            'skipped_iterations': self.state['skipped_iterations'],
            'amp_skipped_updates': self.state['amp_skipped_updates'],
            'event_details': details,
        })
        with open(os.path.join(directory, stem + '.pt'), 'xb') as handle:
            torch.save(payload, handle)
        event['snapshot_saved'] = True
        with open(os.path.join(directory, stem + '.json'), 'x', encoding='utf-8') as handle:
            json.dump(event, handle, indent=2, allow_nan=False)
            handle.write('\n')
        self._write_state()

    def resume_from(self, checkpoint, source_path, best_path):
        # A new segment owns only its new periodic files; never prune the parent.
        self.state.update({
            'iteration': checkpoint['iteration'],
            'completed_train_calls': checkpoint['completed_train_calls'],
            'skipped_iterations': checkpoint['skipped_iterations'],
            'amp_skipped_updates': checkpoint.get('amp_skipped_updates', 0),
            'amp_tracking_start_iteration': checkpoint.get('amp_tracking_start_iteration',
                                                           checkpoint['iteration'] + 1),
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

    def save_checkpoint(self, checkpoint, losses=None, keep_periodic=True):
        if losses is not None and not keep_periodic:
            raise ValueError('Validation checkpoints must retain their numbered file')
        iteration = checkpoint['iteration']
        loss = float(losses['loss']) if losses is not None else float('nan')
        best = self.state['best']
        improved = math.isfinite(loss) and (best is None or loss < best['loss'])
        relative_path = ('checkpoints/%d.pt' % iteration if keep_periodic else 'checkpoints/last.pt')
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
            'amp_skipped_updates': self.state['amp_skipped_updates'],
            'amp_tracking_start_iteration': self.state['amp_tracking_start_iteration'],
            'selection_rule': SELECTION_RULE,
            'periodic_checkpoint_limit': self.state['periodic_checkpoint_limit'],
            'pinned_iterations': self.state['pinned_iterations'],
            'checkpoint_kind': ('validation' if losses is not None else
                                'milestone' if keep_periodic else 'recovery'),
            'best': best,
        })
        path = os.path.join(self.log_dir, relative_path)
        torch.save(payload, path + '.tmp')
        # Stage fully before retiring an owned, unpinned periodic checkpoint.
        pruned = []
        retained = self.state['periodic_checkpoints']
        if keep_periodic and len(retained) == self.state['periodic_checkpoint_limit']:
            oldest = next(path for path in retained
                          if int(os.path.basename(path).split('.')[0]) not in self.state['pinned_iterations'])
            os.remove(os.path.join(self.log_dir, oldest))
            retained.remove(oldest)
            pruned.append(oldest)
        os.replace(path + '.tmp', path)
        if keep_periodic:
            retained.append(relative_path)
        # Best/last copies remain independent of periodic file retention.
        aliases = ['last.pt', 'best.pt'] if improved else ['last.pt']
        for name in (aliases if keep_periodic else []):
            self._copy_checkpoint(path, name)
            if name == 'last.pt':
                self.state['last_checkpoint'] = {'iteration': iteration, 'checkpoint': relative_path}
            else:
                self.state['best'] = best
        if not keep_periodic:
            self.state['last_checkpoint'] = {'iteration': iteration, 'checkpoint': relative_path}
        self._append('checkpoints.jsonl', {
            'time': _now(), 'iteration': iteration, 'kind': payload['checkpoint_kind'],
            'checkpoint': relative_path, 'pruned_checkpoints': pruned,
        })
        if losses is not None:
            self._append('validation.jsonl', {
                'time': _now(), 'iteration': iteration,
                'losses': {key: float(value) if math.isfinite(value) else None
                           for key, value in losses.items()},
                'nonfinite_metrics': [key for key, value in losses.items() if not math.isfinite(value)],
                'is_best': improved, 'checkpoint': relative_path,
                'pruned_checkpoints': pruned,
                'completed_train_calls': self.state['completed_train_calls'],
                'skipped_iterations': self.state['skipped_iterations'],
                'amp_skipped_updates': self.state['amp_skipped_updates'],
            })
        self._write_state()
        return improved
