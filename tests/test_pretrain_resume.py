import copy
import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import torch

from utils.pretrain_record import PretrainRecord, SELECTION_RULE
from utils.pretrain_resume import (capture_rng_state, find_resume_best, load_checkpoint,
                                   load_resume_checkpoint, restore_training_state)


class PretrainResumeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = {
            'model': {'name': 'toy'}, 'train': {'batch_size': 8, 'max_iters': 100},
            'dataset': {'root': '/data', 'split': 'original.pt'}, 'logdir': '/runs',
        }
        self.source = self.new_run('parent')

    def new_run(self, name):
        folder = self.root / name
        (folder / 'checkpoints').mkdir(parents=True)
        return folder

    def checkpoint(self, iteration):
        return {
            'config': self.config, 'model': {'weight': torch.tensor([iteration])},
            'optimizer': {}, 'scheduler': {}, 'scaler': {}, 'iteration': iteration,
        }

    def save_parent(self):
        with PretrainRecord(str(self.source), {}) as record:
            for iteration, loss in ((5, 1.0), (10, 2.0)):
                record.state.update(iteration=iteration,
                                    completed_train_calls=iteration - 1, skipped_iterations=1)
                record.save_checkpoint(self.checkpoint(iteration), {'loss': loss})
        return self.source / 'checkpoints/last.pt'

    def test_resume_in_new_directory_keeps_global_best_counters_and_parent(self):
        path = self.save_parent()
        before = {file: file.read_bytes() for file in self.source.rglob('*') if file.is_file()}
        checkpoint = load_resume_checkpoint(path, self.config)
        best_path = find_resume_best(path, checkpoint)
        child = self.new_run('child')
        with PretrainRecord(str(child), {}) as record:
            record.resume_from(checkpoint, path, best_path)
            self.assertEqual(record.state['iteration'], 10)
            self.assertEqual(record.state['completed_train_calls'], 9)
            self.assertEqual(record.state['skipped_iterations'], 1)
            self.assertEqual(record.state['periodic_checkpoints'], [])
            self.assertEqual(load_checkpoint(child / 'checkpoints/last.pt')['iteration'], 10)
            self.assertFalse(record.save_checkpoint(self.checkpoint(11), {'loss': 1.5}))
            self.assertEqual(load_checkpoint(child / 'checkpoints/best.pt')['iteration'], 5)
            # Exercise retention without ever modifying or pruning parent files.
            for iteration in range(12, 65):
                record.save_checkpoint(self.checkpoint(iteration), {'loss': 1.0})
            self.assertEqual(len(list((child / 'checkpoints').glob('*.pt'))), 52)
        self.assertEqual(before, {file: file.read_bytes() for file in self.source.rglob('*')
                                  if file.is_file()})
        state = json.loads((child / 'run.json').read_text())
        self.assertEqual(state['best']['run_id'], 'parent')
        self.assertEqual(state['resume']['source_run_id'], 'parent')
        self.assertEqual(state['resume']['iteration'], 10)
        self.assertFalse(state['resume']['data_iterator_restored'])
        # Resume a second segment: its best may still belong to the ancestor.
        path2 = child / 'checkpoints/last.pt'
        checkpoint2 = load_resume_checkpoint(path2, self.config)
        self.assertEqual(find_resume_best(path2, checkpoint2), str(child / 'checkpoints/best.pt'))

    def test_new_best_replaces_inherited_best_only_on_strict_improvement(self):
        path = self.save_parent()
        checkpoint = load_resume_checkpoint(path, self.config)
        child = self.new_run('child')
        with PretrainRecord(str(child), {}) as record:
            record.resume_from(checkpoint, path, find_resume_best(path, checkpoint))
            self.assertTrue(record.save_checkpoint(self.checkpoint(11), {'loss': 0.5}))
            self.assertEqual(record.state['best']['run_id'], 'child')
        saved = load_checkpoint(child / 'checkpoints/last.pt')
        self.assertEqual(saved['best']['iteration'], 11)
        self.assertEqual(find_resume_best(child / 'checkpoints/last.pt', saved),
                         str(child / 'checkpoints/last.pt'))

    def test_amp_skip_counter_roundtrips_without_changing_runtime_skip_count(self):
        path = self.save_parent()
        checkpoint = load_checkpoint(path)
        checkpoint.update(amp_skipped_updates=3, amp_tracking_start_iteration=4)
        torch.save(checkpoint, path)
        checkpoint = load_resume_checkpoint(path, self.config)
        child = self.new_run('amp-child')
        with PretrainRecord(str(child), {}) as record:
            record.resume_from(checkpoint, path, find_resume_best(path, checkpoint))
            self.assertEqual(record.state['amp_skipped_updates'], 3)
            self.assertEqual(record.state['amp_tracking_start_iteration'], 4)
            self.assertEqual(record.state['skipped_iterations'], 1)
            record.save_checkpoint(self.checkpoint(11), keep_periodic=False)
        saved = load_resume_checkpoint(child / 'checkpoints/last.pt', self.config)
        self.assertEqual(saved['amp_skipped_updates'], 3)
        self.assertEqual(saved['amp_tracking_start_iteration'], 4)

    def test_legacy_checkpoint_has_unknown_earlier_amp_history(self):
        path = self.save_parent()
        checkpoint = load_checkpoint(path)
        checkpoint.pop('amp_skipped_updates')
        checkpoint.pop('amp_tracking_start_iteration')
        torch.save(checkpoint, path)
        checkpoint = load_resume_checkpoint(path, self.config)
        child = self.new_run('legacy-child')
        with PretrainRecord(str(child), {}) as record:
            record.resume_from(checkpoint, path, find_resume_best(path, checkpoint))
            self.assertEqual(record.state['amp_skipped_updates'], 0)
            self.assertEqual(record.state['amp_tracking_start_iteration'], 11)

    def test_invalid_amp_history_counters_are_rejected(self):
        path = self.save_parent()
        original = load_checkpoint(path)
        for key, value in (('amp_skipped_updates', -1), ('amp_tracking_start_iteration', 12)):
            checkpoint = dict(original, **{key: value})
            torch.save(checkpoint, path)
            with self.subTest(key=key), self.assertRaises(ValueError):
                load_resume_checkpoint(path, self.config)

    def test_missing_or_newer_best_is_not_silently_relabelled(self):
        path = self.save_parent()
        checkpoint = load_resume_checkpoint(path, self.config)
        (self.source / 'checkpoints/5.pt').unlink()
        (self.source / 'checkpoints/best.pt').unlink()
        with self.assertRaisesRegex(ValueError, 'historical best'):
            find_resume_best(path, checkpoint)
        # A later best alias cannot represent the best at an earlier save.
        with PretrainRecord(str(self.new_run('later')), {}) as record:
            record.save_checkpoint(self.checkpoint(15), {'loss': 0.1})
        import shutil
        shutil.copyfile(self.root / 'later/checkpoints/best.pt', self.source / 'checkpoints/best.pt')
        with self.assertRaisesRegex(ValueError, 'historical best'):
            find_resume_best(path, checkpoint)

    def test_resume_with_no_finite_best_yet(self):
        with PretrainRecord(str(self.source), {}) as record:
            record.save_checkpoint(self.checkpoint(5), {'loss': float('nan')})
        path = self.source / 'checkpoints/last.pt'
        checkpoint = load_resume_checkpoint(path, self.config)
        self.assertIsNone(find_resume_best(path, checkpoint))
        child = self.new_run('child')
        with PretrainRecord(str(child), {}) as record:
            record.resume_from(checkpoint, path, None)
            self.assertFalse((child / 'checkpoints/best.pt').exists())
            self.assertTrue(record.save_checkpoint(self.checkpoint(6), {'loss': 1.0}))

    def test_config_changes_are_limited_to_roots_and_total_target(self):
        path = self.save_parent()
        allowed = copy.deepcopy(self.config)
        allowed['dataset']['root'] = '/new-data'
        allowed['logdir'] = '/new-runs'
        allowed['train']['max_iters'] = 200
        self.assertEqual(load_resume_checkpoint(path, allowed)['iteration'], 10)
        for section, key, value in (('train', 'batch_size', 24),
                                    ('dataset', 'split', 'different.pt'),
                                    ('model', 'name', 'different')):
            changed = copy.deepcopy(self.config)
            changed[section][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'configuration differs'):
                load_resume_checkpoint(path, changed)
        for target in (5, 10):
            changed = copy.deepcopy(self.config)
            changed['train']['max_iters'] = target
            with self.assertRaisesRegex(ValueError, 'total target'):
                load_resume_checkpoint(path, changed)

    def test_incomplete_or_different_selection_checkpoints_are_rejected(self):
        path = self.save_parent()
        original = load_checkpoint(path)
        for field in ('model', 'optimizer', 'scheduler', 'scaler', 'best', 'iteration'):
            incomplete = dict(original)
            incomplete.pop(field)
            torch.save(incomplete, self.root / 'incomplete.pt')
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'complete recorded'):
                load_resume_checkpoint(self.root / 'incomplete.pt', self.config)
        original['selection_rule'] = 'different'
        torch.save(original, self.root / 'incomplete.pt')
        with self.assertRaisesRegex(ValueError, 'selection rule'):
            load_resume_checkpoint(self.root / 'incomplete.pt', self.config)

    def test_nonfinite_model_optimizer_and_amp_state_are_rejected(self):
        original = load_checkpoint(self.save_parent())
        cases = (
            ('model', {'weight': torch.tensor([float('nan')])}),
            ('optimizer', {'state': {0: {'exp_avg': torch.tensor([float('inf')])}}}),
            ('scaler', {'scale': float('nan')}),
            ('scaler', {'scale': 0.0}),
        )
        config = copy.deepcopy(self.config)
        config['train']['use_amp'] = True
        for section, values in cases:
            checkpoint = copy.deepcopy(original)
            checkpoint['config'] = config
            checkpoint['scaler'] = {'scale': 64.0}
            checkpoint[section] = values
            path = self.root / 'bad.pt'
            torch.save(checkpoint, path)
            with self.subTest(section=section, values=values), self.assertRaises(ValueError):
                load_resume_checkpoint(path, config)

    def test_scheduler_infinity_sentinel_is_not_treated_as_corruption(self):
        checkpoint = load_checkpoint(self.save_parent())
        checkpoint['scheduler'] = {'mode_worse': float('inf'), 'best': float('inf')}
        path = self.root / 'sentinel.pt'
        torch.save(checkpoint, path)
        self.assertEqual(load_resume_checkpoint(path, self.config)['iteration'], 10)

    def test_failure_snapshot_cannot_be_used_for_resume(self):
        path = self.root / 'diagnostic.pt'
        torch.save({'diagnostic_only': True, 'iteration': 11}, path)
        with self.assertRaisesRegex(ValueError, 'diagnostic only'):
            load_resume_checkpoint(path, self.config)

    def test_full_state_and_rng_roundtrip_with_cpu_toy_model(self):
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=0)
        scaler = Mock()
        scaler.state_dict.return_value = {'scale': 128.0, 'growth_tracker': 3}
        optimizer.zero_grad()
        model(torch.ones(1, 2)).sum().backward()
        optimizer.step()
        scheduler.step(1.0)
        scheduler.step(2.0)
        with patch('torch.cuda.is_available', return_value=False):
            rng = capture_rng_state()
        checkpoint = dict(self.checkpoint(10), model=model.state_dict(),
                          optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                          scaler=scaler.state_dict(), rng_state=rng)
        # Serialize to avoid state_dict references to live tensors.
        torch.save(checkpoint, self.root / 'toy.pt')
        checkpoint = load_checkpoint(self.root / 'toy.pt')
        expected = (random.random(), np.random.rand(), torch.rand(3))
        restored_model = torch.nn.Linear(2, 1)
        restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=0.9)
        restored_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(restored_optimizer)
        restored_scaler = Mock()
        next_iteration = restore_training_state(checkpoint, restored_model, restored_optimizer,
                                               restored_scheduler, restored_scaler)
        self.assertEqual(next_iteration, 11)
        self.assertEqual(random.random(), expected[0])
        self.assertEqual(np.random.rand(), expected[1])
        self.assertTrue(torch.equal(torch.rand(3), expected[2]))
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, restored_model.state_dict()[key]))
        self.assertEqual(optimizer.param_groups[0]['lr'], restored_optimizer.param_groups[0]['lr'])
        for parameter, state in optimizer.state_dict()['state'].items():
            for key, value in state.items():
                self.assertTrue(torch.equal(value, restored_optimizer.state_dict()['state'][parameter][key]))
        self.assertEqual(scheduler.state_dict(), restored_scheduler.state_dict())
        restored_scaler.load_state_dict.assert_called_once_with(checkpoint['scaler'])
        # With the same next batch, restored Adam moments must produce the same update.
        for current_model, current_optimizer, current_scheduler in (
                (model, optimizer, scheduler),
                (restored_model, restored_optimizer, restored_scheduler)):
            current_optimizer.zero_grad()
            current_model(torch.ones(1, 2)).sum().backward()
            current_optimizer.step()
            current_scheduler.step(3.0)
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, restored_model.state_dict()[key]))
        self.assertEqual(scheduler.state_dict(), restored_scheduler.state_dict())

    def test_legacy_checkpoint_without_rng_still_restores_training_state(self):
        checkpoint = self.checkpoint(5)
        model, optimizer, scheduler, scaler = [Mock() for _ in range(4)]
        self.assertEqual(restore_training_state(checkpoint, model, optimizer, scheduler, scaler), 6)
        for name, target in (('model', model), ('optimizer', optimizer),
                             ('scheduler', scheduler), ('scaler', scaler)):
            target.load_state_dict.assert_called_once_with(checkpoint[name])

    def test_recovery_checkpoint_resumes_before_first_validation(self):
        with PretrainRecord(str(self.source), {}) as record:
            record.state.update(iteration=5, completed_train_calls=5)
            record.save_checkpoint(self.checkpoint(5), keep_periodic=False)
        path = self.source / 'checkpoints/last.pt'
        checkpoint = load_resume_checkpoint(path, self.config)
        self.assertIsNone(checkpoint['best'])
        self.assertIsNone(find_resume_best(path, checkpoint))
        child = self.new_run('child')
        with PretrainRecord(str(child), {}) as record:
            record.resume_from(checkpoint, path, None)
            self.assertEqual(record.state['completed_train_calls'], 5)
            self.assertEqual(load_checkpoint(child / 'checkpoints/last.pt')['iteration'], 5)

    def test_recovery_and_milestone_checkpoints_keep_historical_best(self):
        with PretrainRecord(str(self.source), {}, pinned_iterations=(6,)) as record:
            record.save_checkpoint(self.checkpoint(5), {'loss': 1.0})
            record.state.update(iteration=6, completed_train_calls=6)
            record.save_checkpoint(self.checkpoint(6))
            self.assertEqual(record.state['best']['iteration'], 5)
            record.state.update(iteration=8, completed_train_calls=8)
            record.save_checkpoint(self.checkpoint(8), keep_periodic=False)
        path = self.source / 'checkpoints/last.pt'
        checkpoint = load_resume_checkpoint(path, self.config)
        best_path = find_resume_best(path, checkpoint)
        self.assertEqual(load_checkpoint(best_path)['iteration'], 5)
        self.assertEqual(checkpoint['completed_train_calls'], 8)
        self.assertTrue((self.source / 'checkpoints/6.pt').is_file())
        model, optimizer, scheduler, scaler = [Mock() for _ in range(4)]
        self.assertEqual(restore_training_state(checkpoint, model, optimizer, scheduler, scaler), 9)

    def test_cuda_rng_device_count_mismatch_fails_clearly(self):
        with patch('torch.cuda.is_available', return_value=False):
            checkpoint = dict(self.checkpoint(5), rng_state=capture_rng_state())
        checkpoint['rng_state']['cuda'] = [torch.tensor([0], dtype=torch.uint8)]
        with patch('torch.cuda.device_count', return_value=0):
            with self.assertRaisesRegex(ValueError, 'visible CUDA devices'):
                restore_training_state(checkpoint, Mock(), Mock(), Mock(), Mock())


if __name__ == '__main__':
    unittest.main()
