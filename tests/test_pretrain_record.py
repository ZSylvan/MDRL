import ast
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import yaml

from utils.pretrain_record import MAX_PERIODIC_CHECKPOINTS, PretrainRecord, SELECTION_RULE
from utils.pretrain_checks import AmpSkippedUpdate, NonfiniteTrainingError


ROOT = Path(__file__).resolve().parents[1]


class PretrainRecordTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'checkpoints').mkdir()

    def read_json(self, name):
        return json.loads((self.root / name).read_text(encoding='utf-8'))

    def read_events(self, name):
        return [json.loads(line) for line in
                (self.root / name).read_text(encoding='utf-8').splitlines()]

    def checkpoint(self, iteration):
        return {
            'config': {'seed': 2023}, 'model': {'weight': torch.tensor([iteration])},
            'optimizer': {}, 'scheduler': {}, 'scaler': {}, 'iteration': iteration,
        }

    def test_best_last_periodic_and_nonfinite_losses(self):
        with PretrainRecord(str(self.root), {'git': {'commit': 'test'}}) as record:
            selected = []
            for it, loss in enumerate([4.0, 2.0, 3.0, 2.0, float('nan'), float('inf')], 1):
                record.state.update(iteration=it, completed_train_calls=it)
                selected.append(record.save_checkpoint(
                    self.checkpoint(it), {'loss': loss, 'loss_pos': 0.5}))
        self.assertEqual(selected, [True, True, False, False, False, False])
        best = torch.load(self.root / 'checkpoints/best.pt', map_location='cpu')
        last = torch.load(self.root / 'checkpoints/last.pt', map_location='cpu')
        self.assertEqual(best['iteration'], 2)
        self.assertEqual(last['iteration'], 6)
        self.assertEqual(best['selection_rule'], SELECTION_RULE)
        self.assertTrue({'config', 'model', 'optimizer', 'scheduler', 'scaler'} <= best.keys())
        for it in range(1, 7):
            self.assertTrue((self.root / 'checkpoints' / ('%d.pt' % it)).is_file())
        events = self.read_events('validation.jsonl')
        self.assertIsNone(events[-1]['losses']['loss'])
        self.assertEqual(events[-1]['nonfinite_metrics'], ['loss'])
        self.assertEqual(events[1]['losses']['loss_pos'], 0.5)
        state = self.read_json('run.json')
        self.assertEqual(state['status'], 'completed')
        self.assertEqual(state['best']['iteration'], 2)
        self.assertEqual(state['last_checkpoint']['iteration'], 6)

    def test_skip_count_is_not_a_clean_completion(self):
        with PretrainRecord(str(self.root), {}) as record:
            record.state.update(iteration=1, phase='train')
            try:
                raise RuntimeError('test training failure')
            except RuntimeError as exc:
                record.record_skip(exc)
        state = self.read_json('run.json')
        self.assertEqual(state['status'], 'completed_with_skips')
        self.assertEqual(state['skipped_iterations'], 1)
        event = self.read_events('failures.jsonl')[0]
        self.assertEqual(event['outcome'], 'skipped')
        self.assertIn('test training failure', event['traceback'])

    def test_retains_latest_50_periodic_plus_best_and_last(self):
        with PretrainRecord(str(self.root), {}) as record:
            for it in range(1, MAX_PERIODIC_CHECKPOINTS + 6):
                record.state.update(iteration=it, completed_train_calls=it)
                record.save_checkpoint(self.checkpoint(it), {'loss': float(it)})
                self.assertLessEqual(len(list((self.root / 'checkpoints').glob('*.pt'))), 52)
        periodic = {int(path.stem) for path in (self.root / 'checkpoints').glob('*.pt')
                    if path.stem.isdecimal()}
        self.assertEqual(periodic, set(range(6, 56)))
        best = torch.load(self.root / 'checkpoints/best.pt', map_location='cpu')
        last = torch.load(self.root / 'checkpoints/last.pt', map_location='cpu')
        self.assertEqual(best['iteration'], 1)
        self.assertEqual(last['iteration'], 55)
        state = self.read_json('run.json')
        self.assertEqual(state['best']['checkpoint'], 'checkpoints/best.pt')
        self.assertEqual(state['best']['source_checkpoint'], 'checkpoints/1.pt')
        self.assertEqual(state['periodic_checkpoint_limit'], 50)
        self.assertEqual(len(state['periodic_checkpoints']), 50)
        self.assertTrue((self.root / state['best']['checkpoint']).is_file())
        events = self.read_events('validation.jsonl')
        self.assertEqual(len(events), 55)
        self.assertEqual(events[50]['pruned_checkpoints'], ['checkpoints/1.pt'])
        self.assertFalse(list((self.root / 'checkpoints').glob('*.tmp')))

    def test_failed_staging_does_not_prune_previous_checkpoints(self):
        with self.assertRaises(OSError):
            with PretrainRecord(str(self.root), {}) as record:
                for it in range(1, 51):
                    record.save_checkpoint(self.checkpoint(it), {'loss': float(it)})
                with patch('utils.pretrain_record.torch.save', side_effect=OSError('write failed')):
                    record.save_checkpoint(self.checkpoint(51), {'loss': 0.5})
        self.assertEqual(len(list((self.root / 'checkpoints').glob('*.pt'))), 52)
        self.assertTrue((self.root / 'checkpoints/1.pt').is_file())
        state = self.read_json('run.json')
        self.assertEqual(state['status'], 'failed')
        self.assertEqual(state['last_checkpoint']['iteration'], 50)
        self.assertEqual(state['best']['iteration'], 1)

    def test_alias_failure_at_capacity_keeps_file_count_bounded(self):
        with self.assertRaises(OSError):
            with PretrainRecord(str(self.root), {}) as record:
                for it in range(1, 51):
                    record.save_checkpoint(self.checkpoint(it), {'loss': float(it)})
                with patch('utils.pretrain_record.shutil.copyfile', side_effect=OSError('copy failed')):
                    record.save_checkpoint(self.checkpoint(51), {'loss': 0.5})
        self.assertEqual(len(list((self.root / 'checkpoints').glob('*.pt'))), 52)
        self.assertFalse((self.root / 'checkpoints/1.pt').exists())
        self.assertTrue((self.root / 'checkpoints/51.pt').is_file())
        best = torch.load(self.root / 'checkpoints/best.pt', map_location='cpu')
        self.assertEqual(best['iteration'], 1)
        self.assertEqual(self.read_json('run.json')['status'], 'failed')

    def test_setup_failure_and_interrupt_are_not_swallowed(self):
        for error, status, fatal_count in (
                (ValueError('bad setup'), 'failed', 1),
                (KeyboardInterrupt(), 'interrupted', 0)):
            with self.subTest(status=status):
                folder = self.root / status
                folder.mkdir()
                with self.assertRaises(type(error)):
                    with PretrainRecord(str(folder), {}) as record:
                        raise error
                state = json.loads((folder / 'run.json').read_text(encoding='utf-8'))
                self.assertEqual(state['status'], status)
                self.assertEqual(state['fatal_errors'], fatal_count)
                self.assertEqual(state['phase'], 'setup')
                self.assertIsNone(state['last_checkpoint'])

    def test_incomplete_alias_copy_keeps_previous_checkpoint(self):
        def fail_copy(source, destination):
            Path(destination).write_bytes(b'incomplete')
            raise OSError('simulated full disk')

        with self.assertRaises(OSError):
            with PretrainRecord(str(self.root), {}) as record:
                record.state.update(iteration=1, phase='checkpoint')
                record.save_checkpoint(self.checkpoint(1), {'loss': 4.0})
                record.state['iteration'] = 2
                with patch('utils.pretrain_record.shutil.copyfile', side_effect=fail_copy):
                    record.save_checkpoint(self.checkpoint(2), {'loss': 2.0})
        for name in ('best.pt', 'last.pt'):
            saved = torch.load(self.root / 'checkpoints' / name, map_location='cpu')
            self.assertEqual(saved['iteration'], 1)
        self.assertTrue((self.root / 'checkpoints/2.pt').is_file())
        self.assertEqual(self.read_json('run.json')['status'], 'failed')

    def test_best_copy_failure_records_updated_last_but_previous_best(self):
        import shutil
        copyfile = shutil.copyfile

        def fail_best_copy(source, destination):
            if str(destination).endswith('best.pt.tmp'):
                raise OSError('best copy failed')
            return copyfile(source, destination)

        with self.assertRaises(OSError):
            with PretrainRecord(str(self.root), {}) as record:
                record.state.update(iteration=1, phase='checkpoint')
                record.save_checkpoint(self.checkpoint(1), {'loss': 4.0})
                record.state['iteration'] = 2
                with patch('utils.pretrain_record.shutil.copyfile', side_effect=fail_best_copy):
                    record.save_checkpoint(self.checkpoint(2), {'loss': 2.0})
        state = self.read_json('run.json')
        self.assertEqual(state['status'], 'failed')
        self.assertEqual(state['last_checkpoint']['iteration'], 2)
        self.assertEqual(state['best']['iteration'], 1)
        best = torch.load(self.root / 'checkpoints/best.pt', map_location='cpu')
        last = torch.load(self.root / 'checkpoints/last.pt', map_location='cpu')
        self.assertEqual(best['iteration'], 1)
        self.assertEqual(last['iteration'], 2)

    def test_recovery_only_updates_last_without_validation_or_new_periodic_file(self):
        with PretrainRecord(str(self.root), {}) as record:
            record.save_checkpoint(self.checkpoint(1), {'loss': 1.0})
            before_best = (self.root / 'checkpoints/best.pt').read_bytes()
            self.assertFalse(record.save_checkpoint(self.checkpoint(2), keep_periodic=False))
            self.assertEqual((self.root / 'checkpoints/best.pt').read_bytes(), before_best)
            last = torch.load(self.root / 'checkpoints/last.pt', map_location='cpu')
            self.assertEqual(last['iteration'], 2)
            self.assertIsNone(last['validation_losses'])
            self.assertEqual(last['best']['iteration'], 1)
            self.assertEqual(last['checkpoint_kind'], 'recovery')
            self.assertEqual(record.state['last_checkpoint']['checkpoint'], 'checkpoints/last.pt')
            self.assertFalse((self.root / 'checkpoints/2.pt').exists())
            self.assertEqual(len(self.read_events('validation.jsonl')), 1)
            self.assertEqual([event['kind'] for event in self.read_events('checkpoints.jsonl')],
                             ['validation', 'recovery'])

    def test_recovery_write_failure_keeps_previous_last(self):
        with self.assertRaises(OSError):
            with PretrainRecord(str(self.root), {}) as record:
                record.save_checkpoint(self.checkpoint(1), {'loss': 1.0})
                with patch('utils.pretrain_record.torch.save', side_effect=OSError('full disk')):
                    record.save_checkpoint(self.checkpoint(2), keep_periodic=False)
        last = torch.load(self.root / 'checkpoints/last.pt', map_location='cpu')
        self.assertEqual(last['iteration'], 1)

    def test_fifty_total_files_protect_pinned_milestones_when_rotating(self):
        with PretrainRecord(str(self.root), {}, periodic_checkpoint_limit=48,
                            pinned_iterations=(1, 2)) as record:
            for it in range(1, 56):
                record.save_checkpoint(self.checkpoint(it), {'loss': float(it)})
                self.assertLessEqual(len(list((self.root / 'checkpoints').glob('*.pt'))), 50)
        self.assertTrue((self.root / 'checkpoints/1.pt').is_file())
        self.assertTrue((self.root / 'checkpoints/2.pt').is_file())
        self.assertFalse((self.root / 'checkpoints/3.pt').exists())
        self.assertEqual(len(list((self.root / 'checkpoints').glob('*.pt'))), 50)
        self.assertEqual(self.read_json('run.json')['pinned_iterations'], [1, 2])

    def run_simulated_loop(self, train, validate, start_iter=1, max_iters=3,
                           val_freq=2, checkpoint_freq=0, checkpoint_milestones=()):
        # Execute the real loop with fake train/validate calls, not the MDRL model.
        tree = ast.parse((ROOT / 'scripts/train_drug3d.py').read_text(encoding='utf-8'))
        function = next(node for node in tree.body
                        if isinstance(node, ast.FunctionDef) and node.name == 'run_training')
        loop = next(node for node in function.body if isinstance(node, ast.For))
        code = compile(ast.Module(body=[loop], type_ignores=[]), '<pretraining loop>', 'exec')
        model = Mock()
        model.state_dict.return_value = {}
        model.named_parameters.return_value = []
        optimizer = Mock()
        optimizer.state_dict.return_value = {}
        with PretrainRecord(str(self.root), {}, pinned_iterations=checkpoint_milestones) as record:
            exec(code, {
                'config': SimpleNamespace(train=SimpleNamespace(max_iters=max_iters, val_freq=val_freq)),
                'model': model, 'optimizer': optimizer, 'scheduler': optimizer,
                'scaler': optimizer, 'record': record, 'train': train,
                'validate': validate, 'logger': Mock(),
                'start_iter': start_iter, 'capture_rng_state': lambda: {},
                'snapshot': self.checkpoint, 'checkpoint_freq': checkpoint_freq,
                'checkpoint_milestones': set(checkpoint_milestones),
                'NonfiniteTrainingError': NonfiniteTrainingError,
                'AmpSkippedUpdate': AmpSkippedUpdate,
            })

    def test_sparse_validation_recovery_and_milestones_have_independent_schedules(self):
        validate = Mock(side_effect=[{'loss': 1.0}, {'loss': 2.0}])
        self.run_simulated_loop(Mock(), validate, max_iters=9, val_freq=6,
                                checkpoint_freq=2, checkpoint_milestones=(5, 9))
        self.assertEqual([call.args[0] for call in validate.call_args_list], [6, 9])
        self.assertEqual([event['iteration'] for event in self.read_events('checkpoints.jsonl')],
                         [2, 4, 5, 6, 8, 9])
        self.assertEqual([event['iteration'] for event in self.read_events('validation.jsonl')], [6, 9])
        self.assertEqual({path.stem for path in (self.root / 'checkpoints').glob('*.pt')},
                         {'5', '6', '9', 'best', 'last'})
        self.assertEqual(self.read_json('run.json')['best']['iteration'], 6)

    def test_resumed_sparse_loop_keeps_global_iteration_cadence(self):
        validate = Mock(return_value={'loss': 1.0})
        self.run_simulated_loop(Mock(), validate, start_iter=7, max_iters=13, val_freq=6,
                                checkpoint_freq=2, checkpoint_milestones=(9, 13))
        self.assertEqual([call.args[0] for call in validate.call_args_list], [12, 13])
        self.assertEqual([event['iteration'] for event in self.read_events('checkpoints.jsonl')],
                         [8, 9, 10, 12, 13])

    def test_resumed_loop_starts_at_next_iteration_and_keeps_total_target(self):
        train = Mock()
        validate = Mock(return_value={'loss': 1.0})
        self.run_simulated_loop(train, validate, start_iter=4, max_iters=7)
        self.assertEqual([call.args[0] for call in train.call_args_list], [4, 5, 6, 7])
        self.assertEqual([call.args[0] for call in validate.call_args_list], [4, 6, 7])

    def test_loop_preserves_skip_and_validation_schedule(self):
        train = Mock(side_effect=[RuntimeError('skip iteration 1'), None, None])
        validate = Mock(side_effect=[{'loss': 4.0}, {'loss': 2.0}])
        self.run_simulated_loop(train, validate)
        self.assertEqual([call.args[0] for call in validate.call_args_list], [2, 3])
        state = self.read_json('run.json')
        self.assertEqual(state['status'], 'completed_with_skips')
        self.assertEqual(state['completed_train_calls'], 2)
        self.assertEqual(state['skipped_iterations'], 1)
        self.assertEqual(state['best']['iteration'], 3)

    def test_loop_validation_failure_is_fatal(self):
        with self.assertRaises(ValueError):
            self.run_simulated_loop(Mock(), Mock(side_effect=ValueError('validation failed')))
        state = self.read_json('run.json')
        self.assertEqual(state['status'], 'failed')
        self.assertEqual(state['phase'], 'validation')
        self.assertEqual(state['iteration'], 2)
        self.assertIsNone(state['last_checkpoint'])

    def test_loop_interrupt_is_not_a_skip(self):
        with self.assertRaises(KeyboardInterrupt):
            self.run_simulated_loop(Mock(side_effect=KeyboardInterrupt), Mock())
        state = self.read_json('run.json')
        self.assertEqual(state['status'], 'interrupted')
        self.assertEqual(state['skipped_iterations'], 0)

    def test_loop_records_nonfinite_loss_and_stops_instead_of_skipping(self):
        batch = SimpleNamespace(to_dict=lambda: {'mol_id': torch.tensor([7])})
        error = NonfiniteTrainingError('train_loss', ['loss'],
                                      {'loss': torch.tensor(float('nan'))}, {'batch': batch})
        with self.assertRaises(NonfiniteTrainingError):
            self.run_simulated_loop(Mock(side_effect=error), Mock())
        state = self.read_json('run.json')
        self.assertEqual(state['status'], 'failed')
        self.assertEqual(state['skipped_iterations'], 0)
        self.assertEqual(state['numerical_failure']['iteration'], 1)
        self.assertTrue((self.root / 'diagnostics/first_failure.pt').is_file())
        self.assertIsNone(state['last_checkpoint'])

    def test_loop_continues_after_confirmed_amp_skip_and_keeps_validation_schedule(self):
        batch = SimpleNamespace(to_dict=lambda: {'mol_id': torch.tensor([7])})
        error = NonfiniteTrainingError('train_gradient_norm', ['gradient_norm'],
                                      {'gradient_norm': torch.tensor(float('nan'))},
                                      {'batch': batch, 'losses': {'loss': torch.tensor(1.0)}})
        event = AmpSkippedUpdate(error, 2048.0, 1024.0)
        train = Mock(side_effect=[event, None, None])
        validate = Mock(return_value={'loss': 1.0})
        self.run_simulated_loop(train, validate)
        state = self.read_json('run.json')
        self.assertEqual(state['status'], 'completed')
        self.assertEqual(state['completed_train_calls'], 3)
        self.assertEqual(state['skipped_iterations'], 0)
        self.assertEqual(state['amp_skipped_updates'], 1)
        self.assertEqual([call.args[0] for call in train.call_args_list], [1, 2, 3])
        self.assertEqual([call.args[0] for call in validate.call_args_list], [2, 3])
        self.assertEqual(self.read_events('failures.jsonl'), [])

    def test_amp_snapshot_write_error_is_fatal_not_an_original_runtime_skip(self):
        batch = SimpleNamespace(to_dict=lambda: {'mol_id': torch.tensor([7])})
        error = NonfiniteTrainingError('train_gradient_norm', ['gradient_norm'], {},
                                      {'batch': batch, 'losses': {'loss': torch.tensor(1.0)}})
        with patch('utils.pretrain_record.torch.save', side_effect=RuntimeError('disk write failed')):
            with self.assertRaisesRegex(RuntimeError, 'disk write failed'):
                self.run_simulated_loop(Mock(side_effect=AmpSkippedUpdate(error, 2.0, 1.0)), Mock())
        state = self.read_json('run.json')
        self.assertEqual(state['status'], 'failed')
        self.assertEqual(state['skipped_iterations'], 0)

    def test_failure_snapshot_write_error_is_fatal_not_skipped(self):
        batch = SimpleNamespace(to_dict=lambda: {'mol_id': torch.tensor([7])})
        error = NonfiniteTrainingError('train_loss', ['loss'],
                                      {'loss': torch.tensor(float('nan'))}, {'batch': batch})
        with patch('utils.pretrain_record.torch.save', side_effect=RuntimeError('disk write failed')):
            with self.assertRaisesRegex(RuntimeError, 'disk write failed'):
                self.run_simulated_loop(Mock(side_effect=error), Mock())
        state = self.read_json('run.json')
        self.assertEqual(state['status'], 'failed')
        self.assertEqual(state['skipped_iterations'], 0)
        self.assertFalse(state['numerical_failure']['snapshot_saved'])


class PretrainCompatibilityTests(unittest.TestCase):
    def test_original_settings_are_unchanged(self):
        def read_config(name):
            with (ROOT / 'configs/train' / name).open(encoding='utf-8') as handle:
                return yaml.safe_load(handle)

        original = read_config('train_MDRL.yml')
        reproduction = read_config('train_MDRL_reproduction.yml')
        output_root = reproduction.pop('logdir')
        dataset_root = reproduction['dataset'].pop('root')
        original['dataset'].pop('root')
        self.assertEqual(original['train'].pop('batch_size'), 24)
        self.assertEqual(reproduction['train'].pop('batch_size'), 8)
        self.assertEqual(original, reproduction)
        self.assertEqual(dataset_root,
                         '/home/zhuenqiang_1/Student_Project/zms/raw_data/MDRL/geom_drug')
        self.assertTrue(output_root.endswith('/experiment_data/mdrl/pretrain'))

    def test_tensorboard_import_order_and_no_torchsummary(self):
        tree = ast.parse((ROOT / 'scripts/train_drug3d.py').read_text(encoding='utf-8'))
        imports = [name.name for node in tree.body if isinstance(node, ast.Import)
                   for name in node.names]
        self.assertLess(imports.index('distutils.version'), imports.index('torch.utils.tensorboard'))
        self.assertFalse(any(isinstance(node, ast.ImportFrom) and node.module == 'torchsummary'
                             for node in ast.walk(tree)))
        for source in ('scripts/train_drug3d.py', 'utils/pretrain_record.py',
                       'utils/pretrain_resume.py'):
            ast.parse((ROOT / source).read_text(encoding='utf-8'), feature_version=(3, 8))


if __name__ == '__main__':
    unittest.main()
