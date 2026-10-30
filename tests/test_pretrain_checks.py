import ast
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import torch
from torch.nn.utils import clip_grad_norm_

from utils import pretrain_checks as checks
from utils.pretrain_checks import AmpSkippedUpdate, NonfiniteTrainingError, require_finite
from utils.pretrain_record import PretrainRecord
from utils.pretrain_resume import capture_rng_state, load_checkpoint


ROOT = Path(__file__).resolve().parents[1]


class PretrainChecksTests(unittest.TestCase):
    def training_function(self, name, bad_loss=False, bad_gradient=False):
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        if bad_gradient:
            parameter.register_hook(lambda grad: torch.full_like(grad, float('nan')))
        loss = parameter.sum() * (float('nan') if bad_loss else 1.0)
        model = Mock()
        model.get_loss.return_value = {'loss': loss, 'loss_pos': loss}
        model.parameters.return_value = [parameter]
        batch = SimpleNamespace(
            node_type=torch.tensor([0]), node_pos=torch.zeros(1, 3),
            node_type_batch=torch.tensor([0]), halfedge_type=torch.tensor([], dtype=torch.long),
            halfedge_index=torch.empty(2, 0, dtype=torch.long),
            halfedge_type_batch=torch.tensor([], dtype=torch.long), num_graphs=1)
        batch.to = lambda device: batch
        batch.to_dict = lambda: {'mol_id': torch.tensor([42]), 'node_pos': batch.node_pos}
        scaler = Mock()
        scaler.is_enabled.return_value = False
        scaler.scale.side_effect = lambda value: value
        optimizer = Mock()
        optimizer.param_groups = [{'lr': 1e-4}]
        scheduler = Mock()
        namespace = {
            'torch': torch, 'model': model, 'optimizer': optimizer, 'scaler': scaler,
            'scheduler': scheduler, 'train_iterator': iter([batch]), 'val_loader': [batch],
            'tqdm': lambda values, **kwargs: values, 'clip_grad_norm_': clip_grad_norm_,
            'require_finite': require_finite, 'capture_rng_state': capture_rng_state,
            'logger': Mock(), 'writer': Mock(), 'args': SimpleNamespace(device='cpu'),
            'config': SimpleNamespace(train=SimpleNamespace(
                pos_noise_std=0.05, use_amp=False, max_grad_norm=50,
                scheduler=SimpleNamespace(type='plateau'))),
        }
        namespace.update(AmpSkippedUpdate=AmpSkippedUpdate,
                         NonfiniteTrainingError=NonfiniteTrainingError,
                         handle_amp_gradient=checks.handle_amp_gradient,
                         require_amp_scale=checks.require_amp_scale)
        tree = ast.parse((ROOT / 'scripts/train_drug3d.py').read_text(encoding='utf-8'))
        outer = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name == 'run_training')
        function = next(node for node in outer.body if isinstance(node, ast.FunctionDef)
                        and node.name == name)
        exec(compile(ast.Module(body=[function], type_ignores=[]), '<real training function>', 'exec'), namespace)
        return namespace[name], namespace, parameter

    def test_real_train_stops_before_backward_on_nonfinite_loss(self):
        train, env, parameter = self.training_function('train', bad_loss=True)
        with self.assertRaises(NonfiniteTrainingError) as caught:
            train(11)
        self.assertEqual(caught.exception.stage, 'train_loss')
        self.assertIsNone(parameter.grad)
        env['scaler'].step.assert_not_called()
        self.assertIn('rng_before_loss', caught.exception.context)

    def test_real_train_stops_before_update_on_nonfinite_gradient_norm(self):
        train, env, parameter = self.training_function('train', bad_gradient=True)
        with self.assertRaises(NonfiniteTrainingError) as caught:
            train(11)
        self.assertEqual(caught.exception.stage, 'train_gradient_norm')
        env['scaler'].step.assert_not_called()
        env['scaler'].update.assert_not_called()
        self.assertEqual(parameter.item(), 1.0)

    def test_real_train_allows_only_a_confirmed_amp_skip(self):
        train, env, parameter = self.training_function('train', bad_gradient=True)
        scaler = env['scaler']
        scaler.is_enabled.return_value = True
        scaler.get_scale.side_effect = [2048.0, 1024.0]
        with self.assertRaises(AmpSkippedUpdate) as caught:
            train(11)
        self.assertEqual(caught.exception.scale_before, 2048.0)
        self.assertEqual(caught.exception.scale_after, 1024.0)
        env['optimizer'].step.assert_not_called()
        scaler.step.assert_called_once_with(env['optimizer'])
        scaler.update.assert_called_once_with()
        self.assertEqual(parameter.item(), 1.0)

    def test_bad_gradient_cannot_update_when_amp_does_not_skip(self):
        train, env, _ = self.training_function('train', bad_gradient=True)
        original_step = env['optimizer'].step
        env['scaler'].is_enabled.return_value = True
        env['scaler'].get_scale.return_value = 2048.0
        env['scaler'].step.side_effect = lambda optimizer: optimizer.step()
        with self.assertRaises(NonfiniteTrainingError):
            train(11)
        original_step.assert_not_called()
        self.assertIs(env['optimizer'].step, original_step)
        env['scaler'].update.assert_not_called()

    def test_amp_scale_underflow_is_fatal_even_after_a_skipped_update(self):
        train, env, _ = self.training_function('train', bad_gradient=True)
        env['scaler'].is_enabled.return_value = True
        env['scaler'].get_scale.side_effect = [1e-45, 0.0]
        with self.assertRaises(NonfiniteTrainingError) as caught:
            train(11)
        self.assertEqual(caught.exception.stage, 'amp_scale')

    def test_amp_must_reduce_scale_before_training_can_continue(self):
        train, env, _ = self.training_function('train', bad_gradient=True)
        env['scaler'].is_enabled.return_value = True
        env['scaler'].get_scale.return_value = 2048.0
        with self.assertRaises(NonfiniteTrainingError):
            train(11)

    def test_amp_handling_runtime_error_stops_instead_of_becoming_a_skip(self):
        train, env, _ = self.training_function('train', bad_gradient=True)
        original_step = env['optimizer'].step
        env['scaler'].is_enabled.return_value = True
        env['scaler'].get_scale.return_value = 2048.0
        env['scaler'].step.side_effect = RuntimeError('no inf checks recorded')
        with self.assertRaises(NonfiniteTrainingError) as caught:
            train(11)
        self.assertIn('no inf checks recorded', caught.exception.context['amp_error'])
        self.assertIs(env['optimizer'].step, original_step)

    @unittest.skipUnless(hasattr(getattr(torch, 'amp', None), 'GradScaler'),
                         'Local CPU GradScaler is unavailable in this PyTorch version')
    def test_real_cpu_amp_skips_nan_gradient_then_performs_next_finite_update(self):
        train, env, parameter = self.training_function('train')
        optimizer = torch.optim.Adam([parameter], lr=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        scaler = torch.amp.GradScaler('cpu', init_scale=2048.0)
        env.update(optimizer=optimizer, scaler=scaler)
        batch = next(env['train_iterator'])
        env['train_iterator'] = iter([batch, batch])
        hook = parameter.register_hook(lambda grad: torch.full_like(grad, float('nan')))
        with self.assertRaises(AmpSkippedUpdate):
            train(11)
        self.assertEqual(parameter.item(), 1.0)
        self.assertEqual(optimizer.state_dict()['state'], {})
        self.assertEqual(scaler.get_scale(), 1024.0)
        hook.remove()
        loss = parameter.sum()
        env['model'].get_loss.return_value = {'loss': loss, 'loss_pos': loss}
        train(12)
        scheduler.step()
        self.assertLess(parameter.item(), 1.0)
        self.assertTrue(torch.isfinite(parameter).all().item())

    @unittest.skipUnless(hasattr(getattr(torch, 'amp', None), 'GradScaler'),
                         'Local CPU GradScaler is unavailable in this PyTorch version')
    def test_norm_overflow_not_detected_by_amp_is_blocked_before_adam(self):
        train, env, parameter = self.training_function('train')
        optimizer = torch.optim.Adam([parameter], lr=1e-4)
        env.update(optimizer=optimizer, scaler=torch.amp.GradScaler('cpu', init_scale=2048.0))
        env['clip_grad_norm_'] = lambda *args: torch.tensor(float('inf'))
        with self.assertRaises(NonfiniteTrainingError):
            train(11)
        self.assertEqual(parameter.item(), 1.0)
        self.assertEqual(optimizer.state_dict()['state'], {})

    def test_real_validation_stops_before_scheduler_on_nonfinite_loss(self):
        validate, env, _ = self.training_function('validate', bad_loss=True)
        with self.assertRaises(NonfiniteTrainingError) as caught:
            validate(12)
        self.assertEqual(caught.exception.stage, 'validation_loss')
        self.assertEqual(caught.exception.context['validation_batch_index'], 1)
        env['scheduler'].step.assert_not_called()

    def test_finite_train_preserves_original_gradient_and_step_calls(self):
        train, env, parameter = self.training_function('train')
        env['optimizer'].param_groups = [{'lr': 1e-4}]
        train(11)
        env['scaler'].step.assert_called_once_with(env['optimizer'])
        env['scaler'].update.assert_called_once_with()
        self.assertEqual(parameter.grad.item(), 1.0)

    def test_nonfinite_is_not_swallowed_by_runtime_error_skip_handler(self):
        self.assertFalse(issubclass(NonfiniteTrainingError, RuntimeError))
        with self.assertRaises(NonfiniteTrainingError):
            require_finite('loss', {'loss': torch.tensor(float('inf'))}, {})

    def test_first_failure_preserves_aliases_and_records_exact_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'checkpoints').mkdir()
            with self.assertRaises(NonfiniteTrainingError):
                with PretrainRecord(str(root), {}) as record:
                    checkpoint = {
                        'iteration': 10, 'model': {'weight': torch.ones(1)},
                        'optimizer': {}, 'scheduler': {'mode_worse': float('inf')},
                        'scaler': {}, 'config': {'train': {'use_amp': False}},
                    }
                    record.save_checkpoint(checkpoint, {'loss': 1.0})
                    aliases = {name: (root / 'checkpoints' / name).read_bytes()
                               for name in ('best.pt', 'last.pt')}
                    record.state.update(iteration=11, phase='train')
                    train, _, _ = self.training_function('train', bad_loss=True)
                    try:
                        train(11)
                    except NonfiniteTrainingError as exc:
                        saved_context = copy.deepcopy(exc.context['rng_before_loss'])
                        record.record_numerical_failure(exc, dict(checkpoint, iteration=11), {})
                        before = (root / 'diagnostics/first_failure.pt').read_bytes()
                        with self.assertRaises(FileExistsError):
                            record.record_numerical_failure(exc, checkpoint, {})
                        self.assertEqual(before, (root / 'diagnostics/first_failure.pt').read_bytes())
                        raise
            for name, expected in aliases.items():
                self.assertEqual((root / 'checkpoints' / name).read_bytes(), expected)
            failure = load_checkpoint(root / 'diagnostics/first_failure.pt')
            self.assertTrue(failure['diagnostic_only'])
            self.assertEqual(failure['batch']['mol_id'].tolist(), [42])
            self.assertTrue(torch.equal(failure['context']['rng_before_loss']['torch'],
                                        saved_context['torch']))
            self.assertEqual(failure['batch']['node_pos'].device.type, 'cpu')
            state = json.loads((root / 'run.json').read_text())
            self.assertEqual(state['status'], 'failed')
            self.assertEqual(state['skipped_iterations'], 0)
            self.assertEqual(state['numerical_failure']['stage'], 'train_loss')
            self.assertTrue((root / 'diagnostics/first_failure.json').is_file())

    def test_amp_skip_records_first_snapshot_only_and_keeps_fatal_failure_separate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'checkpoints').mkdir()
            with PretrainRecord(str(root), {}) as record:
                checkpoint = {'iteration': 10, 'model': {}, 'optimizer': {},
                              'scheduler': {}, 'scaler': {}}
                record.save_checkpoint(checkpoint, {'loss': 1.0})
                aliases = {name: (root / 'checkpoints' / name).read_bytes()
                           for name in ('best.pt', 'last.pt')}
                train, env, _ = self.training_function('train', bad_gradient=True)
                env['scaler'].is_enabled.return_value = True
                env['scaler'].get_scale.side_effect = [2048.0, 1024.0]
                with self.assertRaises(AmpSkippedUpdate) as caught:
                    train(11)
                record.state.update(iteration=11, phase='train')
                record.record_amp_skip(caught.exception, checkpoint, {})
                first = (root / 'diagnostics/first_amp_skip.pt').read_bytes()
                record.state['iteration'] = 12
                record.record_amp_skip(caught.exception, checkpoint, {})
                self.assertEqual(first, (root / 'diagnostics/first_amp_skip.pt').read_bytes())
                self.assertEqual(record.state['amp_skipped_updates'], 2)
                self.assertEqual(record.state['skipped_iterations'], 0)
                self.assertNotIn('numerical_failure', record.state)
                self.assertEqual((root / 'failures.jsonl').read_text(), '')
                for name, expected in aliases.items():
                    self.assertEqual((root / 'checkpoints' / name).read_bytes(), expected)
                record.record_numerical_failure(caught.exception.error, checkpoint, {})
                self.assertTrue((root / 'diagnostics/first_failure.pt').is_file())
            events = [json.loads(line) for line in (root / 'amp_events.jsonl').read_text().splitlines()]
            self.assertEqual([event['iteration'] for event in events], [11, 12])
            self.assertTrue(events[0]['optimizer_step_skipped'])
            self.assertEqual(events[0]['scale_before'], 2048.0)
            self.assertEqual(events[0]['scale_after'], 1024.0)
            self.assertIsNotNone(events[0]['snapshot'])
            self.assertIsNone(events[1]['snapshot'])
            self.assertTrue(load_checkpoint(root / 'diagnostics/first_amp_skip.pt')['diagnostic_only'])


if __name__ == '__main__':
    unittest.main()
