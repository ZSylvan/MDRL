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

from utils.pretrain_checks import NonfiniteTrainingError, require_finite
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


if __name__ == '__main__':
    unittest.main()
