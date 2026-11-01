"""Exercise the actual RL train-step body with a CPU tensor and a fake model."""

import ast
import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
import yaml
from torch.nn.utils import clip_grad_norm_

from utils import pretrain_checks as checks


ROOT = Path(__file__).resolve().parents[1]


class RLTrainingChecksTests(unittest.TestCase):
    def test_entry_sequences_rounds_with_only_fake_sampling_training_and_scoring(self):
        spec = importlib.util.spec_from_file_location('rl_entry_integration', ROOT / 'RL_utils/01_RL.py')
        entry = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(entry)
        from RL_utils.task_config import load_task_config
        source = ROOT / 'RL_utils/config_RL/tasks/chk1_auroraa.yml'
        config = load_task_config(source, check_files=False)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            config.update(output_dir=str(directory / 'output'), device='cpu')
            config['rl'].update(n_epochs=3, initial_num_mols=3, new_num_mols_per_epoch=2, top_k=4)
            training = yaml.safe_load(Path(config['train_config']).read_text())
            checkpoint = directory / 'last.pt'
            torch.save(dict(iteration=985000, config=training, model=dict(weight=torch.tensor([1.0]))), checkpoint)
            config['initial_checkpoint'] = str(checkpoint)
            sampling = Mock(side_effect=lambda path, count, epoch, **kw:
                            [dict(smiles='%d-%d' % (epoch, i), rdmol=None) for i in range(count)])
            trainer = Mock(side_effect=lambda root, epoch, path, **kw: str(directory / ('round%d.pt' % epoch)))
            pool = Mock()

            def score(population, pool, workers, path):
                for molecule in population:
                    molecule['score'] = 0.5
                return population

            def save(population, path):
                path.mkdir()

            modules = {
                'RL_utils.sample_drug3d_forRL': SimpleNamespace(sample=sampling),
                'RL_utils.train_drug3d_forRL': SimpleNamespace(train_forRL=trainer),
                'RL_utils.scoring_workers': SimpleNamespace(create_pool=lambda *args: pool, score_population=score),
            }
            with patch.dict('sys.modules', modules), patch.object(entry, 'save_population', side_effect=save):
                entry.run(config, source)
            self.assertEqual(sampling.call_count, 3)
            self.assertEqual(trainer.call_count, 2)
            pinned = directory / 'output/pretrained_985000.pt'
            self.assertTrue(pinned.is_file())
            self.assertEqual(sampling.call_args_list[0].args[0], str(pinned))
            self.assertEqual(trainer.call_args_list[0].args[2], str(pinned))
            self.assertEqual(sampling.call_args_list[1].args[0], str(directory / 'round1.pt'))
            self.assertEqual(trainer.call_args_list[1].args[2], str(directory / 'round1.pt'))
            rounds = [json.loads(line) for line in (directory / 'output/rounds.jsonl').read_text().splitlines()]
            self.assertEqual([item['selected'] for item in rounds], [3, 4, 4])
            pool.shutdown.assert_called_once_with(wait=True)
            with self.assertRaises(FileExistsError):
                with patch.dict('sys.modules', modules):
                    entry.run(config, source)

    def fixture(self, bad_loss=False, bad_gradient=False, amp_skip=False):
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        if bad_gradient:
            parameter.register_hook(lambda value: torch.full_like(value, float('nan')))
        loss = parameter.sum() * (float('nan') if bad_loss else 1.0)
        model = Mock()
        model.get_loss.return_value = dict(loss=loss, loss_pos=loss)
        model.parameters.return_value = [parameter]
        batch = SimpleNamespace(
            node_type=torch.tensor([0]), node_pos=torch.zeros(1, 3),
            node_type_batch=torch.tensor([0]), halfedge_type=torch.empty(0, dtype=torch.long),
            halfedge_index=torch.empty((2, 0), dtype=torch.long),
            halfedge_type_batch=torch.empty(0, dtype=torch.long), num_graphs=1)
        batch.to = lambda device: batch
        batch.to_dict = lambda: dict(mol_id=torch.tensor([42]), node_pos=batch.node_pos)
        scaler = Mock()
        scaler.is_enabled.return_value = False
        scaler.scale.side_effect = lambda value: value
        optimizer = Mock()
        optimizer.param_groups = [dict(lr=1e-6)]
        namespace = dict(
            torch=torch, os=os, model=model, scaler=scaler, optimizer=optimizer,
            train_iterator=iter([batch]), args=SimpleNamespace(device='cpu'),
            config=SimpleNamespace(train=SimpleNamespace(pos_noise_std=0.05, use_amp=False, max_grad_norm=50)),
            current={}, capture_training_rng=lambda device: {}, require_amp_scale=checks.require_amp_scale,
            require_finite=checks.require_finite, NonfiniteTrainingError=checks.NonfiniteTrainingError,
            clip_grad_norm_=clip_grad_norm_, handle_amp_gradient=checks.handle_amp_gradient,
            append_record=Mock(), log_dir='unused', logger=Mock(), writer=Mock())
        if amp_skip:
            namespace['handle_amp_gradient'] = Mock(return_value=SimpleNamespace(scale_before=256, scale_after=128))
        tree = ast.parse((ROOT / 'RL_utils/train_drug3d_forRL.py').read_text())
        outer = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'train_forRL')
        step = next(node for node in outer.body if isinstance(node, ast.FunctionDef) and node.name == 'train')
        wrapper = ast.parse('def fixture():\n    amp_skipped_updates = 0\n    return None\n').body[0]
        wrapper.body[1:2] = [step, ast.Return(value=ast.Name(id='train', ctx=ast.Load()))]
        module = ast.fix_missing_locations(ast.Module(body=[wrapper], type_ignores=[]))
        exec(compile(module, '<RL train-step fixture>', 'exec'), namespace)
        return namespace['fixture'](), namespace

    def test_finite_loss_reaches_optimizer_step(self):
        train, values = self.fixture()
        train(1)
        values['scaler'].step.assert_called_once_with(values['optimizer'])
        values['scaler'].update.assert_called_once()
        self.assertEqual(values['append_record'].call_args.args[1]['losses']['loss'], 1.0)

    def test_nonfinite_loss_stops_before_backward(self):
        train, values = self.fixture(bad_loss=True)
        with self.assertRaises(checks.NonfiniteTrainingError) as caught:
            train(1)
        self.assertEqual(caught.exception.stage, 'train_loss')
        values['scaler'].scale.assert_not_called()
        values['scaler'].step.assert_not_called()
        self.assertEqual(values['current']['batch']['mol_id'].item(), 42)

    def test_bad_gradient_without_amp_stops_before_update(self):
        train, values = self.fixture(bad_gradient=True)
        with self.assertRaises(checks.NonfiniteTrainingError) as caught:
            train(1)
        self.assertEqual(caught.exception.stage, 'train_gradient_norm')
        values['scaler'].step.assert_not_called()

    def test_amp_overflow_uses_the_existing_skip_helper(self):
        train, values = self.fixture(bad_gradient=True, amp_skip=True)
        train(1)
        values['handle_amp_gradient'].assert_called_once()
        values['scaler'].step.assert_not_called()
        record = values['append_record'].call_args.args[1]
        self.assertEqual(record, dict(iteration=1, scale_before=256, scale_after=128))


if __name__ == '__main__':
    unittest.main()
