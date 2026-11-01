import ast
import copy
import importlib.util
import json
import os
from pathlib import Path
import random
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch
import yaml

from RL_utils import run_state, scoring_workers
from RL_utils.task_config import append_record, load_task_config


ROOT = Path(__file__).resolve().parents[1]


def entry_module():
    spec = importlib.util.spec_from_file_location('rl_resume_test_entry', ROOT / 'RL_utils/01_RL.py')
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    return entry


def fake_chemistry():
    def write(molecule, filename):
        Path(filename).write_text('mock molecule\n')
    return SimpleNamespace(Chem=SimpleNamespace(MolToMolFile=write))


class AttributeDict(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc


def attribute_dict(value):
    return AttributeDict({key: attribute_dict(item) if isinstance(item, dict) else item
                          for key, item in value.items()})


class RLResumeTests(unittest.TestCase):
    def test_population_keeps_duplicate_samples_in_separate_sdf_files(self):
        entry = entry_module()
        population = [dict(smiles='CC', rdmol=None), dict(smiles='CC', rdmol=None)]
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / 'generated'
            with patch.dict('sys.modules', {'rdkit': fake_chemistry()}):
                entry.save_population(population, directory)
                self.assertEqual(len(list(directory.glob('*.sdf'))), 2)
                self.assertEqual(len((directory / 'mol_summary.csv').read_text().splitlines()), 3)
                self.assertEqual(entry.load_population(directory), population)
                with self.assertRaises(FileExistsError):
                    entry.save_population(population, directory)

    def test_sampling_journal_restores_completed_batches_and_rng(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            identity = dict(checkpoint='fixed.pt', requested=3, round=0)
            self.assertEqual(run_state.read_sampling_batches(directory, identity), ([], 0, None, 0))
            rng = run_state.capture_rng('cpu')
            expected = (random.random(), np.random.random(), torch.rand(1))
            run_state.save_state(directory / 'batch_000000.pt', dict(
                molecules=[dict(smiles='CC', rdmol=None)], failed_total=2, rng=rng))
            (directory / 'batch_000001.pt.tmp').write_bytes(b'incomplete')
            molecules, failed, restored, count = run_state.read_sampling_batches(directory, identity)
            self.assertEqual((len(molecules), failed, count), (1, 2, 1))
            run_state.restore_rng(restored, 'cpu')
            actual = (random.random(), np.random.random(), torch.rand(1))
            self.assertEqual(actual[:2], expected[:2])
            self.assertTrue(torch.equal(actual[2], expected[2]))
            with self.assertRaises(ValueError):
                run_state.read_sampling_batches(directory, dict(identity, requested=4))

    def test_resume_scoring_reuses_only_matching_completed_records(self):
        population = [dict(smiles=str(i), rdmol=None) for i in range(3)]
        with tempfile.TemporaryDirectory() as temporary, ThreadPoolExecutor(max_workers=2) as pool:
            directory = Path(temporary)
            filename = directory / 'scores.jsonl'
            append_record(filename, dict(molecule_index=0, smiles='0', reward=0.2))
            with filename.open('ab') as handle:
                handle.write(b'{"molecule_index":')
            score = Mock(side_effect=lambda index, smiles, molecule, path:
                         dict(molecule_index=index, smiles=smiles, reward=index / 2))
            with patch.object(scoring_workers, 'score_one', score):
                result = scoring_workers.score_population(population, pool, 2, directory, resume=True)
            self.assertEqual(sorted(call.args[0] for call in score.call_args_list), [1, 2])
            self.assertEqual([item['smiles'] for item in result], ['2', '1', '0'])
            self.assertEqual(len(run_state.read_records(filename)), 3)
            with self.assertRaises(ValueError):
                scoring_workers.score_population(list(reversed(population)), pool, 2, directory, resume=True)

    def test_actual_sampler_resumes_batches_with_plain_checkpoint_config(self):
        tree = ast.parse((ROOT / 'RL_utils/sample_drug3d_forRL.py').read_text())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'sample')
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            config_path = directory / 'sample.yml'
            config_path.write_text('{}')
            config = attribute_dict(dict(sample=dict(seed=2023, batch_size=1, save_traj_prob=0)))
            checkpoint = directory / 'model.pt'
            run_state.save_state(checkpoint, dict(model={}, config=dict(
                model=dict(name='diffusion'), chem=dict(atomic_numbers=[6], mol_bond_types=[1]),
                transform=dict(use_mask_node=False, use_mask_edge=False))))
            outdir = directory / 'sampling'
            journal = outdir / 'batches'
            identity = dict(checkpoint=str(checkpoint), requested=2, round=0, config=config, batch_size=0)
            run_state.read_sampling_batches(journal, identity)
            old_molecule = dict(smiles='CC', rdmol='saved molecule')
            run_state.save_state(journal / 'batch_000000.pt', dict(
                molecules=[old_molecule], failed_total=1, rng=run_state.capture_rng('cpu')))
            model = Mock()
            model.to.return_value = model
            model.sample.return_value = dict(pred=[torch.zeros(1)])
            factory = Mock(return_value=model)
            logger = Mock()
            logger.handlers = []
            featurizer = SimpleNamespace(num_node_types=1, num_edge_types=1,
                                        decode_output=Mock(return_value={}))

            def log_dir(root, **kwargs):
                path = Path(root) / 'segment'
                path.mkdir()
                return str(path)

            namespace = dict(
                torch=torch, np=np, os=os, sys=SimpleNamespace(argv=['sample']), shutil=shutil,
                time=time, Path=Path, SimpleNamespace=SimpleNamespace, EasyDict=attribute_dict,
                load_config=lambda path: config, seed_all=lambda seed: None,
                load_state=run_state.load_state, save_state=run_state.save_state,
                read_sampling_batches=run_state.read_sampling_batches,
                capture_rng=run_state.capture_rng, restore_rng=run_state.restore_rng,
                get_new_log_dir=log_dir, get_logger=lambda *args: logger,
                append_record=append_record, print_pool_status=lambda *args: None,
                FeaturizeMol=lambda *args, **kwargs: featurizer, MDRL=factory,
                make_data_placeholder=lambda **kwargs: dict(batch_node=torch.zeros(1),
                    halfedge_index=torch.zeros(1), batch_halfedge=torch.zeros(1)),
                seperate_outputs=lambda *args: [dict(pred=[None] * 3, halfedge_index=None)],
                reconstruct_from_generated_with_edges=lambda *args, **kwargs: 'new molecule',
                Chem=SimpleNamespace(MolToSmiles=lambda molecule: 'CC'), MolReconsError=ValueError)
            with patch('torch.utils.tensorboard', create=True) as tensorboard:
                exec(compile(ast.Module(body=[function], type_ignores=[]), '<RL sampler fixture>', 'exec'), namespace)
                generated = namespace['sample'](str(checkpoint), 2, 0, config_path=str(config_path),
                                                 device='cpu', outdir=str(outdir), resume=True)
                self.assertEqual(len(generated), 2)
                self.assertEqual(generated[0], old_molecule)
                factory.assert_called_once()
                self.assertEqual(factory.call_args.kwargs['config'].name, 'diffusion')
                model.sample.assert_called_once()
                model.load_state_dict.assert_called_once_with({})
                tensorboard.SummaryWriter.return_value.close.assert_called_once()
                factory.reset_mock()
                repeated = namespace['sample'](str(checkpoint), 2, 0, config_path=str(config_path),
                                               device='cpu', outdir=str(outdir), resume=True)
                self.assertEqual(repeated, generated)
                factory.assert_not_called()
            restored, failed, _, batches = run_state.read_sampling_batches(journal, identity)
            self.assertEqual((len(restored), failed, batches), (2, 1, 2))

    def test_training_components_restore_optimizer_scheduler_and_random_state(self):
        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        model(torch.ones(1, 1)).sum().backward()
        optimizer.step()
        scheduler.step(0.5)
        saved = dict(optimizer=copy.deepcopy(optimizer.state_dict()),
                     scheduler=copy.deepcopy(scheduler.state_dict()), scaler={},
                     rng_state=run_state.capture_rng('cpu'))
        replacement = torch.optim.AdamW(model.parameters(), lr=0.1)
        replacement_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(replacement)
        scaler = Mock()
        run_state.restore_training_components(saved, replacement, replacement_scheduler, scaler, 'cpu')
        self.assertEqual(replacement.param_groups[0]['lr'], 1e-6)
        self.assertEqual(replacement_scheduler.state_dict(), scheduler.state_dict())
        self.assertEqual(len(replacement.state), len(optimizer.state))
        scaler.load_state_dict.assert_called_once_with({})

    def test_training_resume_rejects_diagnostics_wrong_round_and_nonfinite_weights(self):
        config = dict(train=dict(max_iters=10000, use_amp=False))
        saved = dict(config=config, model=dict(w=torch.ones(1)), optimizer={}, scheduler={}, scaler={},
                     iteration=5000, round=1, source_checkpoint='init.pt', dataset_root='pool',
                     dataset_split={}, rng_state={}, amp_skipped_updates=0)
        run_state.validate_round_checkpoint(saved, config, 'pool', 1, 'init.pt')
        for change in (dict(diagnostic_only=True), dict(round=2), dict(iteration=10001),
                       dict(model=dict(w=torch.tensor([float('nan')])))):
            with self.subTest(change=change), self.assertRaises(ValueError):
                run_state.validate_round_checkpoint(dict(saved, **change), config, 'pool', 1, 'init.pt')

    def exercise_interruption(self, stage):
        entry = entry_module()
        source = ROOT / 'RL_utils/config_RL/tasks/chk1_auroraa.yml'
        config = load_task_config(source, check_files=False)
        config['rl'].update(n_epochs=3, initial_num_mols=3, new_num_mols_per_epoch=2, top_k=4)
        fired = [False]
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            config.update(output_dir=str(directory / 'output'), device='cpu')
            checkpoint = directory / 'initial.pt'
            training = yaml.safe_load(Path(config['train_config']).read_text())
            run_state.save_state(checkpoint, dict(iteration=985000, config=training, model=dict(w=torch.ones(1))))
            config['initial_checkpoint'] = str(checkpoint)

            def interrupt(where, epoch):
                if stage == where and epoch == 1 and not fired[0]:
                    fired[0] = True
                    raise RuntimeError('simulated interruption')

            def sample(path, count, epoch, **kwargs):
                interrupt('sampling', epoch)
                if stage == 'short_sampling' and epoch == 1 and not fired[0]:
                    fired[0] = True
                    return [dict(smiles='CC', rdmol=None)] * (count - 1)
                return [dict(smiles='CC', rdmol=None) for _ in range(count)]

            recovery_path = directory / 'output/round_001/training/segment/checkpoints/epoch1_it5000.pt'
            def train(root, epoch, path, **kwargs):
                if stage == 'training' and epoch == 1 and not fired[0]:
                    recovery_path.parent.mkdir(parents=True)
                    run_state.save_state(recovery_path, dict(iteration=5000))
                interrupt('training', epoch)
                result = directory / ('trained_%d.pt' % epoch)
                run_state.save_state(result, dict(model=dict(w=torch.ones(1))))
                return str(result)

            def score(population, pool, workers, path, **kwargs):
                interrupt('scoring', int(path.name.rsplit('_', 1)[1]))
                for molecule in population:
                    molecule['score'] = 0.5
                return population

            sampler, trainer = Mock(side_effect=sample), Mock(side_effect=train)
            modules = {
                'rdkit': fake_chemistry(),
                'RL_utils.sample_drug3d_forRL': SimpleNamespace(sample=sampler),
                'RL_utils.train_drug3d_forRL': SimpleNamespace(train_forRL=trainer),
                'RL_utils.scoring_workers': SimpleNamespace(create_pool=lambda *args: Mock(), score_population=score),
            }
            with patch.dict('sys.modules', modules):
                message = 'Sampling produced' if stage == 'short_sampling' else 'simulated interruption'
                with self.assertRaisesRegex(RuntimeError, message):
                    entry.run(config, source)
                if stage == 'short_sampling':
                    self.assertFalse((directory / 'output/round_001/generated/manifest.json').exists())
                checkpoint.unlink()
                entry.run(config, source, resume=True)
            self.assertEqual(sampler.call_count, 4 if stage in ('sampling', 'short_sampling') else 3)
            self.assertEqual(trainer.call_count, 3 if stage == 'training' else 2)
            if stage == 'training':
                self.assertEqual(trainer.call_args_list[1].kwargs['resume_from'], str(recovery_path))
            output = directory / 'output'
            for epoch, count in ((0, 3), (1, 2), (2, 2)):
                generated = entry.load_population(output / ('round_%03d/generated' % epoch))
                self.assertEqual(len(generated), count)
                self.assertEqual(len(list((output / ('round_%03d/generated' % epoch)).glob('*.sdf'))), count)
            self.assertEqual(len(entry.load_population(output / 'round_001/training_population')), 3)
            self.assertEqual(len(entry.load_population(output / 'round_002/training_population')), 4)
            self.assertTrue((output / 'round_001/checkpoints/final.pt').is_file())
            self.assertTrue((output / 'round_002/checkpoints/final.pt').is_file())
            self.assertEqual(len((output / 'rounds.jsonl').read_text().splitlines()), 3)
            changed = copy.deepcopy(config)
            changed['train_overrides']['batch_size'] = 8
            with patch.dict('sys.modules', modules), self.assertRaises(ValueError):
                entry.run(changed, source, resume=True)

    def test_resume_after_training_interruption(self):
        self.exercise_interruption('training')

    def test_resume_after_sampling_interruption_does_not_retrain(self):
        self.exercise_interruption('sampling')

    def test_resume_after_scoring_interruption_does_not_regenerate(self):
        self.exercise_interruption('scoring')

    def test_insufficient_sampling_is_not_committed_as_complete(self):
        self.exercise_interruption('short_sampling')


if __name__ == '__main__':
    unittest.main()
