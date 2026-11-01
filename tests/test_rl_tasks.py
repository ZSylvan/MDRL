import ast
import csv
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import yaml

from RL_utils import docking, scoring_workers, task_config


ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT / 'RL_utils/config_RL/tasks'


def fixture_score(index, smiles, molecule, directory):
    return dict(molecule_index=index, smiles=smiles, reward=float(molecule), worker_pid=os.getpid())


class TaskConfigTests(unittest.TestCase):
    def config(self):
        return yaml.safe_load((TASKS / 'chk1_auroraa.yml').read_text())

    def test_three_tasks_keep_the_same_pretraining_and_method(self):
        tasks = [task_config.load_task_config(path, check_files=False) for path in sorted(TASKS.glob('*.yml'))]
        self.assertEqual(len(tasks), 3)
        self.assertEqual(len({task['output_dir'] for task in tasks}), 3)
        self.assertEqual(len({task['initial_checkpoint'] for task in tasks}), 1)
        expected = [('BRD4', 'CDK2'), ('CHK1', 'AuroraA'), ('Thrombin', 'Trypsin')]
        for task, targets in zip(tasks, expected):
            self.assertEqual(tuple(target['name'] for target in task['targets']), targets)
            self.assertEqual(task['expected_pretrain_iteration'], 985000)
            self.assertEqual(task['train_overrides'], dict(batch_size=24, num_workers=9))
            self.assertEqual(task['sample_overrides'], dict(batch_size=512))
            self.assertEqual(task['rl'], dict(n_epochs=50, initial_num_mols=3000,
                                              new_num_mols_per_epoch=2000, top_k=4000))
            self.assertEqual(task['docking']['workers'], 4)
            self.assertEqual(task['docking']['cpu'], 1)
            self.assertIn('/MDRL/bin/vina', task['docking']['executable'])
            self.assertTrue(Path(task['train_config']).is_file())
            self.assertTrue(Path(task['sample_config']).is_file())
            self.assertTrue(Path(task['fscores']).is_file())
            for target in task['targets']:
                self.assertEqual(target['box_size'], [126, 126, 126])
                self.assertEqual(target['normalization'], dict(mu=-9.0, sigma=1.0, minimize=True))

    def test_bad_configs_are_rejected_before_work(self):
        invalid = []
        for key, value in [('workers', 0), ('cpu', -1), ('timeout', 0)]:
            config = self.config()
            config['docking'][key] = value
            invalid.append(config)
        for value in (0, -1, 1.5):
            config = self.config()
            config['sample_overrides']['batch_size'] = value
            invalid.append(config)
        config = self.config()
        config['targets'][1]['name'] = config['targets'][0]['name']
        invalid.append(config)
        config = self.config()
        config['targets'][0]['box_size'][0] = 0
        invalid.append(config)
        config = self.config()
        config['scoring']['sa']['sigma'] = float('nan')
        invalid.append(config)
        with tempfile.TemporaryDirectory() as temporary:
            filename = Path(temporary) / 'task.yml'
            for config in invalid:
                filename.write_text(yaml.safe_dump(config))
                with self.subTest(config=config), self.assertRaises(ValueError):
                    task_config.load_task_config(filename, check_files=False)

    def test_csv_contains_only_the_configured_two_targets(self):
        config = self.config()
        with tempfile.TemporaryDirectory() as temporary:
            filename = Path(temporary) / 'scoring.csv'
            task_config.write_scoring_csv(config, filename)
            with filename.open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row['name'] for row in rows], ['qed', 'sa', 'CHK1_le', 'AuroraA_le'])
            self.assertEqual(rows[2]['file'], config['targets'][0]['ligand_model'])
            with self.assertRaises(FileExistsError):
                task_config.write_scoring_csv(config, filename)

    def test_check_config_has_no_model_imports_or_output_writes(self):
        spec = importlib.util.spec_from_file_location('rl_entry_for_test', ROOT / 'RL_utils/01_RL.py')
        entry = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(entry)
        config = self.config()
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            dummy = directory / 'input'
            dummy.touch()
            for key in ('initial_checkpoint', 'train_config', 'sample_config', 'fscores'):
                config[key] = str(dummy)
            config['docking']['executable'] = str(dummy)
            for target in config['targets']:
                target.update(receptor=str(dummy), ligand_model=str(dummy))
            config['output_dir'] = str(directory / 'untouched')
            filename = directory / 'task.yml'
            filename.write_text(yaml.safe_dump(config))
            before = set(os.listdir(directory))
            with patch.object(entry, 'run') as run, patch('sys.stdout', new=io.StringIO()):
                self.assertEqual(entry.main(['--config', str(filename), '--check-config']), 0)
                run.assert_not_called()
            self.assertEqual(set(os.listdir(directory)), before)

    def test_training_overrides_merge_without_dropping_optimizer_settings(self):
        spec = importlib.util.spec_from_file_location('rl_overrides_test', ROOT / 'RL_utils/01_RL.py')
        entry = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(entry)
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / 'train.yml'
            config = entry._effective_config(ROOT / 'RL_utils/config_RL/train_MDRL_RL.yml',
                                            'train', dict(batch_size=16, optimizer=dict(lr=2e-6)),
                                            2023, destination)
            self.assertEqual(config['train']['batch_size'], 16)
            self.assertEqual(config['train']['optimizer']['lr'], 2e-6)
            self.assertEqual(config['train']['optimizer']['weight_decay'], 1e-8)
            with self.assertRaises(ValueError):
                entry._effective_config(ROOT / 'RL_utils/config_RL/train_MDRL_RL.yml',
                                        'train', dict(batch_szie=16), 2023, Path(temporary) / 'bad.yml')

    def test_modules_accept_explicit_settings_and_python38(self):
        for relative, function in [('sample_drug3d_forRL.py', 'sample'), ('train_drug3d_forRL.py', 'train_forRL')]:
            source = (ROOT / 'RL_utils' / relative).read_text()
            tree = ast.parse(source, feature_version=(3, 8))
            definition = next(item for item in tree.body if isinstance(item, ast.FunctionDef) and item.name == function)
            self.assertIn('config_path', [arg.arg for arg in definition.args.kwonlyargs])
            self.assertNotIn('parse_known_args', source)
            self.assertLess(source.index('import distutils.version'), source.index('import torch.utils.tensorboard'))
        for relative in ('01_RL.py', 'docking.py', 'task_config.py', 'scoring_workers.py'):
            ast.parse((ROOT / 'RL_utils' / relative).read_text(), feature_version=(3, 8))


class DockingTests(unittest.TestCase):
    def target(self):
        return dict(name='CHK1', receptor='/input/CHK1.pdbqt', center=[1, 2, 3], box_size=[126] * 3,
                    normalization=dict(mu=-9.0, sigma=1.0, minimize=True))

    def settings(self):
        return dict(executable='/MDRL/bin/vina', cpu=2, timeout=300, exhaustiveness=8, num_modes=1)

    def test_affinity_and_original_half_gaussian(self):
        self.assertEqual(docking.parse_affinity('header\n   1   -8.3  0.0  0.0\n'), -8.3)
        self.assertEqual(docking.normalize_affinity(-10, self.target()['normalization']), 1)
        self.assertAlmostEqual(docking.normalize_affinity(-8, self.target()['normalization']), math.exp(-0.5))
        with self.assertRaises(ValueError):
            docking.parse_affinity('ERROR: receptor unavailable')

    def test_uses_configured_vina_cpu_target_and_log(self):
        with tempfile.TemporaryDirectory() as temporary:
            logfile = Path(temporary) / 'CHK1.log'
            response = SimpleNamespace(returncode=0, stdout='1 -8.3 0.0 0.0\n', stderr='warning')
            with patch.object(docking.subprocess, 'run', return_value=response) as execute:
                result = docking.run_vina('/ligand.pdbqt', self.target(), self.settings(), logfile)
            command = execute.call_args.args[0]
            self.assertEqual(command[0], '/MDRL/bin/vina')
            self.assertEqual(command[command.index('--receptor') + 1], '/input/CHK1.pdbqt')
            self.assertEqual(command[command.index('--cpu') + 1], '2')
            self.assertEqual(command[command.index('--out') + 1], str(logfile.with_suffix('.pdbqt')))
            self.assertEqual(execute.call_args.kwargs['timeout'], 300)
            self.assertEqual(result['affinity'], -8.3)
            self.assertIn('warning', logfile.read_text())

    def test_failure_and_timeout_are_not_replaced_with_rewards(self):
        with tempfile.TemporaryDirectory() as temporary:
            logfile = Path(temporary) / 'failed.log'
            response = SimpleNamespace(returncode=1, stdout='', stderr='failed receptor')
            with patch.object(docking.subprocess, 'run', return_value=response), self.assertRaises(RuntimeError):
                docking.run_vina('/ligand', self.target(), self.settings(), logfile)
            self.assertIn('failed receptor', logfile.read_text())
            timeout = subprocess.TimeoutExpired('vina', 300, output=b'partial', stderr=b'error')
            with patch.object(docking.subprocess, 'run', side_effect=timeout), self.assertRaises(RuntimeError):
                docking.run_vina('/ligand', self.target(), self.settings(), logfile)
            self.assertIn('TIMEOUT', logfile.read_text())


class ParallelScoringTests(unittest.TestCase):
    def test_worker_limits_prediction_threads_and_keeps_predictions_on_cpu(self):
        model = Mock()
        builder = SimpleNamespace(build_scoring_function=Mock(return_value=(dict(le=SimpleNamespace(rfr=model)), None)))
        config = dict(scoring=dict(predictor_threads=2), fscores='/scores.gz')
        with patch.dict('sys.modules', {'RL_utils.score_fuc': builder}), patch.dict(os.environ):
            with patch.object(scoring_workers, '_CONFIG', None), patch.object(scoring_workers, '_SCORERS', None):
                scoring_workers.initialize_worker(config, '/scoring.csv')
                self.assertEqual(os.environ['OMP_NUM_THREADS'], '2')
                model.set_params.assert_called_once_with(n_jobs=2, device='cpu')

    def test_worker_keeps_the_original_reward_combination(self):
        scorers = {name: SimpleNamespace(raw_score=lambda smiles, value=value: value,
                                         modify_score=lambda value: value)
                   for name, value in zip(('qed', 'sa', 'a_le', 'b_le'), (0.1, 0.2, 0.3, 0.4))}
        config = dict(targets=[], docking={})
        docked = [dict(normalized_score=0.6), dict(normalized_score=0.8)]
        with patch.object(scoring_workers, '_SCORERS', scorers), patch.object(scoring_workers, '_CONFIG', config):
            with patch.object(docking, 'dock_molecule', return_value=docked):
                result = scoring_workers.score_one(7, 'C', None, '/unused')
        self.assertNotIn('error', result)
        self.assertAlmostEqual(result['reward'], (0.25 + 0.7) / 2)

    def test_nonfinite_prediction_never_reaches_docking_or_a_reward(self):
        scorer = SimpleNamespace(raw_score=lambda smiles: float('nan'), modify_score=lambda value: value)
        with patch.object(scoring_workers, '_SCORERS', dict(a_le=scorer)):
            with patch.object(docking, 'dock_molecule') as dock:
                result = scoring_workers.score_one(0, 'C', None, '/unused')
        dock.assert_not_called()
        self.assertIn('error', result)
        self.assertNotIn('reward', result)
        json.dumps(result, allow_nan=False)

    def test_scoring_records_cross_a_real_spawn_process_boundary_without_docking(self):
        population = [dict(smiles=str(i), rdmol=value) for i, value in enumerate([0.1, 0.3, 0.2])]
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context('spawn')) as pool:
                with patch.object(scoring_workers, 'score_one', fixture_score):
                    result = scoring_workers.score_population(population, pool, 2, directory)
            self.assertEqual([item['score'] for item in result], [0.3, 0.2, 0.1])
            records = [json.loads(line) for line in (directory / 'scores.jsonl').read_text().splitlines()]
            self.assertTrue(all(item['worker_pid'] != os.getpid() for item in records))

    def test_pool_uses_spawn_and_configured_worker_count(self):
        config = dict(docking=dict(workers=3))
        with patch.object(scoring_workers, 'ProcessPoolExecutor') as executor:
            scoring_workers.create_pool(config, '/scoring.csv')
        kwargs = executor.call_args.kwargs
        self.assertEqual(kwargs['max_workers'], 3)
        self.assertEqual(kwargs['mp_context'].get_start_method(), 'spawn')

    def test_parallel_completion_preserves_ranking_and_tie_order(self):
        population = [dict(smiles=str(index), rdmol=reward) for index, reward in enumerate([0.2, 0.9, 0.9, 0.1])]
        with tempfile.TemporaryDirectory() as temporary, ThreadPoolExecutor(max_workers=2) as pool:
            directory = Path(temporary)
            with patch.object(scoring_workers, 'score_one', fixture_score):
                ranked = scoring_workers.score_population(population, pool, 2, directory)
            self.assertEqual([m['smiles'] for m in ranked], ['1', '2', '0', '3'])
            records = [json.loads(line) for line in (directory / 'scores.jsonl').read_text().splitlines()]
            self.assertEqual(len(records), 4)
            self.assertEqual({record['molecule_index'] for record in records}, set(range(4)))

    def test_failure_stops_queuing_and_records_the_molecule(self):
        class FailedPool:
            def __init__(self):
                self.submitted = 0

            def submit(self, function, index, smiles, molecule, directory):
                self.submitted += 1
                future = Future()
                future.set_result(dict(molecule_index=index, smiles=smiles, error='bad score'))
                return future

        pool = FailedPool()
        population = [dict(smiles=str(i), rdmol=None) for i in range(100)]
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            with self.assertRaises(RuntimeError):
                scoring_workers.score_population(population, pool, 2, directory)
            self.assertEqual(pool.submitted, 4)
            failure = json.loads((directory / 'failures.jsonl').read_text())
            self.assertEqual(failure['error'], 'bad score')


if __name__ == '__main__':
    unittest.main()
