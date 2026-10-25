import ast
import csv
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

from scripts import probe_pretrain_batch as probe
from scripts.prepare_batch_probe_pbs import render_pbs


EXPERIMENT = probe.ROOT / 'configs/experiments/exp001_batch_probe.yml'
SMALL_EXPERIMENT = probe.ROOT / 'configs/experiments/exp002_batch_probe.yml'
GIT = shutil.which('git')
BASH = (str(Path(GIT).parent.parent / 'bin/bash.exe') if os.name == 'nt' and GIT
        else shutil.which('bash'))


class BatchProbeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def candidate(self, size, status='ok', used=60, updates=20):
        return dict(batch_size=size, status=status, estimated_peak_device_bytes=used,
                    device_total_bytes=100, measured_optimizer_updates=updates)

    def test_recommendation_requires_success_updates_and_headroom(self):
        results = [self.candidate(8), self.candidate(16, used=80),
                   self.candidate(24, used=81), self.candidate(32, updates=0),
                   self.candidate(48, status='oom')]
        self.assertEqual(probe.recommend(results, 0.8), 16)
        self.assertIsNone(probe.recommend(results[2:], 0.8))

    def test_only_cuda_oom_is_a_capacity_boundary(self):
        self.assertTrue(probe.is_cuda_oom(RuntimeError('CUDA out of memory. Tried to allocate ...')))
        self.assertFalse(probe.is_cuda_oom(RuntimeError('shape mismatch')))
        self.assertFalse(probe.is_cuda_oom(MemoryError('host memory exhausted')))

    def test_default_probe_is_bounded_and_includes_original_batch(self):
        experiment = probe.load_experiment(EXPERIMENT)
        self.assertIn(24, experiment['batch_sizes'])
        self.assertEqual(experiment['measured_steps'], 20)
        self.assertLessEqual(sum(experiment[key] for key in
                                 ('warmup_steps', 'measured_steps', 'stress_steps')), 30)

    def test_small_probe_only_changes_experiment_id_and_candidates(self):
        original = probe.load_experiment(EXPERIMENT)
        small = probe.load_experiment(SMALL_EXPERIMENT)
        self.assertEqual(small['id'], 'mdrl-exp002-batch-capacity')
        self.assertEqual(small['batch_sizes'], [2, 4, 6])
        self.assertEqual(original['batch_sizes'], [8, 16, 24, 32, 48, 64, 96, 128])
        excluded = {'id', 'batch_sizes'}
        self.assertEqual({key: value for key, value in original.items() if key not in excluded},
                         {key: value for key, value in small.items() if key not in excluded})

    def test_small_probe_uses_separate_run_and_preserves_first_probe(self):
        first_output = self.root / 'exp001/results'
        small_output = self.root / 'exp002/results'
        with patch.object(probe, 'launch_worker', side_effect=self.fake_worker({16: 'oom'})):
            probe.run_experiment(EXPERIMENT, first_output)
        first_summary = (first_output / 'summary.json').read_bytes()
        with patch.object(probe, 'launch_worker', side_effect=self.fake_worker({})) as launch:
            self.assertEqual(probe.run_experiment(SMALL_EXPERIMENT, small_output), 0)
        self.assertEqual(launch.call_count, 3)
        summary = json.loads((small_output / 'summary.json').read_text(encoding='utf-8'))
        self.assertEqual(summary['experiment_id'], 'mdrl-exp002-batch-capacity')
        self.assertEqual([row['batch_size'] for row in summary['results']], [2, 4, 6])
        self.assertEqual(summary['recommended_batch_size'], 6)
        self.assertEqual(summary['status'], 'completed_grid')
        self.assertEqual(summary['untested_batch_sizes'], [])
        self.assertEqual((small_output / 'experiment.yml').read_bytes(), SMALL_EXPERIMENT.read_bytes())
        self.assertEqual((first_output / 'summary.json').read_bytes(), first_summary)

    def test_missing_input_is_rejected_without_loading_or_processing_data(self):
        dataset_config = SimpleNamespace(root=str(self.root), split='split.pt',
                                         path_dict=SimpleNamespace(processed='processed.lmdb', summary='summary.csv'))
        with self.assertRaises(FileNotFoundError):
            probe.check_dataset_files(dataset_config)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_stress_selection_uses_training_split_not_validation(self):
        summary = self.root / 'summary.csv'
        with summary.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=['mol_id', 'n_heavy_atoms'])
            writer.writeheader()
            writer.writerows([dict(mol_id=10, n_heavy_atoms=20), dict(mol_id=11, n_heavy_atoms=45),
                             dict(mol_id=12, n_heavy_atoms=50), dict(mol_id=13, n_heavy_atoms=45)])
        dataset = SimpleNamespace(molid2idx={10: 0, 11: 1, 12: 2, 13: 3})
        self.assertEqual(probe.largest_training_indices(dataset, [0, 1, 3], summary, 2), [1, 3])
        with self.assertRaises(ValueError):
            probe.largest_training_indices(dataset, [0, 1, 3, 4], summary, 2)

    def fake_worker(self, outcomes):
        def launch(command, console, timeout):
            size = int(command[command.index('--candidate-batch') + 1])
            directory = Path(command[command.index('--outdir') + 1])
            status = outcomes.get(size, 'ok')
            if status == 'timeout':
                probe.write_json(directory / 'result.json', dict(batch_size=size, status='running', phase='dataset'))
                raise subprocess.TimeoutExpired(command, timeout)
            result = self.candidate(size, status)
            probe.write_json(directory / 'result.json', result)
            return 0 if status == 'ok' else 3
        return launch

    def test_controller_stops_at_oom_and_preserves_raw_results(self):
        output = self.root / 'results'
        with patch.object(probe, 'launch_worker', side_effect=self.fake_worker({16: 'oom'})) as launch:
            code = probe.run_experiment(EXPERIMENT, output)
        self.assertEqual(code, 0)
        self.assertEqual(launch.call_count, 2)
        summary = json.loads((output / 'summary.json').read_text(encoding='utf-8'))
        self.assertEqual(summary['status'], 'completed_at_oom')
        self.assertEqual(summary['recommended_batch_size'], 8)
        self.assertIn(24, summary['untested_batch_sizes'])
        self.assertTrue((output / 'batch_016/console.log').is_file())
        self.assertTrue((output / 'base_train.yml').is_file())
        with self.assertRaises(FileExistsError):
            probe.run_experiment(EXPERIMENT, output)

    def test_non_oom_error_suppresses_recommendation(self):
        output = self.root / 'results'
        with patch.object(probe, 'launch_worker', side_effect=self.fake_worker({16: 'error'})):
            self.assertEqual(probe.run_experiment(EXPERIMENT, output), 2)
        summary = json.loads((output / 'summary.json').read_text(encoding='utf-8'))
        self.assertEqual(summary['status'], 'error')
        self.assertIsNone(summary['recommended_batch_size'])

    def test_timeout_preserves_last_worker_phase(self):
        output = self.root / 'results'
        with patch.object(probe, 'launch_worker', side_effect=self.fake_worker({8: 'timeout'})):
            self.assertEqual(probe.run_experiment(EXPERIMENT, output), 2)
        summary = json.loads((output / 'summary.json').read_text(encoding='utf-8'))
        self.assertEqual(summary['results'][0]['status'], 'timeout')
        self.assertEqual(summary['results'][0]['phase'], 'dataset')

    def test_incomplete_worker_result_is_not_success(self):
        output = self.root / 'results'
        with patch.object(probe, 'launch_worker', side_effect=self.fake_worker({8: 'running'})):
            self.assertEqual(probe.run_experiment(EXPERIMENT, output), 2)
        summary = json.loads((output / 'summary.json').read_text(encoding='utf-8'))
        self.assertEqual(summary['results'][0]['status'], 'error')
        self.assertIsNone(summary['recommended_batch_size'])

    def test_timeout_and_interrupt_clean_up_owned_process_group(self):
        for error in (subprocess.TimeoutExpired('worker', 1), KeyboardInterrupt()):
            with self.subTest(error=type(error).__name__):
                process = MagicMock(pid=12345)
                process.__enter__.return_value = process
                process.wait.side_effect = [error, 0]
                with patch.object(probe.subprocess, 'Popen', return_value=process) as popen, \
                        patch.object(probe.os, 'killpg', create=True) as killpg, \
                        patch.object(probe.signal, 'SIGKILL', 9, create=True):
                    with self.assertRaises(type(error)):
                        probe.launch_worker(['fake-worker'], Mock(), 1)
                    killpg.assert_called_once_with(12345, 9)
                    self.assertTrue(popen.call_args.kwargs['start_new_session'])
                    self.assertEqual(process.wait.call_count, 2)

    def test_forward_arguments_match_training_entry(self):
        batch = SimpleNamespace(node_type='nodes', node_type_batch='node_batch',
                                halfedge_type='edges', halfedge_index='edge_index',
                                halfedge_type_batch='edge_batch', num_graphs=24)
        model = Mock()
        probe.loss_for_batch(model, batch, 'positions')
        original = ast.parse((probe.ROOT / 'scripts/train_drug3d.py').read_text(encoding='utf-8'))
        calls = [node for node in ast.walk(original) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute) and node.func.attr == 'get_loss']
        self.assertTrue(calls)
        expected = set(model.get_loss.call_args.kwargs)
        for call in calls:
            self.assertEqual({keyword.arg for keyword in call.keywords}, expected)

    def test_new_sources_support_python38_and_do_not_save_checkpoints(self):
        for source in ('scripts/probe_pretrain_batch.py', 'scripts/prepare_batch_probe_pbs.py'):
            text = (probe.ROOT / source).read_text(encoding='utf-8')
            tree = ast.parse(text, feature_version=(3, 8))
            self.assertFalse(any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                                 and node.func.attr in ('save', 'save_checkpoint') for node in ast.walk(tree)))


class BatchProbePBSTests(unittest.TestCase):
    def render(self, **kwargs):
        return render_pbs('/server/MDRL', '/server/MDRL/probe.yml', '/server/runs/exp001',
                          queue='gentai', resources='nodes=node1:ppn=32,walltime=02:00:00', **kwargs)

    def test_scheduler_mask_is_preserved_and_no_submission_is_performed(self):
        source = self.render(gpu=1)
        self.assertIn('if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then\n    export CUDA_VISIBLE_DEVICES=1\nfi', source)
        self.assertNotIn('qsub', source)
        self.assertIn('/home/zhuenqiang_1/.conda/envs/MDRL/bin/python', source)
        self.assertIn('nodes=node1:ppn=32,walltime=02:00:00', source)
        self.assertIn('"$PY" -u -B scripts/probe_pretrain_batch.py', source)
        self.assertIn('exec > "$DIRECT_LOG" 2>&1', source)
        self.assertIn('launcher_exit_status=$status', source)
        self.assertNotIn('exec "$PY"', source)

    def test_no_implicit_physical_gpu_and_rejects_directive_injection(self):
        source = self.render()
        self.assertNotIn('export CUDA_VISIBLE_DEVICES=', source)
        self.assertIn('exit 2', source)
        with self.assertRaises(ValueError):
            render_pbs('/repo', '/config', '/runs/test', 'gentai\n#PBS bad', 'nodes=node1')
        with self.assertRaises(ValueError):
            render_pbs('/repo', '/config', '/runs/test', 'gentai', 'nodes=node1\necho bad')

    def test_independent_job_name_does_not_change_execution(self):
        original = self.render(gpu=1)
        small = self.render(gpu=1, job_name='mdrl-batch002')
        self.assertEqual(small, original.replace('#PBS -N mdrl-batch001', '#PBS -N mdrl-batch002'))
        for name in ('bad\n#PBS bad', '123invalid', 'x' * 16):
            with self.subTest(job_name=name), self.assertRaises(ValueError):
                self.render(job_name=name)

    @unittest.skipUnless(BASH and Path(BASH).is_file(), 'Bash not available')
    def test_generated_pbs_shell_syntax(self):
        result = subprocess.run([BASH, '--noprofile', '--norc', '-n'],
                                input=self.render(gpu=0), text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
