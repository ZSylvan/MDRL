import ast
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml

from RL_utils.task_config import load_task_config
from scripts import prepare_rl_pbs


ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT / 'RL_utils/config_RL/tasks'


class RLPBSTests(unittest.TestCase):
    def test_pbs_respects_user_resources_without_node_or_occupancy_gates(self):
        source = prepare_rl_pbs.render_pbs('/repo/MDRL', '/runs/example/chk1-auroraa',
                                          'chk1-auroraa', 'gentai',
                                          'nodes=node4:ppn=7,walltime=720:00:00', 1)
        self.assertIn('#PBS -l nodes=node4:ppn=7,walltime=720:00:00', source)
        self.assertIn('export CUDA_VISIBLE_DEVICES=1', source)
        self.assertIn('RL_utils/01_RL.py --config "$JOB_DIR/task.yml"', source)
        self.assertIn('--resume-if-exists', source)
        self.assertIn('exec >> "$JOB_DIR/launcher.log" 2>&1', source)
        self.assertIn('ENV_ROOT=/home/zhuenqiang_1/.conda/envs/MDRL', source)
        self.assertNotIn('nvidia-smi', source)
        self.assertNotIn('expected node', source)
        self.assertNotIn('git status', source)
        self.assertNotIn('qsub', source)

    def test_unsafe_pbs_directives_are_rejected(self):
        for queue, resources, directory, gpu in (
                ('gentai\n#PBS -V', 'nodes=node4:ppn=7', '/runs/job', 0),
                ('gentai', 'nodes=node4:ppn=7\nexit 0', '/runs/job', 0),
                ('gentai', 'nodes=node4:ppn=7', '/runs/with spaces', 0),
                ('gentai', 'nodes=node4:ppn=7', '/runs/job', -1)):
            with self.subTest(queue=queue, resources=resources, directory=directory, gpu=gpu):
                with self.assertRaises(ValueError):
                    prepare_rl_pbs.render_pbs('/repo', directory, 'task', queue, resources, gpu)

    def test_preparation_saves_independent_tasks_and_never_starts_an_experiment(self):
        configs = sorted(TASKS.glob('*.yml'))
        real_loader = lambda path: load_task_config(path, check_files=False)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = ['--run-id', 'example', '--output-root', str(root),
                    '--resources', 'nodes=node4:ppn=15,walltime=720:00:00', '--gpu', '1',
                    '--workers', '14', '--vina-cpu', '1', '--predictor-threads', '1',
                    '--train-batch-size', '16', '--sample-batch-size', '1', '--data-workers', '6']
            for filename in configs:
                args.extend(['--config', str(filename)])
            with patch.object(prepare_rl_pbs.sys, 'platform', 'linux'):
                with patch.object(prepare_rl_pbs, 'load_task_config', side_effect=real_loader):
                    with patch.object(prepare_rl_pbs, 'render_pbs', return_value='#!/bin/bash\n'):
                        with patch.object(prepare_rl_pbs.subprocess, 'check_output', return_value='abc123\n') as git:
                            with redirect_stdout(io.StringIO()):
                                self.assertEqual(prepare_rl_pbs.main(args), 0)
                            git.assert_called_once_with(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True)
                            with self.assertRaises(FileExistsError), redirect_stdout(io.StringIO()):
                                prepare_rl_pbs.main(args)
            jobs = sorted((root / 'example').glob('*/task.yml'))
            self.assertEqual(len(jobs), 3)
            for path in jobs:
                config = yaml.safe_load(path.read_text())
                self.assertEqual(config['expected_pretrain_iteration'], 985000)
                self.assertEqual(config['train_overrides'], dict(batch_size=16, num_workers=6))
                self.assertEqual(config['sample_overrides']['batch_size'], 1)
                self.assertEqual(config['docking']['workers'], 14)
                self.assertEqual(config['docking']['cpu'], 1)
                self.assertEqual(config['device'], 'cuda:0')
                self.assertFalse(Path(config['output_dir']).exists())
                self.assertTrue((path.parent / 'job.pbs').is_file())
                self.assertTrue((path.parent / 'task_source.yml').is_file())
            protocol = json.loads((root / 'example/protocol.json').read_text())
            self.assertFalse(protocol['submitted'])
            self.assertEqual(protocol['physical_gpu'], 1)
            ast.parse((ROOT / 'scripts/prepare_rl_pbs.py').read_text(), feature_version=(3, 8))


if __name__ == '__main__':
    unittest.main()
