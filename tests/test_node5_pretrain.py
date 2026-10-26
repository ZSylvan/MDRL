import ast
import copy
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import yaml

from scripts import prepare_pretrain_pbs as job


ROOT = job.ROOT
GIT = shutil.which('git')
BASH = (str(Path(GIT).parent.parent / 'bin/bash.exe') if os.name == 'nt' and GIT
        else shutil.which('bash'))


class Node5PretrainTests(unittest.TestCase):
    def config(self):
        return yaml.safe_load((ROOT / 'configs/train/train_MDRL_node5_b16.yml').read_text())

    def test_recipe_changes_only_approved_settings(self):
        original = yaml.safe_load((ROOT / 'configs/train/train_MDRL.yml').read_text())
        config = self.config()
        self.assertEqual(config['train']['batch_size'], 16)
        self.assertEqual(config['train']['num_workers'], 14)
        self.assertEqual(config['train']['max_iters'], 2400000)
        self.assertEqual(config['train']['val_every_epochs'], 10)
        self.assertEqual(config['train']['checkpoint_freq'], 5000)
        self.assertEqual(config['train']['checkpoint_milestones'], [1600000, 2400000])
        self.assertEqual(config['train']['max_checkpoint_files'], 50)
        for key in ('batch_size', 'num_workers', 'max_iters', 'val_freq'):
            original['train'].pop(key)
            config['train'].pop(key)
        for key in ('val_every_epochs', 'checkpoint_freq', 'checkpoint_milestones', 'max_checkpoint_files'):
            config['train'].pop(key)
        original['dataset'].pop('root')
        config['dataset'].pop('root')
        config.pop('logdir')
        self.assertEqual(config, original)

    def test_ten_epochs_resolve_from_actual_train_size_without_mutation(self):
        base = self.config()
        before = copy.deepcopy(base)
        config = job.prepare_config(base, 231523, '/server/run')
        self.assertEqual(base, before)
        self.assertEqual(config['train']['val_freq'], 144710)
        self.assertEqual(config['logdir'], str(Path('/server/run') / 'training'))
        self.assertEqual(job.prepare_config(base, 231521, '/server/run')['train']['val_freq'], 144710)
        self.assertEqual(job.prepare_config(base, 160, '/server/run')['train']['val_freq'], 100)
        freq = config['train']['val_freq']
        validations = list(range(freq, 2400001, freq)) + [2400000]
        self.assertEqual(len(validations), 17)
        self.assertNotIn(1600000, validations)
        self.assertEqual(len(set(validations) | {1600000, 2400000}) + 2, 20)

    def test_bad_split_and_checkpoint_policy_are_rejected(self):
        with self.assertRaises(ValueError):
            job.prepare_config(self.config(), 0, '/server/run')
        for key, value in (('max_checkpoint_files', 51), ('checkpoint_freq', 0),
                           ('checkpoint_milestones', [1600000, 1600000]),
                           ('checkpoint_milestones', [2400001])):
            config = self.config()
            config['train'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                job.prepare_config(config, 231523, '/server/run')

    def test_pbs_selects_node5_single_gpu_and_fifteen_cpus(self):
        source = job.render_pbs('/server/repo', '/server/run', 'abc', 'gentai',
                                'nodes=node5:ppn=15,walltime=240:00:00')
        self.assertIn('#PBS -l nodes=node5:ppn=15,walltime=240:00:00', source)
        self.assertIn('export CUDA_VISIBLE_DEVICES=0', source)
        self.assertIn('"$(hostname -s)" == node5', source)
        self.assertIn('--device cuda:0', source)
        self.assertIn('--query-compute-apps=pid', source)
        self.assertIn('repository changed after preparation', source)
        self.assertIn('set -o noclobber', source)
        self.assertIn('OPENBLAS_NUM_THREADS=1', source)
        self.assertNotIn('qsub', source)
        self.assertNotIn('--resume', source)
        if BASH and Path(BASH).is_file():
            checked = subprocess.run([BASH, '--noprofile', '--norc', '-n'], input=source,
                                     text=True, capture_output=True)
            self.assertEqual(checked.returncode, 0, checked.stderr)

    def test_resume_argument_is_quoted_and_does_not_replace_parent_outputs(self):
        source = job.render_pbs('/server/repo', '/server/new-run', 'abc', 'gentai',
                                'nodes=node5:ppn=15', '/server/old run/checkpoints/last.pt')
        self.assertIn("--resume '/server/old run/checkpoints/last.pt'", source)
        self.assertIn('JOB_DIR=/server/new-run', source)
        self.assertNotIn('rm ', source)

    def test_pbs_directive_injection_is_rejected(self):
        for queue, resources, directory in (('gentai\n#PBS bad', 'nodes=node5', '/server/run'),
                                             ('gentai', 'nodes=node5\nexit', '/server/run'),
                                             ('gentai', 'nodes=node5', 'relative/run')):
            with self.assertRaises(ValueError):
                job.render_pbs('/repo', directory, 'abc', queue, resources)

    def test_python38_syntax(self):
        for relative in ('scripts/prepare_pretrain_pbs.py', 'utils/pretrain_record.py',
                         'scripts/train_drug3d.py', 'utils/pretrain_resume.py'):
            ast.parse((ROOT / relative).read_text(), feature_version=(3, 8))

    def test_linux_generator_writes_effective_config_and_refuses_reusing_run_id(self):
        def attributes(value):
            return SimpleNamespace(**{key: attributes(item) if isinstance(item, dict) else item
                                      for key, item in value.items()})

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo = root / 'repo'
            config_dir = repo / 'configs/train'
            config_dir.mkdir(parents=True)
            data = root / 'data'
            data.mkdir()
            config = self.config()
            config['dataset']['root'] = str(data)
            config['train'].update(max_iters=24, checkpoint_milestones=[16, 24])
            (config_dir / 'train_MDRL_node5_b16.yml').write_text(yaml.safe_dump(config))
            (data / 'processed.lmdb').touch()
            (data / 'mol_summary.csv').touch()
            torch.save({10: 0, 20: 1}, data / 'processed_molid2idx.pt')
            torch.save({'train': [10, 20, 99]}, data / 'split_by_molid.pt')
            output = root / 'runs'
            # The PBS output path is POSIX even when these file tests run on Windows.
            with patch.object(job, 'ROOT', repo), patch.object(job, 'OUTPUT_ROOT', str(output)), \
                    patch.object(job.sys, 'platform', 'linux'), \
                    patch.object(job.sys, 'argv', ['prepare_pretrain_pbs.py', '--run-id', 'fresh']), \
                    patch.object(job, 'git_version', return_value={'commit': 'abc', 'dirty': False}), \
                    patch.object(job, 'render_pbs', return_value='#!/bin/bash\n'), \
                    patch.dict('sys.modules', {'easydict': SimpleNamespace(EasyDict=attributes)}):
                self.assertEqual(job.main(), 0)
                run = output / 'fresh'
                protocol = json.loads((run / 'protocol.json').read_text())
                self.assertEqual(protocol['train_size'], 2)
                self.assertEqual(protocol['raw_train_split_entries'], 3)
                self.assertEqual(protocol['steps_per_epoch'], 1)
                self.assertEqual(protocol['val_freq'], 10)
                self.assertEqual(protocol['validation_iterations'], [10, 20, 24])
                self.assertEqual(protocol['planned_named_checkpoint_iterations'], [10, 16, 20, 24])
                self.assertEqual(protocol['planned_files_without_resume'], 6)
                before = {path.name: path.read_bytes() for path in run.iterdir()}
                with self.assertRaises(FileExistsError):
                    job.main()
                self.assertEqual(before, {path.name: path.read_bytes() for path in run.iterdir()})
                with patch.object(job.sys, 'argv', ['prepare_pretrain_pbs.py', '--run-id', 'resume',
                                                    '--resume', str(run / 'missing.pt')]), \
                        patch.object(job, 'load_resume_checkpoint', side_effect=ValueError('wrong recipe')):
                    with self.assertRaisesRegex(ValueError, 'wrong recipe'):
                        job.main()
                    self.assertFalse((output / 'resume').exists())


if __name__ == '__main__':
    unittest.main()
