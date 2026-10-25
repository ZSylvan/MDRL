import ast
import copy
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import yaml

from scripts import prepare_epoch_probe_pbs as epoch


ROOT = epoch.ROOT
GIT = shutil.which('git')
BASH = (str(Path(GIT).parent.parent / 'bin/bash.exe') if os.name == 'nt' and GIT
        else shutil.which('bash'))


class EpochProbeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)

    def test_configs_change_only_batch_total_steps_and_output(self):
        base = yaml.safe_load((ROOT / 'configs/train/train_MDRL_reproduction.yml').read_text())
        original = copy.deepcopy(base)
        configs = epoch.epoch_configs(base, 33, self.directory)
        self.assertEqual(base, original)
        self.assertEqual(list(configs), [16, 24])
        self.assertEqual([configs[b]['train']['max_iters'] for b in configs], [3, 2])
        for batch, config in configs.items():
            expected = copy.deepcopy(base)
            expected['train']['batch_size'] = batch
            expected['train']['max_iters'] = (33 + batch - 1) // batch
            expected['logdir'] = str(self.directory / ('batch_%03d' % batch))
            self.assertEqual(config, expected)
        with self.assertRaises(ValueError):
            epoch.epoch_configs(base, 0, self.directory)

    def fixture(self, batch, status='completed', skips=0, failures=None, code=0):
        folder = self.directory / ('batch_%03d/run' % batch)
        folder.mkdir(parents=True)
        steps = 3 if batch == 16 else 2
        state = dict(status=status, iteration=steps, completed_train_calls=steps - skips,
                     skipped_iterations=skips)
        (folder / 'run.json').write_text(json.dumps(state))
        (folder / 'failures.jsonl').write_text(''.join(json.dumps(event) + '\n' for event in (failures or [])))
        (self.directory / ('batch_%03d.exit' % batch)).write_text(str(code))

    def summarize(self):
        (self.directory / 'protocol.json').write_text(json.dumps(dict(train_size=33, iterations={'16': 3, '24': 2})))
        code = epoch.summarize(self.directory)
        result = json.loads((self.directory / 'summary.json').read_text())
        return code, result['results']

    def test_no_skip_complete_epoch_passes_for_both_batches(self):
        for batch in epoch.BATCHES:
            self.fixture(batch)
        code, rows = self.summarize()
        self.assertEqual(code, 0)
        self.assertEqual([row['status'] for row in rows], ['passed_one_train_pass_and_validation'] * 2)

    def test_oom_in_first_candidate_does_not_hide_second_result(self):
        self.fixture(16, 'completed_with_skips', 1, [{'error': 'CUDA out of memory. Tried to allocate ...'}])
        self.fixture(24)
        code, rows = self.summarize()
        self.assertEqual(code, 2)
        self.assertEqual(rows[0]['status'], 'oom')
        self.assertEqual(rows[0]['cuda_oom_events'], 1)
        self.assertEqual(rows[1]['status'], 'passed_one_train_pass_and_validation')

    def test_non_oom_skips_and_missing_candidates_never_pass(self):
        self.fixture(16, 'completed_with_skips', 1, [{'error': 'shape mismatch'}])
        code, rows = self.summarize()
        self.assertEqual(code, 2)
        self.assertEqual(rows[0]['status'], 'error_or_incomplete')
        self.assertEqual(rows[1]['status'], 'incomplete')

    def test_failed_validation_and_incomplete_work_do_not_pass(self):
        self.fixture(16, 'failed', 0, [{'error': 'CUDA out of memory'}], code=1)
        self.fixture(24)
        path = self.directory / 'batch_024/run/run.json'
        state = json.loads(path.read_text())
        state['iteration'] = 1
        path.write_text(json.dumps(state))
        code, rows = self.summarize()
        self.assertEqual(code, 2)
        self.assertEqual(rows[0]['status'], 'oom')
        self.assertEqual(rows[1]['status'], 'error_or_incomplete')

    def test_python38_and_original_loader_preserves_last_partial_batch(self):
        ast.parse((ROOT / 'scripts/prepare_epoch_probe_pbs.py').read_text(), feature_version=(3, 8))
        tree = ast.parse((ROOT / 'scripts/train_drug3d.py').read_text())
        loaders = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                   and isinstance(node.func, ast.Name) and node.func.id == 'DataLoader']
        for call in loaders:
            for keyword in call.keywords:
                if keyword.arg == 'drop_last':
                    self.assertFalse(ast.literal_eval(keyword.value))

    def test_pbs_continues_after_candidate_failure_and_avoids_busy_gpu(self):
        source = epoch.render_pbs('/server/MDRL', '/server/run', 'abc123', 'gentai',
                                  'nodes=node1:ppn=16,walltime=24:00:00', 0)
        self.assertIn('for batch in 16 24; do', source)
        self.assertIn('|| rc=$?', source)
        self.assertIn('--query-compute-apps=pid', source)
        self.assertIn('scheduler GPU mask differs', source)
        self.assertNotIn('qsub', source)
        self.assertNotIn('--resume', source)
        self.assertIn('set -o noclobber', source)
        if BASH and Path(BASH).is_file():
            result = subprocess.run([BASH, '--noprofile', '--norc', '-n'], input=source,
                                    text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
        with self.assertRaises(ValueError):
            epoch.render_pbs('/repo', '/server/run', 'abc123', 'gentai\n#PBS bad', 'nodes=node1', 0)


if __name__ == '__main__':
    unittest.main()
