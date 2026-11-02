"""Check parser compatibility and readable preprocessing failure messages."""

import ast
from contextlib import redirect_stdout
import importlib.util
import io
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock, patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


class NumpyWithoutLong:
    def __getattr__(self, name):
        if name == 'long':
            raise AttributeError("module 'numpy' has no attribute 'long'")
        return getattr(np, name)


class RLDatasetTests(unittest.TestCase):
    def test_generated_molecule_parses_without_numpy_long(self):
        spec = importlib.util.spec_from_file_location('parser_numpy_compat', ROOT / 'utils/parser.py')
        parser = importlib.util.module_from_spec(spec)
        with patch.dict('sys.modules', {'rdkit': SimpleNamespace(
                Chem=SimpleNamespace(MolToSmiles=lambda molecule: 'CO'))}):
            spec.loader.exec_module(parser)
        parser.np = NumpyWithoutLong()
        atoms = [SimpleNamespace(GetAtomicNum=lambda: 6), SimpleNamespace(GetAtomicNum=lambda: 8)]
        bond = SimpleNamespace(GetBondType=lambda: 1, GetBeginAtomIdx=lambda: 0, GetEndAtomIdx=lambda: 1)
        conformer = SimpleNamespace(GetAtomPosition=lambda index: [float(index), 0.0, 0.0])
        molecule = SimpleNamespace(GetNumAtoms=lambda: 2, GetNumBonds=lambda: 1,
                                   GetConformer=lambda: conformer, GetAtoms=lambda: atoms,
                                   GetBonds=lambda: [bond])
        parsed = parser.parse_conf_list([molecule], smiles='CO')
        self.assertEqual(parsed['num_confs'], 1)
        self.assertEqual(parsed['bond_type'].dtype, np.dtype('int64'))
        self.assertEqual(parsed['bond_index'].dtype, np.dtype('int64'))
        self.assertEqual(parsed['bond_index'].tolist(), [[0, 1], [1, 0]])
        self.assertEqual(parsed['bond_type'].tolist(), [1, 1])
        self.assertEqual(parsed['pos_all_confs'].shape, (1, 2, 3))

    def test_preprocessing_reports_the_actual_skipped_molecule_error(self):
        source = ast.parse((ROOT / 'RL_utils/dataset_forRL.py').read_text())
        dataset = next(node for node in source.body if isinstance(node, ast.ClassDef))
        process = next(node for node in dataset.body if isinstance(node, ast.FunctionDef) and node.name == '_process')

        class Rows:
            def __len__(self):
                return 1

            def iterrows(self):
                return iter([(0, dict(mol_id=0, smiles='CO'))])

        db = MagicMock()
        environment = dict(
            os=os, lmdb=SimpleNamespace(open=Mock(return_value=db)),
            pd=SimpleNamespace(read_csv=lambda *args, **kwargs: Rows()),
            tqdm=lambda iterable, **kwargs: iterable,
            Chem=SimpleNamespace(SDMolSupplier=Mock(side_effect=AttributeError('parser failure detail'))))
        exec(compile(ast.Module(body=[process], type_ignores=[]), '<RL preprocessing fixture>', 'exec'), environment)
        output = io.StringIO()
        with redirect_stdout(output):
            environment['_process'](SimpleNamespace(processed_path='unused.lmdb',
                                                     summary_path='unused.csv', sdf_path='unused'))
        self.assertIn("AttributeError('parser failure detail')", output.getvalue())
        self.assertIn('Processed 0 molecules Skipped 1 molecules', output.getvalue())
        db.close.assert_called_once()


if __name__ == '__main__':
    unittest.main()
