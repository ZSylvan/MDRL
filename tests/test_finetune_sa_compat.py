from pathlib import Path
from unittest.mock import patch
import pytest
from mdrl_finetune import scoring
import builtins
import importlib
from rdkit import Chem, RDConfig


def test_fragment_loader_does_not_swallow_permission_error(tmp_path):
    path = tmp_path / "scores.pkl.gz"
    path.write_bytes(b"x")
    module = type("M", (), {"readFragmentScores": staticmethod(lambda value: (_ for _ in ()).throw(PermissionError(value)))})()
    with pytest.raises(PermissionError):
        scoring._read_fragment_scores(module, path)


def test_fragment_loader_retries_legacy_suffix(tmp_path):
    path = tmp_path / "scores.pkl.gz"
    path.write_bytes(b"x")
    seen = []
    def read(value):
        seen.append(value)
        if value.endswith(".pkl.gz"):
            raise FileNotFoundError(2, "missing", value + ".pkl.gz")
    module = type("M", (), {"readFragmentScores": staticmethod(read)})()
    scoring._read_fragment_scores(module, path)
    assert seen[-1].endswith("scores") and len(seen) == 2


def test_fallback_rdconfig_score_matches_normal_import(monkeypatch):
    normal = scoring._load_sascorer()
    mol = Chem.MolFromSmiles("c1ccccc1")
    scores = Path(normal.__file__).parent / "fpscores.pkl.gz"
    scoring._read_fragment_scores(normal, scores)
    expected = normal.calculateScore(mol)
    original = builtins.__import__
    def block(name, *args, **kwargs):
        if name.startswith("rdkit.Contrib"):
            raise ModuleNotFoundError("blocked", name="rdkit.Contrib")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", block)
    fallback = scoring._load_sascorer()
    scoring._read_fragment_scores(fallback, scores)
    assert fallback.calculateScore(mol) == pytest.approx(expected)


def test_non_layout_module_not_found_is_not_swallowed(monkeypatch):
    original = builtins.__import__
    def broken(name, *args, **kwargs):
        if name == "rdkit.Contrib.SA_Score":
            raise ModuleNotFoundError("dependency", name="missing_dependency")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", broken)
    with pytest.raises(ModuleNotFoundError, match="dependency"):
        scoring._load_sascorer()


def test_missing_rdconfig_module_is_explicit(monkeypatch):
    original = builtins.__import__
    def block(name, *args, **kwargs):
        if name.startswith("rdkit.Contrib"):
            raise ModuleNotFoundError("blocked", name="rdkit.Contrib")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", block)
    monkeypatch.setattr(RDConfig, "RDContribDir", str(Path("does-not-exist")))
    with pytest.raises(FileNotFoundError, match="SA_Score module"):
        scoring._load_sascorer()


def test_modern_loader_full_path_called_once(tmp_path):
    path = tmp_path / "scores.pkl.gz"; path.write_bytes(b"x"); calls = []
    module = type("M", (), {"readFragmentScores": staticmethod(lambda value: calls.append(value))})()
    scoring._read_fragment_scores(module, path)
    assert calls == [str(path)]


def test_missing_asset_rejected_before_reader(monkeypatch, tmp_path):
    module = type("M", (), {"readFragmentScores": staticmethod(lambda value: (_ for _ in ()).throw(AssertionError("called")))})()
    with pytest.raises(FileNotFoundError):
        scoring._read_fragment_scores(module, tmp_path / "missing.pkl.gz")


def test_unrelated_missing_filename_does_not_retry(tmp_path):
    path = tmp_path / "scores.pkl.gz"; path.write_bytes(b"x"); calls = []
    def read(value):
        calls.append(value); raise FileNotFoundError(2, "missing", str(path) + ".other")
    module = type("M", (), {"readFragmentScores": staticmethod(read)})()
    with pytest.raises(FileNotFoundError): scoring._read_fragment_scores(module, path)
    assert len(calls) == 1
