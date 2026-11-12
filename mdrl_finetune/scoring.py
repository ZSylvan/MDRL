"""CPU scoring isolated from the original MDRL RL reward and worker code."""
from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import math
import multiprocessing
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import numpy as np


def _load_sascorer():
    """Load RDKit's canonical SA_Score module across old/new layouts."""
    try:
        from rdkit.Contrib.SA_Score import sascorer
        return sascorer
    except ModuleNotFoundError as exc:
        # Only the absent package layout is eligible for fallback.  Missing
        # transitive dependencies inside an installed module must propagate.
        if exc.name not in {"rdkit.Contrib", "rdkit.Contrib.SA_Score", "rdkit.Contrib.SA_Score.sascorer"}:
            raise
        from rdkit import RDConfig
        import importlib.util
        source = Path(RDConfig.RDContribDir) / "SA_Score" / "sascorer.py"
        if not source.is_file():
            raise FileNotFoundError("RDKit SA_Score module not found: " + str(source)) from exc
        spec = importlib.util.spec_from_file_location("mdrl_finetune_sascorer", str(source))
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module


def _read_fragment_scores(sascorer, scores_path):
    path = Path(scores_path)
    if not path.is_file():
        raise FileNotFoundError("Missing trusted SA fragment scores: " + str(path))
    try:
        sascorer.readFragmentScores(str(path))
    except FileNotFoundError as exc:
        expected = str(path) + ".pkl.gz"
        if not str(path).endswith(".pkl.gz") or str(getattr(exc, "filename", "")) != expected:
            raise
        # RDKit 2022's loader appends .pkl.gz; newer versions accept the full path.
        sascorer.readFragmentScores(str(path)[:-len(".pkl.gz")])


def capped_gaussian(value, mu, sigma, minimize=True):
    if not all(math.isfinite(float(x)) for x in (value, mu, sigma)) or sigma <= 0:
        raise ValueError("invalid Gaussian score")
    clipped = max(value, mu) if minimize else min(value, mu)
    return math.exp(-0.5 * ((clipped - mu) / sigma) ** 2)


def ligand_efficiency(prediction, hydrogen_count):
    if hydrogen_count <= 0:
        raise ValueError("ligand-efficiency denominator has no hydrogens")
    return 1.4 * float(prediction) / int(hydrogen_count)


def preflight_scoring(config, stage="sft"):
    from rdkit import Chem
    sascorer = _load_sascorer()
    from openbabel import openbabel

    scoring = config["scoring"]
    executable = scoring["executable"]
    if not (Path(executable).is_file() or shutil.which(executable)):
        raise FileNotFoundError("Vina executable missing: " + executable)
    paths = [scoring["fscores"]] + [target["receptor"] for target in config["targets"]]
    if stage == "sft":
        paths += [target["le_model"] for target in config["targets"]]
    for path in paths:
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    conversion = openbabel.OBConversion()
    if not conversion.SetInAndOutFormats("sdf", "pdbqt"):
        raise RuntimeError("OpenBabel SDF/PDBQT plugins are unavailable")
    _read_fragment_scores(sascorer, Path(scoring["fscores"]).resolve())
    return True


def _run_vina(ligand, target, settings, seed, log):
    from RL_utils.docking import parse_affinity

    command = [settings["executable"], "--ligand", str(ligand), "--receptor", target["receptor"]]
    for axis, center, size in zip("xyz", target["center"], target["box_size"]):
        command += ["--center_" + axis, str(center), "--size_" + axis, str(size)]
    command += ["--cpu", str(settings["cpu"]), "--seed", str(seed),
                "--exhaustiveness", str(settings["exhaustiveness"]),
                "--num_modes", str(settings["num_modes"]),
                "--out", str(log.with_suffix(".pdbqt"))]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False,
                                timeout=settings["timeout_s"])
    except subprocess.TimeoutExpired as exc:
        log.write_text("TIMEOUT\n" + str(exc), encoding="utf-8")
        raise RuntimeError("Vina timed out for " + target["id"]) from exc
    log.write_text(result.stdout + "\nSTDERR\n" + result.stderr, encoding="utf-8")
    if result.returncode:
        raise RuntimeError("Vina failed for %s, exit %d" % (target["id"], result.returncode))
    return parse_affinity(result.stdout)


class _ScorerCore:
    def __init__(self, config, stage):
        sascorer = _load_sascorer()

        self.config, self.stage = config, stage
        _read_fragment_scores(sascorer, Path(config["scoring"]["fscores"]).resolve())
        self.sascorer = sascorer
        self.le_models = []
        if stage == "sft":
            for target in config["targets"]:
                with Path(target["le_model"]).open("rb") as handle:
                    model = pickle.load(handle)  # Trusted project assets only.
                if not callable(getattr(model, "predict", None)):
                    raise ValueError("LE model has no predict interface")
                # Match the source scorer's CPU prediction behavior.
                if hasattr(model, "set_params"):
                    model.set_params(n_jobs=1, device="cpu")
                self.le_models.append(model)

    def score(self, molecule, directory):
        from rdkit import Chem
        from rdkit.Chem import AllChem, QED
        sascorer = self.sascorer
        from RL_utils.docking import sdf_to_pdbqt

        try:
            directory = Path(directory)
            directory.mkdir(parents=True, exist_ok=False)
            sdf, ligand = directory / "ligand.sdf", directory / "ligand.pdbqt"
            Chem.MolToMolFile(molecule, str(sdf))
            sdf_to_pdbqt(sdf, ligand)
            docking = []
            for target in self.config["targets"]:
                affinity = _run_vina(ligand, target, self.config["scoring"],
                                     self.config["seed"], directory / (target["id"] + ".log"))
                docking.append(dict(target=target["id"], affinity=affinity,
                                    normalized_score=capped_gaussian(affinity, target["mu"], target["sigma"])))
            docking_mean = sum(item["normalized_score"] for item in docking) / 2
            heavy = Chem.RemoveHs(Chem.Mol(molecule))
            qed, sa = float(QED.qed(heavy)), float(sascorer.calculateScore(heavy))
            if not all(math.isfinite(x) for x in (qed, sa)):
                raise ValueError("non-finite property score")
            result = dict(docking=docking, qed=qed, sa=sa)
            if self.stage == "sft":
                hydrogen_count = Chem.AddHs(Chem.Mol(heavy)).GetNumAtoms() - heavy.GetNumHeavyAtoms()
                fp = AllChem.GetMorganFingerprintAsBitVect(heavy, 2, nBits=2048)
                features = np.asarray([list(fp)], dtype=np.float32)
                efficiency = [ligand_efficiency(float(np.asarray(model.predict(features)).reshape(-1)[0]),
                                                hydrogen_count) for model in self.le_models]
                properties = [capped_gaussian(qed, 0.75, 0.1, False), capped_gaussian(sa, 3.0, 0.5)]
                properties += [capped_gaussian(value, target["le_mu"], target["le_sigma"], False)
                               for value, target in zip(efficiency, self.config["targets"])]
                result.update(le=efficiency, property_components=properties,
                              reward=0.5 * sum(properties) / 4 + 0.5 * docking_mean)
            else:
                result["reward"] = docking_mean
            return dict(result, failed=False)
        except Exception as exc:
            return dict(reward=-1.0, failed=True, error=type(exc).__name__ + ": " + str(exc))


_CORE = None


def _initialize(config, stage):
    global _CORE
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[variable] = "1"
    _CORE = _ScorerCore(config, stage)


def _score_task(molecule, directory):
    return _CORE.score(molecule, directory)


class ScoreEngine:
    """Bounded spawn workers; stable input order regardless of completion order."""
    def __init__(self, config, stage="sft"):
        preflight_scoring(config, stage)
        self.config, self.stage = config, stage
        self.core = _ScorerCore(config, stage)  # Load/validate assets before the run is created.
        self.executor = None

    def close(self):
        if self.executor is not None:
            self.executor.shutdown(wait=True)
            self.executor = None

    def score_many(self, rows, round_dir):
        rows = list(rows)
        work = []
        for index, row in enumerate(rows):
            if not row.get("valid") or row.get("rdmol") is None:
                row.update(reward=-1.0, failed=True)
            else:
                work.append((index, row["rdmol"]))
        workers = self.config["scoring"]["workers"]
        directory = Path(round_dir) / "scoring"
        if workers == 1:
            for index, molecule in work:
                rows[index].update(self.core.score(molecule, directory / ("mol_%06d" % index)))
            return rows
        if self.executor is None and work:
            self.executor = ProcessPoolExecutor(
                max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                initializer=_initialize, initargs=(self.config, self.stage))
        iterator, pending = iter(work), {}

        def submit_one():
            item = next(iterator, None)
            if item is not None:
                index, molecule = item
                pending[self.executor.submit(_score_task, molecule, str(directory / ("mol_%06d" % index)))] = index

        for _ in range(min(len(work), 2 * workers)):
            submit_one()
        while pending:
            finished, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in finished:
                index = pending.pop(future)
                rows[index].update(future.result())  # Broken pools/dependencies abort, not fake invalids.
                submit_one()
        return rows
