"""CPU-only scoring workers; use spawn so workers never inherit CUDA state."""

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import math
import multiprocessing
import os

from RL_utils.task_config import append_record
from RL_utils.run_state import read_records


_CONFIG = None
_SCORERS = None


def initialize_worker(config, scoring_csv):
    global _CONFIG, _SCORERS
    threads = str(config['scoring'].get('predictor_threads', 1))
    for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[variable] = threads
    from RL_utils.score_fuc import build_scoring_function

    _CONFIG = config
    _SCORERS, _ = build_scoring_function(scoring_csv, config['fscores'], return_individual=True)
    for scorer in _SCORERS.values():
        if hasattr(scorer, 'rfr'):
            scorer.rfr.set_params(n_jobs=int(threads), device='cpu')


def score_one(index, smiles, molecule, directory):
    from pathlib import Path
    from RL_utils.docking import dock_molecule

    raw = {}
    stage = 'properties'
    try:
        raw = {name: float(scorer.raw_score(smiles)) for name, scorer in _SCORERS.items()}
        components = {name: float(_SCORERS[name].modify_score(value)) for name, value in raw.items()}
        if not all(math.isfinite(v) for v in list(raw.values()) + list(components.values())):
            raise ValueError('Nonfinite property or ligand-efficiency score')
        stage = 'docking'
        docking = dock_molecule(molecule, _CONFIG['targets'],
                                Path(directory) / 'docking' / ('mol_%06d' % index), _CONFIG['docking'])
        property_mean = sum(components.values()) / len(components)
        docking_mean = sum(item['normalized_score'] for item in docking) / len(docking)
        reward = (property_mean + docking_mean) / 2
        if not math.isfinite(reward):
            raise ValueError('Nonfinite combined reward')
        return dict(molecule_index=index, smiles=smiles, raw_scores=raw,
                    normalized_scores=components, docking=docking, reward=reward)
    except Exception as exc:
        return dict(molecule_index=index, smiles=smiles, stage=stage, error=repr(exc),
                    raw_scores={name: repr(value) for name, value in raw.items()})


def create_pool(config, scoring_csv):
    return ProcessPoolExecutor(max_workers=config['docking']['workers'],
                               mp_context=multiprocessing.get_context('spawn'),
                               initializer=initialize_worker, initargs=(config, str(scoring_csv)))


def score_population(population, pool, workers, round_dir, resume=False):
    cached = {}
    if resume:
        for record in read_records(round_dir / 'scores.jsonl'):
            index = record['molecule_index']
            if (type(index) is not int or not 0 <= index < len(population) or index in cached or
                    record['smiles'] != population[index]['smiles'] or
                    not math.isfinite(record['reward'])):
                raise ValueError('Saved scoring records do not match this population')
            cached[index] = record['reward']
            population[index]['score'] = record['reward']
    elif (round_dir / 'scores.jsonl').exists():
        raise FileExistsError('Scoring records already exist; use --resume')
    iterator = iter((index, molecule) for index, molecule in enumerate(population) if index not in cached)
    pending = {}
    completed = len(cached)

    def submit_one():
        item = next(iterator, None)
        if item is None:
            return
        index, molecule = item
        future = pool.submit(score_one, index, molecule['smiles'], molecule['rdmol'], str(round_dir))
        pending[future] = index

    # Bound queued molecules and stop adding work after the first failure.
    try:
        for _ in range(2 * workers):
            submit_one()
        while pending:
            finished, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in finished:
                index = pending.pop(future)
                try:
                    result = future.result()
                except Exception as exc:
                    result = dict(molecule_index=index, smiles=population[index]['smiles'], error=repr(exc))
                if 'error' in result:
                    append_record(round_dir / 'failures.jsonl', result)
                    raise RuntimeError('Scoring stopped at round %s molecule %d: %s' %
                                       (round_dir.name, index, result['error']))
                population[index]['score'] = result['reward']
                append_record(round_dir / 'scores.jsonl', result)
                completed += 1
                if completed % 50 == 0 or completed == len(population):
                    print('RL|Scoring %s | %d/%d | workers=%d' %
                          (round_dir.name, completed, len(population), workers), flush=True)
                submit_one()
    finally:
        for future in pending:
            future.cancel()
    # Preserve the original input order for tied rewards, not completion order.
    return sorted(population, key=lambda molecule: molecule['score'], reverse=True)
