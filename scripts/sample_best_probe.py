"""Fixed-attempt unconditional sampling; does not train or replace failed draws."""

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from rdkit import Chem, RDConfig, rdBase
from rdkit.Chem import Crippen, Descriptors, Lipinski, QED

from models.model import MDRL
from utils.misc import get_logger, seed_all
from utils.pretrain_record import git_version
from utils.pretrain_resume import load_checkpoint
from utils.reconstruct import reconstruct_from_generated_with_edges
from utils.transforms import FeaturizeMol, make_data_placeholder


def write_json(path, value):
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_sa(scores_path):
    try:
        source = Path(RDConfig.RDContribDir) / 'SA_Score' / 'sascorer.py'
        spec = importlib.util.spec_from_file_location('mdrl_probe_sascorer', str(source))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        scores = Path(scores_path) if scores_path else source.with_name('fpscores.pkl.gz')
        if not scores.is_file() or not str(scores).endswith('.pkl.gz'):
            raise FileNotFoundError('Missing trusted SA fragment scores: ' + str(scores))
        module.readFragmentScores(str(scores)[:-len('.pkl.gz')])
        return module.calculateScore, {'available': True, 'source': str(source),
                                       'scores': str(scores), 'scores_sha256': sha256(scores)}
    except Exception as exc:
        return None, {'available': False, 'error': type(exc).__name__ + ': ' + str(exc)}


def summarize(rows, requested, status, **extra):
    valid = [row for row in rows if row['terminal_complete'] and row['chem_valid']]
    connected = [row for row in valid if row['connected']]
    complete = len(rows) == requested
    summary = {
        'status': status, 'requested_attempts': requested, 'recorded_attempts': len(rows),
        'unattempted': requested - len(rows), 'valid': len(valid), 'connected': len(connected),
        'valid_disconnected': len(valid) - len(connected),
        'invalid_or_generation_failed': len(rows) - len(valid),
        'valid_percent': 100.0 * len(valid) / requested if complete else None,
        'connected_percent_among_valid': 100.0 * len(connected) / len(valid) if valid else None,
        'connected_yield_percent': 100.0 * len(connected) / requested if complete else None,
        'unique_connected': len({row['smiles'] for row in connected}),
        'unique_percent_among_connected': (
            100.0 * len({row['smiles'] for row in connected}) / len(connected) if connected else None),
        'properties_population': 'valid and connected; non-missing values per property',
        'properties': {},
    }
    for key in ('qed', 'sa_raw', 'mw', 'logp', 'hbd', 'hba', 'lipinski_pass'):
        values = [float(row[key]) for row in connected if row[key] is not None]
        summary['properties'][key] = {'n': len(values),
                                      'mean': sum(values) / len(values) if values else None}
    summary.update(extra)
    return summary


def draw_grids(outdir, rows):
    from rdkit.Chem import Draw
    selected = [row for row in rows if row['chem_valid'] and row['connected']]
    paths = []
    for start in range(0, len(selected), 20):
        page = selected[start:start + 20]
        mols = [Chem.MolFromSmiles(row['smiles']) for row in page]
        labels = [row['molecule_id'] for row in page]
        image = Draw.MolsToGridImage(mols, molsPerRow=4, subImgSize=(280, 220), legends=labels)
        name = 'structures_%03d.png' % (start // 20 + 1)
        image.save(str(outdir / name))
        paths.append(name)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    if type(config['attempts']) is not int or config['attempts'] <= 0 or config['batch_size'] != 1:
        raise ValueError('Use positive attempts and inference batch_size=1')
    outdir = Path(config['output'])
    outdir.mkdir(parents=True, exist_ok=False)
    (outdir / 'raw').mkdir()
    (outdir / 'sdf').mkdir()
    logger = get_logger('best_probe', str(outdir))
    rows = []
    write_json(outdir / 'summary.json', summarize(rows, config['attempts'], 'setup'))
    try:
        actual_git = git_version(ROOT)
        if actual_git.get('commit') != config['git']['commit'] or actual_git.get('dirty'):
            raise ValueError('Repository revision/status differs from prepared experiment')
        if sha256(config['checkpoint']) != config['checkpoint_sha256']:
            raise ValueError('Checkpoint changed after experiment preparation')
        checkpoint = load_checkpoint(config['checkpoint'])
        if checkpoint.get('iteration') != config['expected_iteration']:
            raise ValueError('Unexpected source checkpoint iteration')
        bad = [key for key, tensor in checkpoint['model'].items()
               if torch.is_tensor(tensor) and tensor.is_floating_point()
               and not torch.isfinite(tensor).all().item()]
        if bad:
            raise ValueError('Nonfinite model tensors: ' + repr(bad[:8]))
        saved = checkpoint['config']
        featurizer = FeaturizeMol(saved.chem.atomic_numbers, saved.chem.mol_bond_types,
                                 saved.transform.use_mask_node, saved.transform.use_mask_edge)
        if saved.model.name != 'diffusion':
            raise ValueError('Expected the original diffusion model')
        seed_all(config['seed'])
        model = MDRL(config=saved.model, num_node_types=featurizer.num_node_types,
                     num_edge_types=featurizer.num_edge_types).to(config['device'])
        model.load_state_dict(checkpoint['model'], strict=True)
        model.eval()
        write_json(outdir / 'source_train_config.json', saved)
        metadata = {
            'config': config, 'checkpoint_iteration': checkpoint['iteration'],
            'source_run_id': checkpoint.get('run_id'),
            'source_skipped_iterations': checkpoint.get('skipped_iterations'),
            'source_validation_losses': checkpoint.get('validation_losses'),
            'git': actual_git, 'diffusion_timesteps': model.num_timesteps,
            'torch': torch.__version__, 'rdkit': rdBase.rdkitVersion,
            'device': config['device'], 'inference_precision': 'original sampler; no autocast',
            'guidance': None, 'bond_predictor': None,
            'node_count_distribution': 'original make_data_placeholder(max_size=None)',
            'seed_rule': 'seed_all(seed) once; not derived from output directory',
            'validity_rule': 'finite prediction, nonempty RDKit-sanitized original MDRL reconstruction',
            'reconstruction_repairs': 'original aromatic/valence repair retained; not raw-graph validity',
            'lipinski_rule': 'MolWt<=500, MolLogP<=5, HBD<=5, HBA<=10',
            'scope': 'exploratory unconditional generation; no docking or activity claim',
        }
        sa_function, sa_metadata = load_sa(config.get('sa_scores'))
        metadata['sa'] = sa_metadata
        write_json(outdir / 'metadata.json', metadata)
        del checkpoint
        logger.info('Checkpoint iteration %d; attempts=%d; batch=1; timesteps=%d',
                    config['expected_iteration'], config['attempts'], model.num_timesteps)
        properties = {'qed': QED.qed, 'mw': Descriptors.MolWt, 'logp': Crippen.MolLogP,
                      'hbd': Lipinski.NumHDonors, 'hba': Lipinski.NumHAcceptors}
        if sa_function is not None:
            properties['sa_raw'] = sa_function
        with (outdir / 'molecules.jsonl').open('x', encoding='utf-8') as handle:
            for index in range(config['attempts']):
                row = dict.fromkeys(('smiles', 'sdf', 'raw_prediction', 'qed', 'sa_raw', 'mw',
                                     'logp', 'hbd', 'hba', 'lipinski_pass', 'error'))
                row.update(index=index, molecule_id='mol_%03d' % index,
                           terminal_complete=False, chem_valid=False, connected=False,
                           status='generation_failed', metric_errors={})
                output = holder = None
                stage = 'generation'
                try:
                    holder = make_data_placeholder(n_graphs=1, device=config['device'], max_size=None)
                    output = model.sample(n_graphs=1, batch_node=holder['batch_node'],
                                          halfedge_index=holder['halfedge_index'],
                                          batch_halfedge=holder['batch_halfedge'],
                                          bond_predictor=None, guidance=None)
                    row['terminal_complete'] = True
                    predictions = [tensor.detach().cpu().numpy() for tensor in output['pred']]
                    edge_index = holder['halfedge_index'].cpu().numpy()
                    row['raw_prediction'] = 'raw/%s.npz' % row['molecule_id']
                    # I/O failures are fatal, not chemical-generation failures.
                    stage = 'artifact_io'
                    np.savez_compressed(outdir / row['raw_prediction'], pred_node=predictions[0],
                                        pred_pos=predictions[1], pred_halfedge=predictions[2],
                                        halfedge_index=edge_index)
                    stage = 'prediction'
                    if not all(np.isfinite(value).all() for value in predictions):
                        raise ValueError('Nonfinite final prediction')
                    stage = 'reconstruction'
                    info = featurizer.decode_output(*predictions, halfedge_index=edge_index)
                    mol = reconstruct_from_generated_with_edges(info, add_edge=None)
                    Chem.SanitizeMol(mol)
                    if mol.GetNumAtoms() == 0:
                        raise ValueError('Empty molecule')
                    row['smiles'] = Chem.MolToSmiles(mol, canonical=True)
                    row['chem_valid'] = True
                    row['connected'] = len(Chem.GetMolFrags(mol)) == 1
                    row['status'] = 'valid_connected' if row['connected'] else 'valid_disconnected'
                    row['sdf'] = 'sdf/%s.sdf' % row['molecule_id']
                    mol.SetProp('_Name', row['molecule_id'])
                    stage = 'artifact_io'
                    Chem.MolToMolFile(mol, str(outdir / row['sdf']))
                    if row['connected']:
                        for key, function in properties.items():
                            try:
                                value = float(function(mol))
                                if not math.isfinite(value):
                                    raise ValueError('Nonfinite property')
                                row[key] = value
                            except Exception as exc:
                                row['metric_errors'][key] = type(exc).__name__ + ': ' + str(exc)
                        if all(row[key] is not None for key in ('mw', 'logp', 'hbd', 'hba')):
                            row['lipinski_pass'] = (row['mw'] <= 500 and row['logp'] <= 5
                                                    and row['hbd'] <= 5 and row['hba'] <= 10)
                except Exception as exc:
                    if stage == 'artifact_io':
                        raise
                    row['status'] = stage + '_failed'
                    row['error'] = type(exc).__name__ + ': ' + str(exc)
                    logger.exception('Attempt %d failed at %s; no replacement draw', index, stage)
                    if isinstance(exc, RuntimeError) and 'out of memory' in str(exc).lower():
                        torch.cuda.empty_cache()
                finally:
                    del output, holder
                rows.append(row)
                handle.write(json.dumps(row, allow_nan=False) + '\n')
                handle.flush()
                write_json(outdir / 'summary.json', summarize(rows, config['attempts'], 'running'))
                logger.info('Attempt %d/%d | %s | %s', index + 1, config['attempts'],
                            row['status'], row['smiles'])
        fields = [key for key in rows[0] if key != 'metric_errors']
        with (outdir / 'molecules.csv').open('x', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(rows)
        visualizations = {}
        try:
            visualizations['files'] = draw_grids(outdir, rows)
        except Exception as exc:
            visualizations['error'] = type(exc).__name__ + ': ' + str(exc)
            logger.exception('Structure grids unavailable; molecular records retained')
        summary = summarize(rows, config['attempts'], 'completed', sa=sa_metadata,
                            visualizations=visualizations)
        write_json(outdir / 'summary.json', summary)
        logger.info('Summary: %s', json.dumps(summary, allow_nan=False))
        return 0
    except BaseException:
        write_json(outdir / 'summary.json', summarize(rows, config['attempts'], 'failed',
                                                     error=traceback.format_exc()))
        raise


if __name__ == '__main__':
    sys.exit(main())
