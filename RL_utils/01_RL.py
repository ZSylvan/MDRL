"""MDRL's population-based dual-target fine-tuning entry point."""

import argparse
import csv
import json
from pathlib import Path
import shutil
import sys
import time

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from RL_utils.task_config import append_record, load_task_config, write_scoring_csv


def save_population(population, directory):
    from rdkit import Chem

    directory.mkdir(parents=True, exist_ok=False)
    with (directory / 'mol_summary.csv').open('x', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['', 'mol_id', 'smiles'])
        for index, molecule in enumerate(population):
            writer.writerow([index, index, molecule['smiles']])
            Chem.MolToMolFile(molecule['rdmol'], str(directory / ('%d.sdf' % index)))


def _effective_config(source, section, overrides, seed, destination):
    import yaml

    with Path(source).open() as handle:
        config = yaml.safe_load(handle)
    def merge(target, updates):
        for key, value in updates.items():
            if key not in target:
                raise ValueError('Unknown %s override: %s' % (section, key))
            if isinstance(value, dict) and isinstance(target[key], dict):
                merge(target[key], value)
            else:
                target[key] = value
    merge(config[section], overrides)
    config[section]['seed'] = seed
    positive = ('batch_size', 'max_iters', 'val_freq') if section == 'train' else ('batch_size',)
    for key in positive:
        if type(config[section][key]) is not int or config[section][key] <= 0:
            raise ValueError('%s.%s must be a positive integer' % (section, key))
    if section == 'train' and (type(config[section]['num_workers']) is not int or config[section]['num_workers'] < 0):
        raise ValueError('train.num_workers must be a nonnegative integer')
    with destination.open('x') as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    return config


def run(config, source):
    import numpy as np
    import torch
    import yaml
    from RL_utils.scoring_workers import create_pool, score_population
    from RL_utils.sample_drug3d_forRL import sample
    from RL_utils.train_drug3d_forRL import train_forRL

    # Fine-tuning initializes the model, not pretraining optimizer/progress.
    checkpoint = torch.load(config['initial_checkpoint'], map_location='cpu')
    if checkpoint.get('iteration') != config['expected_pretrain_iteration']:
        raise ValueError('Unexpected pretrained checkpoint iteration: %s' % checkpoint.get('iteration'))
    if checkpoint.get('diagnostic_only'):
        raise ValueError('A diagnostic snapshot is not a pretrained checkpoint')
    for name, value in checkpoint['model'].items():
        if not bool(torch.isfinite(value).all()):
            raise ValueError('Nonfinite pretrained parameter: ' + name)
    with Path(config['train_config']).open() as handle:
        training = yaml.safe_load(handle)
    for section in ('model', 'chem', 'transform'):
        if checkpoint['config'][section] != training[section]:
            raise ValueError('Pretraining and fine-tuning configs differ: ' + section)
    output = Path(config['output_dir'])
    output.mkdir(parents=True, exist_ok=False)
    # Pin the already-validated state, not a mutable source file read again later.
    initial_snapshot = output / ('pretrained_%d.pt' % config['expected_pretrain_iteration'])
    torch.save(checkpoint, initial_snapshot)
    del checkpoint
    shutil.copyfile(source, output / 'task_source.yml')
    (output / 'task_resolved.json').write_text(json.dumps(config, indent=2, allow_nan=False))
    train_path = output / 'train.yml'
    sample_path = output / 'sample.yml'
    _effective_config(config['train_config'], 'train', config.get('train_overrides', {}), config['seed'], train_path)
    _effective_config(config['sample_config'], 'sample', config.get('sample_overrides', {}), config['seed'], sample_path)
    scoring_path = output / 'scoring_definition.csv'
    write_scoring_csv(config, scoring_path)
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    device = torch.device(config['device'])
    if device.type == 'cuda':
        torch.cuda.set_device(device.index or 0)
        torch.cuda.manual_seed(config['seed'])
    model_ckpt = str(initial_snapshot)
    population = []
    previous_pool = None
    pool = create_pool(config, scoring_path)
    try:
        # Preserve range(1, n_epochs): 50 means 49 model-update rounds.
        for epoch in range(config['rl']['n_epochs']):
            round_dir = output / ('round_%03d' % epoch)
            round_dir.mkdir()
            round_record = dict(round=epoch, input_checkpoint=model_ckpt,
                                population_source=str(previous_pool) if previous_pool else None)
            (round_dir / 'round.json').write_text(json.dumps(round_record, indent=2))
            if epoch:
                model_ckpt = train_forRL(str(previous_pool), epoch, model_ckpt,
                                         config_path=str(train_path), device=config['device'],
                                         logdir=str(round_dir / 'training'))
            round_record['sampling_checkpoint'] = model_ckpt
            (round_dir / 'round.json').write_text(json.dumps(round_record, indent=2))
            requested = config['rl']['new_num_mols_per_epoch'] if epoch else config['rl']['initial_num_mols']
            generated = sample(model_ckpt, requested, epoch, config_path=str(sample_path),
                               device=config['device'], outdir=str(round_dir / 'sampling'))
            if len(generated) < requested:
                raise RuntimeError('Sampling produced %d/%d molecules' % (len(generated), requested))
            start_scoring = time.perf_counter()
            population = score_population(population + generated, pool, config['docking']['workers'], round_dir)
            scoring_seconds = time.perf_counter() - start_scoring
            scored_count = len(population)
            if epoch:
                population = population[:config['rl']['top_k']]
            previous_pool = round_dir / 'population'
            save_population(population, previous_pool)
            append_record(output / 'rounds.jsonl', dict(
                round=epoch, checkpoint=model_ckpt, generated=len(generated),
                scored=scored_count, scoring_seconds=scoring_seconds,
                selected=len(population), mean_reward=sum(m['score'] for m in population) / len(population),
                population=str(previous_pool)))
            print('RL|Completed round %d | selected=%d | checkpoint=%s' % (epoch, len(population), model_ckpt), flush=True)
        (output / 'completed.json').write_text(json.dumps(dict(checkpoint=model_ckpt, last_round=epoch), indent=2))
    except BaseException as exc:
        append_record(output / 'failures.jsonl', dict(error=repr(exc)))
        raise
    finally:
        pool.shutdown(wait=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, help='Dual-target task YAML')
    parser.add_argument('--device', help='Override the logical Torch device')
    parser.add_argument('--output', help='Override the new output directory')
    parser.add_argument('--check-config', action='store_true', help='Read inputs only; no models, sampling, or docking')
    args = parser.parse_args(argv)
    config = load_task_config(args.config)
    if args.device:
        config['device'] = args.device
    if args.output:
        config['output_dir'] = str(Path(args.output).resolve())
    if args.check_config:
        print(json.dumps(config, indent=2, allow_nan=False))
        return 0
    run(config, Path(args.config).resolve())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
