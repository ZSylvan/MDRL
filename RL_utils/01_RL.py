"""MDRL's population-based dual-target fine-tuning entry point."""

import argparse
import copy
import csv
import json
import os
from pathlib import Path
import shutil
import sys
import time

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from RL_utils.task_config import append_record, load_task_config, write_scoring_csv
from RL_utils.run_state import find_training_checkpoint, load_state, save_state, write_json


def save_population(population, directory):
    from rdkit import Chem

    directory = Path(directory)
    if (directory / 'manifest.json').exists():
        raise FileExistsError('Population is already complete: ' + str(directory))
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'mol_summary.csv.tmp').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['', 'mol_id', 'smiles'])
        for index, molecule in enumerate(population):
            writer.writerow([index, index, molecule['smiles']])
            temporary = directory / ('%d.sdf.tmp' % index)
            Chem.MolToMolFile(molecule['rdmol'], str(temporary))
            os.replace(temporary, directory / ('%d.sdf' % index))
    os.replace(directory / 'mol_summary.csv.tmp', directory / 'mol_summary.csv')
    save_state(directory / 'population.pt', population)
    write_json(directory / 'manifest.json', dict(count=len(population), duplicates_retained=True))


def load_population(directory):
    directory = Path(directory)
    metadata = json.loads((directory / 'manifest.json').read_text())
    population = load_state(directory / 'population.pt')
    if len(population) != metadata['count']:
        raise ValueError('Saved population count differs from manifest: ' + str(directory))
    return population


def _resume_signature(config):
    config = copy.deepcopy(config)
    config.pop('device', None)
    for key in ('workers', 'cpu'):
        config['docking'].pop(key, None)
    config['scoring'].pop('predictor_threads', None)
    return config


def _write_round_summaries(output):
    records = [json.loads(path.read_text()) for path in sorted(output.glob('round_*/round.json'))]
    temporary = output / 'rounds.jsonl.tmp'
    with temporary.open('w') as handle:
        for record in records:
            if record.get('status') == 'completed':
                handle.write(json.dumps(record, allow_nan=False) + '\n')
    os.replace(temporary, output / 'rounds.jsonl')


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


def run(config, source, resume=False):
    import numpy as np
    import torch
    import yaml
    from RL_utils.scoring_workers import create_pool, score_population
    from RL_utils.sample_drug3d_forRL import sample
    from RL_utils.train_drug3d_forRL import train_forRL

    output = Path(config['output_dir'])
    initial_snapshot = output / ('pretrained_%d.pt' % config['expected_pretrain_iteration'])
    train_path = output / 'train.yml'
    sample_path = output / 'sample.yml'
    scoring_path = output / 'scoring_definition.csv'
    if resume:
        version = json.loads((output / 'run_state.json').read_text())
        if version.get('format_version') != 1:
            raise ValueError('This output does not support the current RL resume format')
        saved_config = json.loads((output / 'task_resolved.json').read_text())
        if _resume_signature(config) != _resume_signature(saved_config):
            raise ValueError('Resume task differs from the saved task; do not change training or sampling settings')
        for path in (initial_snapshot, train_path, sample_path, scoring_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        append_record(output / 'resumes.jsonl', dict(device=config['device'], docking=config['docking']))
    else:
        # Fine-tuning initializes the model, not pretraining optimizer/progress.
        checkpoint = load_state(config['initial_checkpoint'])
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
        output.mkdir(parents=True, exist_ok=False)
        save_state(initial_snapshot, checkpoint)
        del checkpoint
        shutil.copyfile(source, output / 'task_source.yml')
        write_json(output / 'task_resolved.json', config)
        _effective_config(config['train_config'], 'train', config.get('train_overrides', {}), config['seed'], train_path)
        _effective_config(config['sample_config'], 'sample', config.get('sample_overrides', {}), config['seed'], sample_path)
        write_scoring_csv(config, scoring_path)
        write_json(output / 'run_state.json', dict(format_version=1))
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
            round_dir.mkdir(exist_ok=resume)
            record_path = round_dir / 'round.json'
            if record_path.exists():
                round_record = json.loads(record_path.read_text())
                if round_record.get('status') == 'completed':
                    model_ckpt = round_record['checkpoint']
                    previous_pool = Path(round_record['population'])
                    population = load_population(previous_pool)
                    continue
            else:
                round_record = dict(round=epoch, input_checkpoint=model_ckpt,
                                    population_source=str(previous_pool) if previous_pool else None,
                                    status='started')
                write_json(record_path, round_record)
            if epoch and 'checkpoint' not in round_record:
                training_pool = round_dir / 'training_population'
                if not (training_pool / 'manifest.json').exists():
                    save_population(population, training_pool)
                round_record['training_population'] = str(training_pool)
                round_record['training_population_count'] = len(population)
                write_json(record_path, round_record)
                recovery = find_training_checkpoint(round_dir / 'training', epoch) if resume else None
                trained = train_forRL(str(training_pool), epoch, model_ckpt,
                                      config_path=str(train_path), device=config['device'],
                                      logdir=str(round_dir / 'training'), resume_from=recovery)
                final = round_dir / 'checkpoints' / 'final.pt'
                final.parent.mkdir(exist_ok=True)
                shutil.copyfile(trained, str(final) + '.tmp')
                os.replace(str(final) + '.tmp', final)
                round_record['checkpoint'] = str(final)
                write_json(record_path, round_record)
            model_ckpt = round_record.get('checkpoint', model_ckpt)
            round_record['sampling_checkpoint'] = model_ckpt
            write_json(record_path, round_record)
            requested = config['rl']['new_num_mols_per_epoch'] if epoch else config['rl']['initial_num_mols']
            generated_path = round_dir / 'generated'
            if (generated_path / 'manifest.json').exists():
                generated = load_population(generated_path)
            else:
                generated = sample(model_ckpt, requested, epoch, config_path=str(sample_path),
                                   device=config['device'], outdir=str(round_dir / 'sampling'), resume=resume)
            if len(generated) < requested:
                raise RuntimeError('Sampling produced %d/%d molecules' % (len(generated), requested))
            if not (generated_path / 'manifest.json').exists():
                save_population(generated, generated_path)
            start_scoring = time.perf_counter()
            population = score_population(population + generated, pool, config['docking']['workers'], round_dir,
                                          **({'resume': True} if resume else {}))
            scoring_seconds = time.perf_counter() - start_scoring
            scored_count = len(population)
            if epoch:
                population = population[:config['rl']['top_k']]
            previous_pool = round_dir / 'population'
            if not (previous_pool / 'manifest.json').exists():
                save_population(population, previous_pool)
            round_record.update(
                round=epoch, checkpoint=model_ckpt, generated=len(generated),
                generated_population=str(generated_path), requested=requested,
                scored=scored_count, scoring_seconds=scoring_seconds,
                selected=len(population), mean_reward=sum(m['score'] for m in population) / len(population),
                population=str(previous_pool), status='completed')
            write_json(record_path, round_record)
            _write_round_summaries(output)
            print('RL|Completed round %d | selected=%d | checkpoint=%s' % (epoch, len(population), model_ckpt), flush=True)
        _write_round_summaries(output)
        write_json(output / 'completed.json', dict(checkpoint=model_ckpt, last_round=epoch))
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
    parser.add_argument('--resume', action='store_true', help='Continue this task in its existing output directory')
    parser.add_argument('--resume-if-exists', action='store_true', help='Resume a saved run, otherwise start a new run')
    parser.add_argument('--check-config', action='store_true', help='Read inputs only; no models, sampling, or docking')
    args = parser.parse_args(argv)
    config = load_task_config(args.config, check_files=not (args.resume or args.resume_if_exists))
    if args.device:
        config['device'] = args.device
    if args.output:
        config['output_dir'] = str(Path(args.output).resolve())
    resume = args.resume or (args.resume_if_exists and (Path(config['output_dir']) / 'run_state.json').is_file())
    if args.resume_if_exists and not resume:
        load_task_config(args.config)
    if args.check_config:
        print(json.dumps(config, indent=2, allow_nan=False))
        return 0
    run(config, Path(args.config).resolve(), resume=resume)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
