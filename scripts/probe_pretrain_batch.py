"""Short, isolated GPU capacity probes. Never saves a training checkpoint."""

import argparse
import csv
import json
import math
import os
import signal
import shutil
import socket
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import yaml

from utils.pretrain_record import git_version


def write_json(path, value):
    path = Path(path)
    with path.with_suffix(path.suffix + '.tmp').open('w', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')
    os.replace(str(path) + '.tmp', path)


def load_experiment(path):
    with open(path, encoding='utf-8') as handle:
        config = yaml.safe_load(handle)
    batches = config['batch_sizes']
    if (not batches or any(type(size) is not int or size <= 0 for size in batches)
            or batches != sorted(set(batches))):
        raise ValueError('batch_sizes must be positive, unique, increasing integers')
    for name in ('warmup_steps', 'measured_steps', 'stress_steps', 'candidate_timeout_seconds'):
        if type(config[name]) is not int or config[name] <= 0:
            raise ValueError('%s must be a positive integer' % name)
    if not 0 < config['max_memory_fraction'] < 1:
        raise ValueError('max_memory_fraction must be between zero and one')
    if config['device'] != 'cuda:0':
        raise ValueError('Use logical cuda:0 within the allocated/explicitly selected GPU mask')
    return config


def check_dataset_files(config):
    root = Path(config.root)
    processed = root / config.path_dict.processed
    # Match Drug3DDataset's index naming, and refuse its automatic preprocessing.
    index = Path(str(processed).split('.lmdb', 1)[0] + '_molid2idx.pt')
    for path in (processed, index, root / config.split, root / config.path_dict.summary):
        if not path.is_file():
            raise FileNotFoundError('Missing processed input (no processing attempted): %s' % path)


def largest_training_indices(dataset, train_indices, summary_path, count):
    train_indices = set(train_indices)
    sizes = {}
    with open(summary_path, newline='', encoding='utf-8') as handle:
        for row in csv.DictReader(handle):
            index = dataset.molid2idx.get(int(row['mol_id']))
            if index in train_indices:
                sizes[index] = int(row['n_heavy_atoms'])
    if set(sizes) != train_indices or len(sizes) < count:
        raise ValueError('Summary must cover all training indices and the requested stress batch')
    return sorted(sizes, key=lambda index: (-sizes[index], index))[:count]


def is_cuda_oom(exc):
    message = str(exc).lower()
    return isinstance(exc, RuntimeError) and (
        'cuda out of memory' in message or 'cuda error: out of memory' in message)


def recommend(results, fraction):
    eligible = [row['batch_size'] for row in results
                if row['status'] == 'ok' and row['measured_optimizer_updates'] > 0
                and row['estimated_peak_device_bytes'] <= fraction * row['device_total_bytes']]
    return max(eligible) if eligible else None


def loss_for_batch(model, batch, node_pos):
    return model.get_loss(
        node_type=batch.node_type, node_pos=node_pos, batch_node=batch.node_type_batch,
        halfedge_type=batch.halfedge_type, halfedge_index=batch.halfedge_index,
        batch_halfedge=batch.halfedge_type_batch, num_mol=batch.num_graphs,
    )


def optimization_step(model, optimizer, scaler, batch, config):
    # Same update sequence as train_drug3d.py; diagnostics are read after the step.
    optimizer.zero_grad(set_to_none=True)
    pos_noise = torch.randn_like(batch.node_pos) * config.train.pos_noise_std
    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=config.train.use_amp):
        losses = loss_for_batch(model, batch, batch.node_pos + pos_noise)
    loss = losses['loss']
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    grad = torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
    scale_before = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    skipped = scaler.get_scale() < scale_before
    return losses, skipped, float(grad)


def run_candidate(experiment, batch_size, outdir):
    result = {'batch_size': batch_size, 'status': 'running', 'phase': 'imports'}
    write_json(outdir / 'result.json', result)
    try:
        from torch_geometric.loader import DataLoader
        from torch.utils.data import Subset
        from models.model import MDRL
        from utils.dataset import get_dataset
        from utils.misc import load_config, seed_all
        from utils.train import get_optimizer, inf_iterator
        from utils.transforms import Compose, FeaturizeMol

        torch.multiprocessing.set_sharing_strategy('file_system')
        result['phase'] = 'input_check'
        config = load_config(ROOT / experiment['train_config'])
        config.train.batch_size = batch_size
        check_dataset_files(config.dataset)
        with (outdir / 'effective_train.yml').open('x', encoding='utf-8') as handle:
            # EasyDict subclasses dict; safe_dump needs ordinary nested containers.
            yaml.safe_dump(json.loads(json.dumps(config)), handle, sort_keys=False)
        seed_all(config.train.seed)
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is required; no CPU fallback is allowed')
        device = torch.device(experiment['device'])
        torch.cuda.set_device(device)
        free, total = torch.cuda.mem_get_info(device)
        initial_used = total - free
        result.update(device_name=torch.cuda.get_device_name(device), device_total_bytes=total,
                      initial_device_used_bytes=initial_used, torch=torch.__version__,
                      cuda=torch.version.cuda, cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                      python=sys.version, hostname=socket.gethostname())
        result['phase'] = 'dataset'
        featurizer = FeaturizeMol(config.chem.atomic_numbers, config.chem.mol_bond_types,
                                 config.transform.use_mask_node, config.transform.use_mask_edge)
        dataset, subsets = get_dataset(config.dataset, transform=Compose([featurizer]))
        stress_indices = largest_training_indices(
            dataset, subsets['train'].indices,
            Path(config.dataset.root) / config.dataset.path_dict.summary, batch_size)
        loader_args = dict(follow_batch=featurizer.follow_batch, exclude_keys=featurizer.exclude_keys)
        train_iterator = inf_iterator(DataLoader(
            subsets['train'], batch_size=batch_size, shuffle=True,
            num_workers=config.train.num_workers, pin_memory=config.train.pin_memory,
            drop_last=True, **loader_args))
        stress_iterator = inf_iterator(DataLoader(
            Subset(dataset, stress_indices), batch_size=batch_size, shuffle=False,
            num_workers=0, **loader_args))
        result['train_size'] = len(subsets['train'])
        result['stress_dataset_indices'] = stress_indices
        result['phase'] = 'model'
        model = MDRL(config=config.model, num_node_types=featurizer.num_node_types,
                     num_edge_types=featurizer.num_edge_types).to(device)
        optimizer = get_optimizer(config.train.optimizer, model)
        scaler = torch.cuda.amp.GradScaler(enabled=config.train.use_amp)
        torch.cuda.reset_peak_memory_stats(device)
        measured_seconds = []
        result.update(measured_optimizer_updates=0, measured_amp_skips=0)
        max_device_used = initial_used

        def run_step(iterator, phase, step):
            nonlocal max_device_used
            result.update(phase=phase, step=step)
            write_json(outdir / 'result.json', result)
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            batch = next(iterator)
            ids = batch.mol_id
            ids = [int(value) for value in (ids.tolist() if hasattr(ids, 'tolist') else ids)]
            batch = batch.to(device)
            if phase == 'stress_validation':
                model.eval()
                with torch.no_grad(), torch.autocast(
                        device_type='cuda', dtype=torch.float16, enabled=config.train.use_amp):
                    losses = loss_for_batch(model, batch, batch.node_pos)
                skipped, grad = False, None
            else:
                model.train()
                losses, skipped, grad = optimization_step(model, optimizer, scaler, batch, config)
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - start
            values = {name: value.item() for name, value in losses.items()}
            if not all(math.isfinite(value) for value in values.values()):
                raise FloatingPointError('Nonfinite diagnostic loss: %s' % values)
            free_now, _ = torch.cuda.mem_get_info(device)
            max_device_used = max(max_device_used, total - free_now)
            event = dict(phase=phase, step=step, seconds=elapsed, losses=values,
                         mol_ids=ids, nodes=batch.node_type.numel(),
                         halfedges=batch.halfedge_type.numel(), amp_skipped=skipped,
                         grad_norm=grad if grad is not None and math.isfinite(grad) else None,
                         peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                         peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
            with (outdir / 'steps.jsonl').open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(event, allow_nan=False) + '\n')
            if phase == 'measured_train':
                measured_seconds.append(elapsed)
                result['measured_optimizer_updates'] += int(not skipped)
                result['measured_amp_skips'] += int(skipped)
            result.update(peak_allocated_bytes=event['peak_allocated_bytes'],
                          peak_reserved_bytes=event['peak_reserved_bytes'])
            write_json(outdir / 'result.json', result)
            print('batch=%d phase=%s step=%d seconds=%.3f peak_reserved=%.2fGiB' % (
                batch_size, phase, step, elapsed, event['peak_reserved_bytes'] / 2**30), flush=True)

        for phase, steps, iterator in (
                ('warmup_train', experiment['warmup_steps'], train_iterator),
                ('measured_train', experiment['measured_steps'], train_iterator),
                ('stress_train', experiment['stress_steps'], stress_iterator),
                ('stress_validation', 1, stress_iterator)):
            for step in range(1, steps + 1):
                run_step(iterator, phase, step)
        peak_reserved = torch.cuda.max_memory_reserved(device)
        result.update(
            status='ok', phase='finished', measured_steps=len(measured_seconds),
            mean_step_seconds=sum(measured_seconds) / len(measured_seconds),
            samples_per_second=batch_size * len(measured_seconds) / sum(measured_seconds),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
            peak_reserved_bytes=peak_reserved, max_observed_device_used_bytes=max_device_used,
            estimated_peak_device_bytes=max(initial_used + peak_reserved, max_device_used))
    except BaseException as exc:
        result.update(status='oom' if is_cuda_oom(exc) else (
            'interrupted' if isinstance(exc, KeyboardInterrupt) else 'error'),
            error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
        traceback.print_exc()
    write_json(outdir / 'result.json', result)
    return result


def write_summary(outdir, summary, fraction):
    results = summary['results']
    summary['recommended_batch_size'] = None if summary['status'] in ('error', 'interrupted') else recommend(
        results, fraction)
    write_json(outdir / 'summary.json', summary)
    fields = ['batch_size', 'status', 'mean_step_seconds', 'samples_per_second',
              'peak_allocated_bytes', 'peak_reserved_bytes', 'estimated_peak_device_bytes',
              'device_total_bytes', 'measured_optimizer_updates', 'measured_amp_skips', 'phase']
    with (outdir / 'summary.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)


def launch_worker(command, console, timeout):
    # A timeout/interrupt must also stop the candidate's DataLoader workers.
    with subprocess.Popen(command, cwd=ROOT, stdout=console, stderr=subprocess.STDOUT,
                          start_new_session=True) as process:
        try:
            return process.wait(timeout=timeout)
        except BaseException:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            raise


def run_experiment(experiment_path, outdir):
    experiment = load_experiment(experiment_path)
    outdir.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(experiment_path, outdir / 'experiment.yml')
    shutil.copyfile(ROOT / experiment['train_config'], outdir / 'base_train.yml')
    summary = dict(experiment_id=experiment['id'], run_id=outdir.parent.name,
                   status='running', git=git_version(ROOT), python=sys.executable,
                   pbs_job_id=os.environ.get('PBS_JOBID'), results=[],
                   recommendation_rule='largest tested passing batch with measured updates and memory headroom',
                   max_memory_fraction=experiment['max_memory_fraction'])
    write_summary(outdir, summary, experiment['max_memory_fraction'])
    try:
        for batch_size in experiment['batch_sizes']:
            candidate_dir = outdir / ('batch_%03d' % batch_size)
            candidate_dir.mkdir()
            command = [sys.executable, '-u', '-B', str(Path(__file__).resolve()),
                       '--experiment', str(Path(experiment_path).resolve()),
                       '--outdir', str(candidate_dir), '--candidate-batch', str(batch_size)]
            with (candidate_dir / 'console.log').open('x', encoding='utf-8') as console:
                try:
                    returncode = launch_worker(command, console, experiment['candidate_timeout_seconds'])
                    path = candidate_dir / 'result.json'
                    result = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {
                        'batch_size': batch_size, 'status': 'error', 'error': 'Worker produced no result'}
                    result['returncode'] = returncode
                    if result['status'] == 'running' or (result['status'] == 'ok' and returncode != 0):
                        result.update(status='error', error='Worker exited without a successful final result')
                except subprocess.TimeoutExpired:
                    path = candidate_dir / 'result.json'
                    result = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
                    result.update(batch_size=batch_size, status='timeout', error='Candidate time limit exceeded')
            write_json(candidate_dir / 'result.json', result)
            summary['results'].append(result)
            print('batch=%d status=%s' % (batch_size, result['status']), flush=True)
            if result['status'] != 'ok':
                summary['status'] = 'completed_at_oom' if result['status'] == 'oom' else 'error'
                break
            write_summary(outdir, summary, experiment['max_memory_fraction'])
        else:
            summary['status'] = 'completed_grid'
    except BaseException as exc:
        summary.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'error',
                       error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        summary['untested_batch_sizes'] = [size for size in experiment['batch_sizes']
                                           if size not in {row['batch_size'] for row in summary['results']}]
        write_summary(outdir, summary, experiment['max_memory_fraction'])
    print('Provisional recommended batch: %s' % summary['recommended_batch_size'], flush=True)
    return 0 if summary['recommended_batch_size'] is not None else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, default=ROOT / 'configs/experiments/exp001_batch_probe.yml')
    parser.add_argument('--outdir', type=Path)
    parser.add_argument('--candidate-batch', type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if sys.platform != 'linux':
        parser.error('GPU probes run on the Linux server only; local checks use unit tests')
    if not os.environ.get('CUDA_VISIBLE_DEVICES'):
        parser.error('Set CUDA_VISIBLE_DEVICES through the scheduler or an authorized explicit GPU selection')
    experiment = load_experiment(args.experiment)
    if args.candidate_batch is not None:
        if args.outdir is None or args.candidate_batch not in experiment['batch_sizes']:
            parser.error('A worker needs its output directory and a configured candidate batch')
        result = run_candidate(experiment, args.candidate_batch, args.outdir)
        return {'ok': 0, 'oom': 3}.get(result['status'], 1)
    outdir = args.outdir or Path(experiment['output_root']) / (
        experiment['id'] + '-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')) / 'results'
    return run_experiment(args.experiment, outdir)


if __name__ == '__main__':
    sys.exit(main())
