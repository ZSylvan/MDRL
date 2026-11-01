import sys
import os
import shutil
from types import SimpleNamespace
sys.path.append('.')
import time

from tqdm import tqdm
import torch
import numpy as np
torch.multiprocessing.set_sharing_strategy('file_system')
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import random_split, Subset
import distutils.version
import torch.utils.tensorboard
from torch_geometric.loader import DataLoader

from models.model import MDRL
from RL_utils.dataset_forRL import get_dataset
from utils.transforms import FeaturizeMol, Compose
from utils.misc import *
from utils.train import *
from RL_utils.task_config import append_record
from RL_utils.run_state import (
    capture_rng, load_state, restore_training_components, save_state, validate_round_checkpoint,
)
from utils.pretrain_checks import (
    NonfiniteTrainingError, cpu_tree, handle_amp_gradient,
    require_amp_scale, require_finite, validate_resume_numerics,
)


def capture_training_rng(device):
    return capture_rng(device)

def train_forRL(root, epoch_now, model_ckpt, *, config_path, device, logdir, resume_from=None):
    args = SimpleNamespace(config=config_path, device=device, logdir=logdir)

    # Load configs
    config = load_config(args.config)
    config.dataset.root = root
    config_name = os.path.basename(args.config)[:os.path.basename(args.config).rfind('.')]
    seed_all(config.train.seed)
    resumed = load_state(resume_from) if resume_from is not None else None
    if resumed is not None:
        validate_round_checkpoint(resumed, config, root, epoch_now, model_ckpt)
        if resumed['iteration'] == config.train.max_iters:
            return str(resume_from)

    # Logging
    log_dir = get_new_log_dir(args.logdir, prefix=config_name, tag='epoch%d' % epoch_now)
    ckpt_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    logger = get_logger('train.round%d' % epoch_now, log_dir)

    writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    # logger.info(args)
    # logger.info(config)
    shutil.copyfile(args.config, os.path.join(log_dir, os.path.basename(args.config)))

    # Transforms
    featurizer = FeaturizeMol(config.chem.atomic_numbers, config.chem.mol_bond_types,
                            use_mask_node=config.transform.use_mask_node,
                            use_mask_edge=config.transform.use_mask_edge
                            )
    transform = Compose([
        featurizer,
    ])
    # Datasets and loaders
    logger.info('Loading dataset...')
    print('train|Loading dataset...')
    dataset = get_dataset(root, transform = transform)

    train_size = int(0.7 * len(dataset))
    val_size = len(dataset) - train_size

    if not train_size or not val_size:
        raise ValueError('The generated pool needs nonempty training and validation splits')
    if resumed is None:
        train_set, val_set = random_split(dataset, [train_size, val_size])
    else:
        split = resumed['dataset_split']
        indices = split['train_indices'] + split['validation_indices']
        if (len(split['train_indices']) != train_size or len(split['validation_indices']) != val_size or
                sorted(indices) != list(range(len(dataset)))):
            raise ValueError('Saved RL train/validation split does not match the dataset')
        train_set = Subset(dataset, split['train_indices'])
        val_set = Subset(dataset, split['validation_indices'])
    append_record(os.path.join(log_dir, 'dataset.jsonl'), dict(
        root=root, source_checkpoint=model_ckpt, resume_from=resume_from,
        train_indices=train_set.indices, validation_indices=val_set.indices))
    logger.info(f"Dataset length: {len(dataset)}")
    print(f"train|Dataset length: {len(dataset)}")
    if len(dataset) == 0:
        raise ValueError("The dataset is empty. Please check the data loading process.")

    train_iterator = inf_iterator(DataLoader(
        train_set, 
        batch_size = config.train.batch_size, 
        shuffle = True,
        num_workers = config.train.num_workers,
        pin_memory = config.train.pin_memory,
        follow_batch = featurizer.follow_batch,
        exclude_keys = featurizer.exclude_keys,
    ))
    val_loader = DataLoader(val_set, config.train.batch_size, shuffle=False,
                            follow_batch=featurizer.follow_batch, exclude_keys=featurizer.exclude_keys)
    
    # Model
    logger.info('Building model...')
    print('train|Building model...')
    if config.model.name == 'diffusion':
        model = MDRL(
            config=config.model,
            num_node_types=featurizer.num_node_types,
            num_edge_types=featurizer.num_edge_types
        ).to(args.device)
    else:
        raise NotImplementedError('Model %s not implemented' % config.model.name)
    print('train|Num of trainable parameters is', np.sum([p.numel() for p in model.parameters() if p.requires_grad]))
    ckpt = resumed if resumed is not None else load_state(model_ckpt)
    model.load_state_dict(ckpt['model'])
    
    # Optimizer and scheduler
    optimizer = get_optimizer(config.train.optimizer, model)
    scheduler = get_scheduler(config.train.scheduler, optimizer)
    scaler = torch.cuda.amp.GradScaler(enabled=config.train.use_amp)
    current = {}
    amp_skipped_updates = 0
    start_iteration = 1
    if resumed is not None:
        restore_training_components(resumed, optimizer, scheduler, scaler, args.device)
        amp_skipped_updates = resumed['amp_skipped_updates']
        start_iteration = resumed['iteration'] + 1
        logger.info('Resuming round %d from iteration %d: %s', epoch_now, start_iteration - 1, resume_from)

    def training_state(it):
        return dict(config=config, model=model.state_dict(), optimizer=optimizer.state_dict(),
                    scheduler=scheduler.state_dict(), scaler=scaler.state_dict(), iteration=it,
                    round=epoch_now, source_checkpoint=model_ckpt, dataset_root=root,
                    dataset_split=dict(train_indices=train_set.indices, validation_indices=val_set.indices),
                    data_iterator_restored=False,
                    amp_skipped_updates=amp_skipped_updates, rng_state=capture_training_rng(args.device))

    def record_failure(exc, it):
        directory = os.path.join(log_dir, 'diagnostics')
        os.makedirs(directory, exist_ok=True)
        filename = os.path.join(directory, 'first_failure.pt')
        if not os.path.exists(filename):
            state = training_state(it)
            state.update(diagnostic_only=True, error=repr(exc), context=current,
                         stage=getattr(exc, 'stage', current.get('phase')),
                         values=getattr(exc, 'values', {}),
                         gradients={name: parameter.grad for name, parameter in model.named_parameters()
                                    if parameter.grad is not None})
            torch.save(cpu_tree(state), filename)
        append_record(os.path.join(log_dir, 'failures.jsonl'), dict(
            round=epoch_now, iteration=it, error=repr(exc), snapshot=filename))

    def train(it):
        nonlocal amp_skipped_updates
        optimizer.zero_grad(set_to_none=True)
        batch = next(train_iterator).to(args.device)
        current.clear()
        current.update(phase='train', iteration=it, batch=batch.to_dict(), rng_before_loss=capture_training_rng(args.device))
        require_amp_scale(scaler, current)
        
        pos_noise = torch.randn_like(batch.node_pos) * config.train.pos_noise_std
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=config.train.use_amp):
            loss_dict = model.get_loss(
                # compose
                node_type = batch.node_type,
                node_pos = batch.node_pos + pos_noise,
                batch_node = batch.node_type_batch,
                halfedge_type = batch.halfedge_type,
                halfedge_index = batch.halfedge_index,
                batch_halfedge = batch.halfedge_type_batch,
                num_mol = batch.num_graphs,
            )
        loss = loss_dict['loss']
        current['losses'] = loss_dict
        require_finite('train_loss', loss_dict, current)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        orig_grad_norm = clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
        try:
            require_finite('train_gradient_norm', {'gradient_norm': orig_grad_norm}, current)
        except NonfiniteTrainingError as exc:
            skipped = handle_amp_gradient(exc, scaler, optimizer)
            amp_skipped_updates += 1
            append_record(os.path.join(log_dir, 'amp_skips.jsonl'), dict(
                iteration=it, scale_before=skipped.scale_before, scale_after=skipped.scale_after))
            logger.warning('AMP skipped update %d: %s -> %s', it, skipped.scale_before, skipped.scale_after)
            return
        scaler.step(optimizer)
        scaler.update()
        require_amp_scale(scaler, current)

        log_info = '[Train] Iter %d | ' % it + ' | '.join([
            '%s: %.6f' % (k, v.item()) for k, v in loss_dict.items()
        ])
        if it % 1000 == 0:
            logger.info(log_info)
            print(log_info)
        for k, v in loss_dict.items():
            writer.add_scalar('train/%s' % k, v.item(), it)
        writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], it)
        writer.add_scalar('train/grad', orig_grad_norm, it)
        writer.flush()
        append_record(os.path.join(log_dir, 'train.jsonl'), dict(
            iteration=it, losses={key: value.item() for key, value in loss_dict.items()},
            gradient_norm=orig_grad_norm.item(), lr=optimizer.param_groups[0]['lr']))

    def validate(it):
        sum_n =  0   # num of loss
        sum_loss_dict = {} 
        with torch.no_grad():
            model.eval()
            # for batch in tqdm(val_loader, desc='Validate'):
            for batch in val_loader:
                batch = batch.to(args.device)
                current.clear()
                current.update(phase='validation', iteration=it, batch_index=sum_n,
                               batch=batch.to_dict(), rng_before_loss=capture_training_rng(args.device))
                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=config.train.use_amp):
                    loss_dict = model.get_loss(
                        # compose
                        node_type = batch.node_type,
                        node_pos = batch.node_pos,
                        batch_node = batch.node_type_batch,
                        halfedge_type = batch.halfedge_type,
                        halfedge_index = batch.halfedge_index,
                        batch_halfedge = batch.halfedge_type_batch,
                        num_mol = batch.num_graphs,
                    )
                current['losses'] = loss_dict
                require_finite('validation_loss', loss_dict, current)
                if len(sum_loss_dict) == 0:
                    sum_loss_dict = {k: v.item() for k, v in loss_dict.items()}
                else:
                    for key in sum_loss_dict.keys():
                        sum_loss_dict[key] += loss_dict[key].item()
                sum_n += 1
        
        # finish all batches
        avg_loss_dict = {k: v / sum_n for k, v in sum_loss_dict.items()}
        avg_loss = avg_loss_dict['loss']
        # update lr scheduler
        if config.train.scheduler.type == 'plateau':
            scheduler.step(avg_loss)
        elif config.train.scheduler.type == 'warmup_plateau':
            scheduler.step_ReduceLROnPlateau(avg_loss)
        else:
            scheduler.step()

        log_info = '[Validate] Iter %d | ' % it + ' | '.join([
            '%s: %.6f' % (k, v) for k, v in avg_loss_dict.items()
        ])
        logger.info(log_info)
        print(log_info)
        for k, v in avg_loss_dict.items():
            writer.add_scalar('val/%s' % k, v, it)
        writer.flush()
        append_record(os.path.join(log_dir, 'validation.jsonl'), dict(iteration=it, losses=avg_loss_dict))
        return avg_loss

    try:
        now = time.localtime()
        print(time.strftime("%Y-%m-%d %H:%M:%S", now))
        model.train()
        for it in range(start_iteration, config.train.max_iters+1):
            try:
                train(it)
                if it % config.train.val_freq == 0 or it == config.train.max_iters:
                    validation_loss = validate(it)
                    ckpt_path = os.path.join(ckpt_dir, 'epoch%d_it%d.pt' % (epoch_now, it))
                    state = training_state(it)
                    state['validation_loss'] = validation_loss
                    validate_resume_numerics(state)
                    save_state(ckpt_path, state)
                    model.train()
            except Exception as exc:
                try:
                    record_failure(exc, it)
                except Exception:
                    logger.exception('Could not save failure diagnostics; original error: %r', exc)
                raise
        now = time.localtime()
        print(time.strftime("%Y-%m-%d %H:%M:%S", now))
        return ckpt_path
    except KeyboardInterrupt:
        logger.info('Terminating...')                    
        print('train|Terminating...')
        raise
    finally:
        writer.close()
        if dataset.db is not None:
            dataset._close_db()
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)
