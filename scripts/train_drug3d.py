import sys
import os
import shutil
import argparse
import distutils.version
sys.path.append('.')

from tqdm import tqdm
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
from torch.nn.utils import clip_grad_norm_
import torch.utils.tensorboard
from torch_geometric.loader import DataLoader

from models.model import MDRL
from utils.dataset import get_dataset
from utils.transforms import FeaturizeMol, Compose
from utils.misc import *
from utils.train import *
from utils.pretrain_checks import (AmpSkippedUpdate, NonfiniteTrainingError,
                                    handle_amp_gradient, require_amp_scale, require_finite)
from utils.pretrain_record import PretrainRecord, git_version
from utils.pretrain_resume import (capture_rng_state, find_resume_best,
                                   load_resume_checkpoint, restore_training_state)



def run_training(args, config, logger, writer, record, resume_checkpoint=None):
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
    dataset, subsets = get_dataset(
        config = config.dataset,
        transform = transform,
    )
    train_set, val_set = subsets['train'], subsets['val']
    steps_per_epoch = (len(train_set) + config.train.batch_size - 1) // config.train.batch_size
    val_every_epochs = config.train.get('val_every_epochs')
    if val_every_epochs is not None and config.train.val_freq != steps_per_epoch * val_every_epochs:
        raise ValueError('val_freq does not match the effective training split and val_every_epochs')
    checkpoint_freq = config.train.get('checkpoint_freq', 0)
    checkpoint_milestones = set(config.train.get('checkpoint_milestones', []))
    record.state['schedule'] = {
        'train_size': len(train_set), 'steps_per_epoch': steps_per_epoch,
        'val_freq': config.train.val_freq, 'val_every_epochs': val_every_epochs,
        'checkpoint_freq': checkpoint_freq, 'checkpoint_milestones': sorted(checkpoint_milestones),
    }
    logger.info('Training schedule: %s', record.state['schedule'])
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
    if config.model.name == 'diffusion':
        model = MDRL(
            config=config.model,
            num_node_types=featurizer.num_node_types,
            num_edge_types=featurizer.num_edge_types
        ).to(args.device)
    else:
        raise NotImplementedError('Model %s not implemented' % config.model.name)
    print('Num of trainable parameters is', np.sum([p.numel() for p in model.parameters() if p.requires_grad]))

    # Optimizer and scheduler
    optimizer = get_optimizer(config.train.optimizer, model)
    scheduler = get_scheduler(config.train.scheduler, optimizer)
    scaler = torch.cuda.amp.GradScaler(enabled=config.train.use_amp)
    start_iter = 1
    if resume_checkpoint is not None:
        start_iter = restore_training_state(resume_checkpoint, model, optimizer, scheduler, scaler)
        logger.info('Resuming after iteration %d; total target %d',
                    start_iter - 1, config.train.max_iters)
        logger.warning('DataLoader shuffle/prefetch position is not restored; '
                       'resume is not bitwise equivalent to uninterrupted training')
        if resume_checkpoint.get('rng_state') is None:
            logger.warning('Legacy checkpoint has no RNG state; using seeded process state')

    def train(it):
        optimizer.zero_grad(set_to_none=True)
        batch = next(train_iterator).to(args.device)
        
        pos_noise = torch.randn_like(batch.node_pos) * config.train.pos_noise_std
        rng_before_loss = capture_rng_state()
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
        context = {'batch': batch, 'pos_noise': pos_noise,
                   'rng_before_loss': rng_before_loss, 'losses': loss_dict}
        require_finite('train_loss', loss_dict, context)
        loss = loss_dict['loss']
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        orig_grad_norm = clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
        # The returned norm describes the unclipped gradients.
        try:
            require_finite('train_gradient_norm', {'gradient_norm': orig_grad_norm}, context)
        except NonfiniteTrainingError as error:
            raise handle_amp_gradient(error, scaler, optimizer)
        scaler.step(optimizer)
        scaler.update()
        require_amp_scale(scaler, context)

        log_info = '[Train] Iter %d | ' % it + ' | '.join([
            '%s: %.6f' % (k, v.item()) for k, v in loss_dict.items()
        ])
        logger.info(log_info)
        for k, v in loss_dict.items():
            writer.add_scalar('train/%s' % k, v.item(), it)
        writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], it)
        writer.add_scalar('train/grad', orig_grad_norm, it)
        writer.flush()

    def validate(it):
        sum_n =  0   # num of loss
        sum_loss_dict = {} 
        with torch.no_grad():
            model.eval()
            for batch_index, batch in enumerate(tqdm(val_loader, desc='Validate'), 1):
                batch = batch.to(args.device)
                rng_before_loss = capture_rng_state()
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
                require_finite('validation_loss', loss_dict, {
                    'batch': batch, 'validation_batch_index': batch_index,
                    'rng_before_loss': rng_before_loss, 'losses': loss_dict,
                })
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
        for k, v in avg_loss_dict.items():
            writer.add_scalar('val/%s' % k, v, it)
        writer.flush()
        return avg_loss_dict

    def snapshot(it):
        return {
            'config': config, 'model': model.state_dict(),
            'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
            'scaler': scaler.state_dict(), 'iteration': it,
            'rng_state': capture_rng_state(),
        }

    model.train()
    for it in range(start_iter, config.train.max_iters+1):
        record.state['iteration'] = it
        record.state['phase'] = 'train'
        try:
            train(it)
            record.state['completed_train_calls'] += 1
        except AmpSkippedUpdate as event:
            record.state['completed_train_calls'] += 1
            record.record_amp_skip(event, snapshot(it), {
                name: parameter.grad for name, parameter in model.named_parameters()
                if parameter.grad is not None
            })
            logger.warning('[AMP] Iter %d | parameter update skipped | scale %s -> %s | loss %.6f',
                           it, event.scale_before, event.scale_after,
                           event.error.context['losses']['loss'].item())
        except NonfiniteTrainingError as e:
            record.record_numerical_failure(e, snapshot(it), {
                name: parameter.grad for name, parameter in model.named_parameters()
                if parameter.grad is not None
            })
            raise
        except RuntimeError as e:
            logger.error('Runtime Error ' + str(e))
            logger.error('Skipping Iteration %d' % it)
            record.record_skip(e)
        if it % config.train.val_freq == 0 or it == config.train.max_iters:
            record.state['phase'] = 'validation'
            try:
                validation_losses = validate(it)
            except NonfiniteTrainingError as e:
                record.record_numerical_failure(e, snapshot(it), {})
                raise
            record.state['phase'] = 'checkpoint'
            improved = record.save_checkpoint(snapshot(it), validation_losses)
            if improved:
                logger.info('Best checkpoint: iteration %d, validation loss %.6f',
                            it, validation_losses['loss'])
            model.train()
        elif it in checkpoint_milestones or (checkpoint_freq and it % checkpoint_freq == 0):
            record.state['phase'] = 'checkpoint'
            record.save_checkpoint(snapshot(it), keep_periodic=it in checkpoint_milestones)
            logger.info('Saved %s checkpoint at iteration %d without validation',
                        'milestone' if it in checkpoint_milestones else 'recovery last', it)

    # model.train()
    # for it in range(1, config.train.max_iters+1):
    #     train(it)
    #     if it % config.train.val_freq == 0 or it == config.train.max_iters:
    #         validate(it)
    #         ckpt_path = os.path.join(ckpt_dir, '%d.pt' % it)
    #         torch.save({
    #             'config': config,
    #             'model': model.state_dict(),
    #             'optimizer': optimizer.state_dict(),
    #             'scheduler': scheduler.state_dict(),
    #             'iteration': it,
    #         }, ckpt_path)
    #         model.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/train/train_MDRL.yml')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--logdir', type=str, default=None)
    parser.add_argument('--resume', type=str, default=None,
                        help='Trusted complete checkpoint; continue in a new output directory')
    args = parser.parse_args()

    config = load_config(args.config)
    config_name = os.path.splitext(os.path.basename(args.config))[0]
    args.logdir = args.logdir or config.get('logdir', 'logs')
    seed_all(config.train.seed)
    log_dir = get_new_log_dir(args.logdir, prefix=config_name)
    os.makedirs(os.path.join(log_dir, 'checkpoints'))
    logger = get_logger('train', log_dir)
    logger.info(args)
    logger.info(config)
    shutil.copyfile(args.config, os.path.join(log_dir, os.path.basename(args.config)))
    metadata = {
        'git': git_version(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        'args': vars(args),
        'config_path': os.path.abspath(args.config),
        'dataset_root': os.path.abspath(config.dataset.root),
        'python': sys.version,
        'executable': sys.executable,
        'torch': torch.__version__,
    }
    try:
        with PretrainRecord(
                log_dir, metadata,
                periodic_checkpoint_limit=config.train.get('max_checkpoint_files', 52) - 2,
                pinned_iterations=config.train.get('checkpoint_milestones', [])) as record:
            resume_checkpoint = None
            if args.resume is not None:
                record.state['phase'] = 'resume'
                resume_checkpoint = load_resume_checkpoint(args.resume, config)
                best_path = find_resume_best(args.resume, resume_checkpoint)
                record.resume_from(resume_checkpoint, args.resume, best_path)
            writer = torch.utils.tensorboard.SummaryWriter(log_dir)
            try:
                run_training(args, config, logger, writer, record, resume_checkpoint)
            finally:
                writer.close()
    except KeyboardInterrupt:
        logger.info('Terminating...')
        raise
    except Exception:
        logger.exception('Pretraining failed')
        raise

