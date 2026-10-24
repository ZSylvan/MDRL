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
from utils.pretrain_record import PretrainRecord, git_version



def run_training(args, config, logger, writer, record):
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

    def train(it):
        optimizer.zero_grad(set_to_none=True)
        batch = next(train_iterator).to(args.device)
        
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
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        orig_grad_norm = clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

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
            for batch in tqdm(val_loader, desc='Validate'):
                batch = batch.to(args.device)
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

    model.train()
    for it in range(1, config.train.max_iters+1):
        record.state['iteration'] = it
        record.state['phase'] = 'train'
        try:
            train(it)
            record.state['completed_train_calls'] += 1
        except RuntimeError as e:
            logger.error('Runtime Error ' + str(e))
            logger.error('Skipping Iteration %d' % it)
            record.record_skip(e)
        if it % config.train.val_freq == 0 or it == config.train.max_iters:
            record.state['phase'] = 'validation'
            validation_losses = validate(it)
            record.state['phase'] = 'checkpoint'
            improved = record.save_checkpoint({
                'config': config,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'scaler': scaler.state_dict(),
                'iteration': it,
            }, validation_losses)
            if improved:
                logger.info('Best checkpoint: iteration %d, validation loss %.6f',
                            it, validation_losses['loss'])
            model.train()

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
        with PretrainRecord(log_dir, metadata) as record:
            writer = torch.utils.tensorboard.SummaryWriter(log_dir)
            try:
                run_training(args, config, logger, writer, record)
            finally:
                writer.close()
    except KeyboardInterrupt:
        logger.info('Terminating...')
        raise
    except Exception:
        logger.exception('Pretraining failed')
        raise

