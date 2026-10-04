"""
CryoCCD Train Script
"""

import time
import torch
import logging
import argparse
from tqdm import tqdm
import torch.distributed as dist

from cryoccd.options import process_opt, base_add_args
from cryoccd.datasets import create_dataset
from cryoccd.models import create_model
from cryoccd import models
from cryoccd import datasets
from cryoccd.config import _select_model, _select_dataset


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def add_training_args(parser):
    parser.add_argument('--save_latest_freq', type=int, default=100000,
                        help='frequency of saving the latest results')
    parser.add_argument('--save_epoch_freq', type=int, default=50,
                        help='frequency of saving checkpoints at the end of epochs')
    parser.add_argument('--save_by_iter', action='store_true',
                        help='whether saves model by iteration')
    parser.add_argument('--continue_train', action='store_true',
                        help='continue training: load the latest model')
    parser.add_argument('--epoch_count', type=int, default=1,
                        help='the starting epoch count')
    parser.add_argument('--phase', type=str, default='train',
                        help='train, val, test, etc')
    
    # Training parameters
    parser.add_argument('--n_epochs', type=int, default=25,
                        help='number of epochs with the initial learning rate')
    parser.add_argument('--n_epochs_decay', type=int, default=75,
                        help='number of epochs to linearly decay learning rate to zero')
    parser.add_argument('--beta1', type=float, default=0.5,
                        help='momentum term of adam')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='initial learning rate for adam')
    parser.add_argument('--gan_mode', type=str, default='lsgan',
                        help='the type of GAN objective. [vanilla| lsgan | wgangp]')
    parser.add_argument('--pool_size', type=int, default=50,
                        help='the size of image buffer that stores previously generated images')
    parser.add_argument('--lr_policy', type=str, default='linear',
                        help='learning rate policy. [linear | step | plateau | cosine]')
    parser.add_argument('--lr_decay_iters', type=int, default=50,
                        help='multiply by a gamma every lr_decay_iters iterations')
    parser.add_argument('--iters_per_epoch', type=int, default=100,
                        help='number of iterations per epoch')
    parser.add_argument('--T', type=int, default=1000,
                        help='T parameter')
    parser.add_argument('--sampling_steps', type=int, default=100,
                        help='sampling steps parameter')
    
    # Logging parameters
    parser.add_argument('--print_freq', type=int, default=50,
                        help='frequency of showing training results on console')
    
    return parser


def setup_argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser = base_add_args(parser)
    parser = add_training_args(parser)
    parser = models.get_option_setter(_select_model)(parser, is_train=True)
    parser = datasets.get_option_setter(_select_dataset)(parser, is_train=True)
    return parser


def is_primary():
    """Check if this is the primary process in distributed training."""
    return (not dist.is_initialized()) or dist.get_rank() == 0


def print_losses(epoch, epoch_iter, losses, t_comp, t_data):
    """Print current losses to console."""
    message = f'(epoch: {epoch}, iters: {epoch_iter}, time: {t_comp:.3f}, data: {t_data:.3f}) '
    for k, v in losses.items():
        message += f'{k}: {v:.3f} '
    logger.info(message)


def initialize_model(model, opt, data, epoch, iteration):
    """Initialize model based on type and training stage."""
    if epoch == opt.epoch_count and iteration == 0:
        # builds lazily-sized modules (e.g. the NCE projection head) before optimizers/schedulers are set up
        model.data_dependent_initialize(data)
        model.setup(opt)
        model.parallelize()


def save_model_checkpoint(model, opt, epoch, total_iters, save_type='latest'):
    """Save model checkpoint."""
    if save_type == 'latest':
        logger.info(f'Saving the latest model (epoch {epoch}, total_iters {total_iters})')
        save_suffix = f'iter_{total_iters}' if opt.save_by_iter else 'latest'
        model.save_networks(save_suffix)
    elif save_type == 'epoch':
        logger.info(f'Saving the model at the end of epoch {epoch}, iters {total_iters}')
        model.save_networks('latest')
        model.save_networks(epoch)


def train_epoch(model, dataset, opt, epoch, total_iters_start):
    """Train for one epoch."""
    epoch_start_time = time.time()
    iter_data_time = time.time()
    epoch_iter = 0
    total_iters = total_iters_start
    
    dataset_size = len(dataset)
    progress_bar = tqdm(dataset, desc=f"Epoch {epoch}/{opt.n_epochs + opt.n_epochs_decay}")
    
    for i, data in enumerate(progress_bar):
        iter_start_time = time.time()
        
        if total_iters % opt.print_freq == 0:
            t_data = iter_start_time - iter_data_time
        
        total_iters += opt.batch_size
        epoch_iter += opt.batch_size
        
        if len(opt.gpu_ids) > 0:
            torch.cuda.synchronize()
        
        initialize_model(model, opt, data, epoch, i)
        
        model.set_input(data)
        model.optimize_parameters()
        
        if len(opt.gpu_ids) > 0:
            torch.cuda.synchronize()
        
        if total_iters % opt.print_freq == 0:
            losses = model.get_current_losses()
            t_comp = (time.time() - iter_start_time) / opt.batch_size
            print_losses(epoch, epoch_iter, losses, t_comp, t_data)
        
        if total_iters % opt.save_latest_freq == 0:
            save_model_checkpoint(model, opt, epoch, total_iters, 'latest')
        
        progress_bar.set_postfix({
            'iter': f'{epoch_iter}/{dataset_size}',
            'total_iters': total_iters
        })
        
        iter_data_time = time.time()
    
    epoch_time = time.time() - epoch_start_time
    logger.info(f'End of epoch {epoch} / {opt.n_epochs + opt.n_epochs_decay} '
                f'\t Time Taken: {epoch_time:.0f} sec')
    
    return total_iters


def main(args):
    opt = process_opt(args)
    
    logger.info(f"Using model: {opt.model}")
    logger.info(f"Training configuration: T={opt.T}, sampling_steps={opt.sampling_steps}, "
                f"lambda_NCE={getattr(opt, 'lambda_NCE', 'N/A')}")
    
    dataset = create_dataset(opt)
    model = create_model(opt)
    dataset_size = len(dataset)
    
    logger.info(f'The number of training images = {dataset_size}')
    
    model.setup(opt)
    
    total_iters = 0
    total_epochs = opt.n_epochs + opt.n_epochs_decay
    
    for epoch in range(opt.epoch_count, total_epochs + 1):
        total_iters = train_epoch(model, dataset, opt, epoch, total_iters)
        model.update_learning_rate()
        if epoch % opt.save_epoch_freq == 0:
            save_model_checkpoint(model, opt, epoch, total_iters, 'epoch')
    
    logger.info('Training completed. Saving final model...')
    model.save_networks('latest')
    model.save_networks(epoch)
    
    logger.info('Training finished successfully!')


if __name__ == '__main__':
    parser = setup_argument_parser()
    args = parser.parse_args()
    main(args)