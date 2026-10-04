"""
CryoCCD Test Script
==================

This script performs inference using trained CryoCCD models to generate synthetic
cryo-EM micrographs and extract particle stacks for evaluation.
"""

import os
import time
import pickle
import logging
import argparse
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing
import starfile

from cryoccd.options import process_opt, base_add_args
from cryoccd.datasets import create_dataset
from cryoccd.models import create_model
from cryoccd.utils import make_dirs, save_as_mrc, save_as_png
from cryoccd import models, datasets
from cryoccd.config import _select_model, _select_dataset

torch.multiprocessing.set_sharing_strategy('file_system')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def add_test_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add test-specific arguments to the parser."""
    # Basic test parameters
    parser.add_argument('--phase', type=str, default='test',
                        help='Phase: train, val, test, etc')
    parser.add_argument('--eval', action='store_true',
                        help='Use eval mode during test time')
    parser.add_argument('--num_test', type=int, default=50,
                        help='Number of test images to run')
    parser.add_argument('--T', type=int, default=1000,
                        help='Total diffusion steps; must match the value used in training')
    parser.add_argument('--sampling_steps', type=int, default=20,
                        help='Number of sampler steps at inference')
    
    # Output configuration
    parser.add_argument('--save_dir', type=str, required=True,
                        help='Path to save generated micrographs')
    
    # Shift generation
    parser.add_argument('--generate_shift', action='store_true', default=False,
                        help='Generate shift images')
    parser.add_argument('--pixel_shift_max', type=int, default=5,
                        help='Maximum pixel shift for shift images')
    
    # Data storage options
    parser.add_argument('--store_real_A_trans', action='store_true', default=False,
                        help='Whether to save image translation if exists')
    parser.add_argument('--store_realA_ps', action='store_true',
                        help='Store particle stack of real micrographs')
    parser.add_argument('--store_fakeB_ps', action='store_true',
                        help='Store particle stack of fake micrographs')
    parser.add_argument('--store_realA_rot', action='store_true',
                        help='Store rotation matrix of real micrographs')
    
    # Starfile configuration
    parser.add_argument('--starfile', type=str,
                        help='Path to starfile')
    parser.add_argument('--starfile_upsample', type=int, default=1,
                        help='Upsample factor for starfile')
    
    return parser


def setup_argument_parser() -> argparse.ArgumentParser:
    """Setup and return the argument parser with all required arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser = base_add_args(parser)
    parser = add_test_args(parser)
    parser = models.get_option_setter(_select_model)(parser, is_train=False)
    parser = datasets.get_option_setter(_select_dataset)(parser, is_train=False)
    return parser

class ParticleDataManager:
    def __init__(self, particle_limit: int = 100000):
        self.particle_limit = particle_limit
        self.reset()
    
    def reset(self):
        """Reset all data structures."""
        self.particle_table = {
            'rlnCoordinateX': [],
            'rlnCoordinateY': [],
            'rlnMicrographName': [],
            'rlnRandomSubset': [],
        }
        self.fake_B_ps = []
        self.clean_real_A_ps = []
        self.gt_rot = []
        self.gt_trans = []
    
    def add_particles(self, img_name: str, fake_B: np.ndarray, clean_real_A: np.ndarray,
                     rotations: np.ndarray, boxes: np.ndarray, centers: np.ndarray,
                     H: int, W: int, generate_shift: bool = False, 
                     pixel_shift_max: int = 5) -> int:
        particles_added = 0
        pN = rotations.shape[0]
        
        for j in range(pN):
            if len(self.fake_B_ps) >= self.particle_limit:
                break
                
            xi, yi, xa, ya = boxes[j]
            x_shift, y_shift = 0, 0
            
            # Apply random shift if requested
            if generate_shift:
                x_shift, y_shift = np.random.randint(-pixel_shift_max, pixel_shift_max+1, 2)
                xi, yi, xa, ya = xi + x_shift, yi + y_shift, xa + x_shift, ya + y_shift
            
            # Check bounds
            if xi < 0 or yi < 0 or xa > H or ya > W:
                continue
            
            # Extract particle data
            xi, yi, xa, ya = int(xi), int(yi), int(xa), int(ya)
            
            # Store particle information
            self.particle_table['rlnCoordinateX'].append(int(centers[j, 1]))
            self.particle_table['rlnCoordinateY'].append(int(centers[j, 0]))
            self.particle_table['rlnMicrographName'].append(f'{img_name}.mrc')
            
            # Store particle data
            self.fake_B_ps.append(fake_B[xi:xa, yi:ya])
            self.clean_real_A_ps.append(clean_real_A[xi:xa, yi:ya])
            self.gt_rot.append(rotations[j])
            self.gt_trans.append([y_shift, x_shift])
            
            particles_added += 1
        
        return particles_added
    
    def finalize_and_save(self, save_dir: str, starfile_upsample: int = 1):
        """Finalize data processing and save all results."""
        if not self.fake_B_ps:
            logger.warning("No particles to save!")
            return
        
        # Convert lists to arrays
        gt_rot = np.stack(self.gt_rot, axis=0).astype(np.float32)
        gt_trans = np.stack(self.gt_trans, axis=0).astype(np.float32)
        fake_B_ps = np.stack(self.fake_B_ps)
        clean_real_A_ps = np.stack(self.clean_real_A_ps)
        
        N = len(self.particle_table['rlnCoordinateX'])
        
        # Normalize translations
        if fake_B_ps.shape[0] > 0:
            H, W = fake_B_ps[0].shape
            gt_trans[:, 1] /= float(H)
            gt_trans[:, 0] /= float(W)
        
        # Update particle table
        self.particle_table['rlnCoordinateX'] = [x * starfile_upsample 
                                                for x in self.particle_table['rlnCoordinateX']]
        self.particle_table['rlnCoordinateY'] = [y * starfile_upsample 
                                                for y in self.particle_table['rlnCoordinateY']]
        self.particle_table['rlnRandomSubset'] = np.zeros(N, dtype=np.int32)
        
        # Save all data
        self._save_rotations(save_dir, gt_rot)
        self._save_translations(save_dir, gt_trans)
        self._save_poses(save_dir, gt_rot, gt_trans)
        self._save_starfile(save_dir)
        self._save_particle_stacks(save_dir, fake_B_ps, clean_real_A_ps)
        
        logger.info(f'Successfully saved {N} particles to {save_dir}')
    
    def _save_rotations(self, save_dir: str, gt_rot: np.ndarray):
        """Save rotation matrices."""
        save_path = os.path.join(save_dir, 'gt_rots.npy')
        np.save(save_path, gt_rot)
        logger.info(f'Saved gt_rot to {save_path}, shape: {gt_rot.shape}')
    
    def _save_translations(self, save_dir: str, gt_trans: np.ndarray):
        """Save translation vectors."""
        save_path = os.path.join(save_dir, 'gt_trans.npy')
        np.save(save_path, gt_trans)
        logger.info(f'Saved gt_trans to {save_path}, shape: {gt_trans.shape}')
    
    def _save_poses(self, save_dir: str, gt_rot: np.ndarray, gt_trans: np.ndarray):
        """Save combined pose data."""
        save_path = os.path.join(save_dir, 'gt_pose.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump((gt_rot, gt_trans), f)
        logger.info(f'Saved gt_pose to {save_path}')
    
    def _save_starfile(self, save_dir: str):
        """Save particle metadata as starfile."""
        star_path = os.path.join(save_dir, 'particles.star')
        particle_df = pd.DataFrame(self.particle_table)
        starfile.write(particle_df, star_path, overwrite=True)
        logger.info(f'Saved particle table to {star_path}, total particles: {len(particle_df)}')
    
    def _save_particle_stacks(self, save_dir: str, fake_B_ps: np.ndarray, 
                            clean_real_A_ps: np.ndarray):
        """Save particle stacks."""
        # Save fake B particles
        fake_B_path = os.path.join(save_dir, 'fake_B_ps.mrc')
        save_as_mrc(fake_B_ps, fake_B_path)
        logger.info(f'Saved fakeB_ps to {fake_B_path}, shape: {fake_B_ps.shape}')
        
        # Save clean real A particles
        clean_A_path = os.path.join(save_dir, 'clean_real_A_ps.mrc')
        save_as_mrc(clean_real_A_ps, clean_A_path)
        logger.info(f'Saved clean_real_A_ps to {clean_A_path}, shape: {clean_real_A_ps.shape}')


class OutputManager:
    """Manages output directory structure and file saving."""
    
    def __init__(self, base_save_dir: str, model_name: str, experiment_name: str):
        self.save_dir = os.path.join(base_save_dir, model_name, experiment_name)
        self.mrc_save_dirs = {}
        self.png_save_dirs = {}
    
    def setup_directories(self, mrc_names: List[str], png_names: List[str]):
        """Create all necessary directories."""
        for name in mrc_names:
            dir_path = os.path.join(self.save_dir, f'mics_mrc_{name}')
            self.mrc_save_dirs[name] = dir_path
            make_dirs(dir_path)
        
        for name in png_names:
            dir_path = os.path.join(self.save_dir, f'mics_png_{name}')
            self.png_save_dirs[name] = dir_path
            make_dirs(dir_path)
    
    def save_visuals(self, visuals: Dict[str, torch.Tensor], img_name: str, 
                    mrc_names: List[str], png_names: List[str]):
        """Save visual outputs in both MRC and PNG formats."""
        # Save MRC files
        for name in mrc_names:
            if name in visuals:
                image = visuals[name][0, 0].cpu().numpy()
                save_path = os.path.join(self.mrc_save_dirs[name], f'{img_name}.mrc')
                save_as_mrc(image, save_path)
        
        # Save PNG files
        for name in png_names:
            if name in visuals:
                image = visuals[name][0, 0].cpu().float().numpy()
                save_path = os.path.join(self.png_save_dirs[name], f'{img_name}.png')
                
                # Special handling for mask and weight map images
                normalize = not (name == 'mask_A' or 'weight_map' in name)
                save_as_png(image, save_path, normalize=normalize)


class PerformanceMonitor:
    def __init__(self, total_items: int, print_frequency: int = 10):
        self.total_items = total_items
        self.print_frequency = max(1, total_items // print_frequency)
        self.start_time = time.time()
        self.model_inference_time = 0.0
        self.current_iterations = 0
    
    def update(self, inference_time: float, num_processed: int):
        """Update performance metrics."""
        self.model_inference_time += inference_time
        self.current_iterations += num_processed
    
    def should_print(self, iteration: int) -> bool:
        """Check if performance should be printed."""
        return iteration % self.print_frequency == 0
    
    def get_stats(self) -> Tuple[float, float]:
        """Get current performance statistics."""
        total_elapsed = time.time() - self.start_time
        avg_save_time = (total_elapsed - self.model_inference_time) / max(self.current_iterations, 1e-8)
        avg_inf_time = self.model_inference_time / max(self.current_iterations, 1e-8)
        return avg_save_time, avg_inf_time
    
    def print_progress(self, iteration: int):
        """Print current progress."""
        avg_save_time, avg_inf_time = self.get_stats()
        total_elapsed = time.time() - self.start_time
        
        logger.info(
            f'Processing {self.current_iterations}/{self.total_items} items, '
            f'avg_save: {avg_save_time:.3f}s/it, avg_inf: {avg_inf_time:.3f}s/it, '
            f'total: {total_elapsed:.3f}s'
        )

def validate_options(opt) -> None:
    assert len(opt.gpu_ids) == 1, 'Only single GPU is supported'
    assert opt.batch_size == 1, 'Only batch_size=1 is supported'
    
    if not os.path.exists(opt.save_dir):
        os.makedirs(opt.save_dir, exist_ok=True)


def run_inference(model, dataset, opt) -> Tuple[Dict, List]:
    output_manager = OutputManager(opt.save_dir, opt.model, opt.name)
    particle_manager = ParticleDataManager()
    monitor = PerformanceMonitor(len(dataset))
    
    mrc_names = ['fake_B']
    png_names = model.visual_names
    output_manager.setup_directories(mrc_names, png_names)
    
    for i, data in enumerate(dataset):
        if monitor.should_print(i):
            monitor.print_progress(i)
        
        start_time = time.time()
        model.set_input(data)
        model.test()
        inference_time = time.time() - start_time
        
        visuals = model.get_current_visuals()
        B, C, H, W = visuals['real_A'].shape
        monitor.update(inference_time, B)
        
        img_name = f'gen_{i:04d}'
        output_manager.save_visuals(visuals, img_name, mrc_names, png_names)
        
        if 'mask_A' in png_names and hasattr(model, 'rotations'):
            fake_B = visuals['fake_B'][0, 0].cpu().numpy()
            clean_real_A = visuals['clean_real_A'][0, 0].cpu().numpy()
            
            particles_added = particle_manager.add_particles(
                img_name, fake_B, clean_real_A,
                model.rotations[0], model.boxes[0], model.centers[0],
                H, W, opt.generate_shift, opt.pixel_shift_max
            )
            
            if particles_added == 0:
                logger.debug(f'No valid particles found in {img_name}')
    
    if 'mask_A' in png_names:
        particle_manager.finalize_and_save(output_manager.save_dir, opt.starfile_upsample)
    
    monitor.print_progress(len(dataset))
    return output_manager.save_dir


def main(args):
    opt = process_opt(args)
    validate_options(opt)
    
    opt.display_id = -1
    
    logger.info("Creating dataset and model...")
    dataset = create_dataset(opt)
    model = create_model(opt)
    model.setup(opt)
    
    if opt.eval:
        model.eval()
    
    logger.info(f"Starting test with {len(dataset)} samples")
    logger.info(f"Model: {opt.model}, Name: {opt.name}")
    logger.info(f"Save directory: {opt.save_dir}")
    
    save_dir = run_inference(model, dataset, opt)
    
    logger.info("=" * 50)
    logger.info("Test completed successfully!")
    logger.info(f"Generated {len(dataset)} synthetic micrographs")
    logger.info(f"Results saved to: {save_dir}")
    logger.info("=" * 50)


if __name__ == "__main__":
    parser = setup_argument_parser()
    args = parser.parse_args()
    main(args)