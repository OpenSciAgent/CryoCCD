import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, Dict, Any

from cryoccd.models.base_model import BaseModel
from cryoccd.models import networks
from cryoccd.micrograph import apply_weight_map_and_normalize
from cryoccd.transform import instance_normalize
from cryoccd import utils, losses

import logging
logger = logging.getLogger(__name__)


class CryoCCDModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(no_dropout=True, no_antialias=True, no_antialias_up=True, pool_size=0)
        
        # NCE Loss parameters
        parser.add_argument('--lambda_NCE', type=float, default=10.0, help='Weight for contrastive loss')
        parser.add_argument('--nce_idt', type=utils.str2bool, nargs='?', const=True, default=False,
                           help='Whether to use identity NCE loss')
        parser.add_argument('--nce_layers', type=str, default='1,2,3,4,5', help='Which layers to compute NCE loss on')
        parser.add_argument('--nce_includes_all_negatives_from_minibatch', type=utils.str2bool, 
                           nargs='?', const=True, default=False, help='Whether to use all negatives from minibatch')
        parser.add_argument('--netF', type=str, default='mask_sample', choices=['mask_sample'],
                           help='Feature projection method')
        parser.add_argument('--netF_nc', type=int, default=256, help='Output channels of the F network')
        parser.add_argument('--nce_T', type=float, default=0.07, help='Temperature for NCE')
        parser.add_argument('--num_patches', type=int, default=256, help='Number of patches to sample per layer')
        parser.add_argument('--flip_equivariance', type=utils.str2bool, nargs='?', const=True, default=False,
                           help='Whether to use flip-equivariance regularization')

        # Diffusion parameters
        parser.add_argument('--beta_1', type=float, default=1e-4, help='Initial beta')
        parser.add_argument('--beta_T', type=float, default=0.02, help='Final beta')

        # GAN parameters
        parser.add_argument('--lambda_GAN', type=float, default=1.0, help='Weight for GAN loss')

        # Window Attention parameters
        parser.add_argument('--use_window_attention', type=utils.str2bool, nargs='?', const=True, default=False,
                           help='Whether to use window attention')
        parser.add_argument('--window_size', type=int, default=8, help='Window size for attention')
        parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads')

        # Cycle consistency
        parser.add_argument('--lambda_cycle', type=float, default=10.0, help='Weight for cycle consistency loss')
                            
        # Sampling parameters
        parser.add_argument('--sampler_type', type=str, default='ddpm',
                           choices=['ddpm', 'ddim', 'dpmsolver', 'dpmsolver++', 'unipc', 'lms', 'heun'],
                           help='Sampler type')
        parser.add_argument('--solver_order', type=int, default=2, choices=[1, 2, 3],
                           help='Solver order (1, 2, or 3), effective for DPM-Solver type samplers')
        parser.add_argument('--use_corrector', type=utils.str2bool, nargs='?', const=True, default=False,
                           help='Whether to use a corrector (for DPM-Solver++)')
        parser.add_argument('--eta', type=float, default=0.0, help='Stochasticity parameter for DDIM')

        return parser

    def __init__(self, opt):
        super().__init__(opt)
        self.opt = opt
        self.nce_layers = [int(i) for i in opt.nce_layers.split(',')]
        self.scaler = torch.cuda.amp.GradScaler()
        
        self._setup_loss_names()
        self._setup_visual_names()
        self._setup_model_names()
        self._initialize_networks()
        
        if self.isTrain:
            self._initialize_losses()
            self._initialize_optimizers()

    def _setup_loss_names(self):
        self.loss_names = ['diff_AB', 'diff_BA', 'NCE', 'GAN', 'cycle_A', 'cycle_B']

    def _setup_visual_names(self):
        if self.isTrain:
            self.visual_names = [
                'real_A', 'clean_real_A', 'weight_map', 'noisy_real_A',
                'fake_B', 'cyc_A', 'real_B', 'fake_A', 'cyc_B', 'mask_A'
            ]
        else:
            self.visual_names = ['real_A', 'clean_real_A', 'weight_map', 'noisy_real_A', 'fake_B', 'mask_A']

    def _setup_model_names(self):
        if self.isTrain:
            self.model_names = ['G_AB', 'G_BA', 'F', 'D']
        else:
            self.model_names = ['G_AB']

    def _initialize_networks(self):
        self.netG_AB = self._create_diffusion_unet()
        
        if self.isTrain:
            self.netG_BA = self._create_diffusion_unet()
            self.netF = self._create_feature_network()
            self.netD = self._create_discriminator()

    def _create_diffusion_unet(self):
        return networks.define_diffusion_unet(
            input_nc=1, output_nc=1, ngf=self.opt.ngf,
            T=self.opt.T, beta_1=self.opt.beta_1, beta_T=self.opt.beta_T,
            norm=self.opt.norm, use_dropout=not self.opt.no_dropout,
            init_type=self.opt.init_type, init_gain=self.opt.init_gain,
            gpu_ids=self.gpu_ids, opt=self.opt,
            no_antialias=self.opt.no_antialias, no_antialias_up=self.opt.no_antialias_up
        )

    def _create_feature_network(self):
        return networks.define_F(
            input_nc=1, netF=self.opt.netF,
            norm=self.opt.norm, use_dropout=not self.opt.no_dropout,
            init_type=self.opt.init_type, init_gain=self.opt.init_gain,
            gpu_ids=self.gpu_ids, opt=self.opt
        )

    def _create_discriminator(self):
        return networks.define_D(
            input_nc=1, ndf=self.opt.ndf,
            netD=self.opt.netD, n_layers_D=self.opt.n_layers_D,
            norm=self.opt.norm, init_type=self.opt.init_type,
            init_gain=self.opt.init_gain, gpu_ids=self.gpu_ids
        )

    def _initialize_losses(self):
        if self.opt.netF == 'mask_sample':
            self.criterionNCE = losses.MaskAwaredPatchNCELoss(self.opt).to(self.device)
        else:
            self.criterionNCE = losses.PatchNCELoss(self.opt).to(self.device)
        
        self.criterionGAN = losses.GANLoss(self.opt.gan_mode).to(self.device)
        self.criterionCycle = torch.nn.L1Loss().to(self.device)

    def _initialize_optimizers(self):
        generator_params = (
            list(self.netG_AB.parameters()) +
            list(self.netG_BA.parameters()) +
            list(self.netF.parameters())
        )
        
        self.optimizer_G = torch.optim.Adam(
            generator_params, lr=self.opt.lr, betas=(self.opt.beta1, 0.999)
        )
        self.optimizer_D = torch.optim.Adam(
            self.netD.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999)
        )
        self.optimizers = [self.optimizer_G, self.optimizer_D]

    def set_input(self, data):
        AtoB = (self.opt.direction == 'AtoB')
        
        self.real_A = data['A' if AtoB else 'B'].to(self.device)
        self.clean_real_A = self.real_A.clone()
        self.weight_map = data['weight_map'].to(self.device)
        
        self.real_A = apply_weight_map_and_normalize(
            self.real_A, self.weight_map, instance_normalize
        )
        
        if self.opt.mask_dir:
            self.mask_A = data['mask_A'].to(self.device)
        
        if self.isTrain:
            self.real_B = data['B' if AtoB else 'A'].to(self.device)
        else:
            self._set_test_metadata(data)

    def _set_test_metadata(self, data):
        if 'rotations' in data:
            self.rotations = data['rotations']
        if 'boxes' in data:
            self.boxes = data['boxes']
        if 'centers' in data:
            self.centers = data['centers']

    def forward(self):
        self._preprocess_input()
        
        if self.isTrain:
            self._forward_training()
        else:
            self._forward_inference()

    def _preprocess_input(self):
        G_AB = self._get_unwrapped_model(self.netG_AB)
        self.noisy_real_A, _ = G_AB.pre_process(
            self.clean_real_A,
            apply_ctf=False,
            apply_gaussian_noise=True,
            snr=self.opt.snr,
            apix=self.opt.apix
        )

    def _forward_training(self):
        self._forward_AB_training()
        self._forward_BA_training()
        self._compute_cycle_consistency()

    def _forward_AB_training(self):
        G_AB = self._get_unwrapped_model(self.netG_AB)
        
        t = torch.randint(0, self.opt.T, (self.real_A.size(0),), device=self.device)
        noise_B = torch.randn_like(self.real_B)
        x_t_B, _ = G_AB.q_sample(self.real_B, t, noise=noise_B)
        
        self.pred_noise_AB = G_AB(x_t_B, t, condition=self.noisy_real_A)
        self.target_noise_B = noise_B
        
        with torch.no_grad():
            self.fake_B = self._sample_from_model(G_AB, self.noisy_real_A)

    def _forward_BA_training(self):
        G_BA = self._get_unwrapped_model(self.netG_BA)
        
        t2 = torch.randint(0, self.opt.T, (self.real_B.size(0),), device=self.device)
        noise_A = torch.randn_like(self.real_A)
        x_t_A, _ = G_BA.q_sample(self.real_A, t2, noise=noise_A)
        
        self.pred_noise_BA = G_BA(x_t_A, t2, condition=self.real_B)
        self.target_noise_A = noise_A
        
        with torch.no_grad():
            self.fake_A = self._sample_from_model(G_BA, self.real_B)

    def _compute_cycle_consistency(self):
        G_AB = self._get_unwrapped_model(self.netG_AB)
        G_BA = self._get_unwrapped_model(self.netG_BA)
        
        self.cyc_A = self._sample_from_model(G_BA, self.fake_B)
        self.cyc_B = self._sample_from_model(G_AB, self.fake_A)

    def _forward_inference(self):
        G_AB = self._get_unwrapped_model(self.netG_AB)
        with torch.no_grad():
            self.fake_B = self._sample_from_model(G_AB, self.noisy_real_A)

    def _sample_from_model(self, model, condition):
        return model.sample(
            condition,
            condition=condition,
            steps=min(100, self.opt.sampling_steps),
            sampler_type=self.opt.sampler_type,
            solver_order=self.opt.solver_order,
            use_corrector=self.opt.use_corrector,
            eta=getattr(self.opt, 'eta', 0.0)
        )

    def _get_unwrapped_model(self, model):
        return model.module if hasattr(model, 'module') else model

    def test(self):
        with torch.no_grad():
            self.forward()

    def optimize_parameters(self):
        self.forward()
        self._optimize_discriminator()
        self._optimize_generator()

    def _optimize_discriminator(self):
        self.optimizer_D.zero_grad()
        
        loss_D_real = self.criterionGAN(self.netD(self.real_B), True)
        loss_D_fake = self.criterionGAN(self.netD(self.fake_B.detach()), False)
        self.loss_GAN = 0.5 * (loss_D_real + loss_D_fake)
        
        self.scaler.scale(self.loss_GAN).backward()
        self.scaler.step(self.optimizer_D)

    def _optimize_generator(self):
        self.optimizer_G.zero_grad()
        
        self._compute_generator_losses()
        
        self.scaler.scale(self.loss_G).backward()
        self.scaler.step(self.optimizer_G)
        self.scaler.update()

    def _compute_generator_losses(self):
        # Diffusion losses
        self.loss_diff_AB = F.mse_loss(self.pred_noise_AB, self.target_noise_B)
        self.loss_diff_BA = F.mse_loss(self.pred_noise_BA, self.target_noise_A)
        
        # NCE loss
        self.loss_NCE = self._compute_nce_loss() if self.opt.lambda_NCE > 0 else 0.0
        
        # Cycle consistency losses
        self.loss_cycle_A = self.criterionCycle(self.cyc_A, self.clean_real_A)
        self.loss_cycle_B = self.criterionCycle(self.cyc_B, self.real_B)
        
        # GAN loss for generator
        loss_GAN_G = self.criterionGAN(self.netD(self.fake_B), True)
        
        # Combined loss
        self.loss_G = (
            self.loss_diff_AB + self.loss_diff_BA +
            self.opt.lambda_NCE * self.loss_NCE +
            self.opt.lambda_GAN * loss_GAN_G +
            self.opt.lambda_cycle * (self.loss_cycle_A + self.loss_cycle_B)
        )

    def _compute_nce_loss(self):
        masks = getattr(self, 'mask_A', None)
        return self.calculate_NCE_loss(self.noisy_real_A, self.fake_B, masks=masks)

    def calculate_NCE_loss(self, src, tgt, masks=None):
        G_AB = self._get_unwrapped_model(self.netG_AB)
        
        feat_q = G_AB.extract_features(tgt, self.nce_layers, condition=None)
        feat_k = G_AB.extract_features(src, self.nce_layers, condition=None)
        
        if self.opt.netF == 'mask_sample':
            return self._compute_mask_aware_nce(feat_q, feat_k, masks)
        else:
            return self._compute_standard_nce(feat_q, feat_k)

    def _compute_mask_aware_nce(self, feat_q, feat_k, masks):
        pos_k, neg_k, pgrid, ngrid = self.netF(
            feat_k, num_patches=self.opt.num_patches, masks=masks
        )
        pos_q, neg_q, _, _ = self.netF(
            feat_q, num_patches=self.opt.num_patches,
            pos_grids=pgrid, neg_grids=ngrid, masks=masks
        )
        
        total_loss = 0.0
        for fq_pos, fq_neg, fk_pos, fk_neg in zip(pos_q, neg_q, pos_k, neg_k):
            l1 = self.criterionNCE(fq_pos, fq_neg, fk_pos).mean()
            l2 = self.criterionNCE(fq_neg, fq_pos, fk_neg).mean()
            total_loss += 0.5 * (l1 + l2)
        
        return total_loss / len(feat_k)

    def _compute_standard_nce(self, feat_q, feat_k):
        pool_k, ids = self.netF(feat_k, self.opt.num_patches, None)
        pool_q, _ = self.netF(feat_q, self.opt.num_patches, ids)
        
        total_loss = 0.0
        for fq, fk in zip(pool_q, pool_k):
            total_loss += self.criterionNCE(fk, fq).mean()
        
        return total_loss / len(feat_k)