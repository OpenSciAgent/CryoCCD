import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, Dict, Any

from cryoccd.models.base_model import BaseModel
from cryoccd.models import networks
from cryoccd.models.networks import extract
from cryoccd.micrograph import apply_weight_map_and_normalize
from cryoccd.transform import instance_normalize
from cryoccd import utils, losses

import logging
logger = logging.getLogger(__name__)


class CryoCCDModel(BaseModel):
    """Cycle-consistent predictor-corrector diffusion for unpaired A (simulated) -> B (real) translation.

    Training follows the paper (Sec. 4 and Appendix "Implementation Details"):
      * each iteration draws a single t ~ U{1..T}, forms x_t by the forward process, evaluates each
        noise predictor once and recovers x0_hat = (x_t - sigma_t * eps) / alpha_t in closed form;
      * all losses act on x0_hat; there is no explicit score-matching (denoising MSE) loss and the
        sampler is never unrolled during training;
      * L = gamma_adv * L_adv + gamma_cyc * L_cyc + gamma_NCE * L_NCE with (1, 3, 5);
      * a single unconditional least-squares PatchGAN D on the A -> B direction.
    The multi-step UniPC sampler is used only at inference.
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(no_dropout=True, no_antialias=True, no_antialias_up=True, pool_size=0)

        # NCE Loss parameters
        parser.add_argument('--lambda_NCE', type=float, default=5.0, help='Weight for contrastive loss (gamma_NCE)')
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
        parser.add_argument('--lambda_GAN', type=float, default=1.0, help='Weight for GAN loss (gamma_adv)')

        # Window Attention parameters
        parser.add_argument('--use_window_attention', type=utils.str2bool, nargs='?', const=True, default=False,
                           help='Whether to use window attention')
        parser.add_argument('--window_size', type=int, default=8, help='Window size for attention')
        parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads')

        # Cycle consistency
        parser.add_argument('--lambda_cycle', type=float, default=3.0, help='Weight for cycle consistency loss (gamma_cyc)')

        # Sampling parameters (inference only)
        parser.add_argument('--sampler_type', type=str, default='unipc',
                           choices=['ddpm', 'ddim', 'dpmsolver', 'dpmsolver++', 'unipc', 'lms', 'heun'],
                           help='Sampler type')
        parser.add_argument('--solver_order', type=int, default=2, choices=[1, 2, 3],
                           help='Solver order (1, 2, or 3), effective for DPM-Solver type samplers')
        parser.add_argument('--use_corrector', type=utils.str2bool, nargs='?', const=True, default=True,
                           help='Whether to use the corrector step (UniC for UniPC, also DPM-Solver++)')
        parser.add_argument('--eta', type=float, default=0.0, help='Stochasticity parameter for DDIM')

        return parser

    def __init__(self, opt):
        super().__init__(opt)
        self.opt = opt
        self.T = getattr(opt, 'T', 1000)
        self.nce_layers = [int(i) for i in opt.nce_layers.split(',')]
        self.scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

        self._setup_loss_names()
        self._setup_visual_names()
        self._setup_model_names()
        self._initialize_networks()

        if self.isTrain:
            self._initialize_losses()
            self._initialize_optimizers()

    def _setup_loss_names(self):
        self.loss_names = ['G_adv', 'cycle_A', 'cycle_B', 'NCE', 'D']

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
        # condition = [source image, mask]; G_BA and the closing pass of the B cycle use an empty mask
        return networks.define_diffusion_unet(
            input_nc=1, output_nc=1, ngf=self.opt.ngf,
            T=self.T, beta_1=self.opt.beta_1, beta_T=self.opt.beta_T,
            norm=self.opt.norm, use_dropout=not self.opt.no_dropout,
            init_type=self.opt.init_type, init_gain=self.opt.init_gain,
            gpu_ids=self.gpu_ids, opt=self.opt,
            no_antialias=self.opt.no_antialias, no_antialias_up=self.opt.no_antialias_up,
            cond_nc=2
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
        # netF's MLP heads are created lazily on the first forward pass;
        # data_dependent_initialize() adds them to optimizer_G afterwards.
        generator_params = (
            list(self.netG_AB.parameters()) +
            list(self.netG_BA.parameters())
        )

        self.optimizer_G = torch.optim.Adam(
            generator_params, lr=self.opt.lr, betas=(self.opt.beta1, 0.999)
        )
        self.optimizer_D = torch.optim.Adam(
            self.netD.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999)
        )
        self.optimizers = [self.optimizer_G, self.optimizer_D]

    def data_dependent_initialize(self, data):
        """Build netF's projection heads (their size depends on the feature maps) and register them with optimizer_G."""
        self.set_input(data)
        if not self.isTrain or self.opt.lambda_NCE <= 0:
            return
        with torch.no_grad():
            self.forward()
            self._compute_nce_loss()
        f_params = list(self.netF.parameters())
        if f_params:
            self.optimizer_G.add_param_group({'params': f_params})

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

    def _condition(self, src, mask=None):
        """Condition for eps: [source image, mask]; mask=None is the empty mask (all zeros)."""
        if mask is None:
            mask = torch.zeros_like(src)
        return torch.cat([src, mask.to(src.dtype)], dim=1)

    def _translate(self, G, src, mask, t, noise):
        """One network evaluation: x_t = q(x_t | src), x0_hat = (x_t - sigma_t * eps) / alpha_t."""
        model = self._get_unwrapped_model(G)
        x_t, _ = model.q_sample(src, t, noise=noise)
        eps = G(x_t, t, condition=self._condition(src, mask))
        alpha_t = extract(model.sqrt_alphas_cumprod, t, src.shape)
        sigma_t = extract(model.sqrt_one_minus_alphas_cumprod, t, src.shape)
        return (x_t - sigma_t * eps) / alpha_t

    def _forward_training(self):
        mask_A = getattr(self, 'mask_A', None)
        B = self.noisy_real_A.size(0)

        # single t per iteration, shared by both cycles; reusing the same noise makes each
        # cycle's intermediate step deterministic so the reconstruction is well defined
        t = torch.randint(0, self.T, (B,), device=self.device)
        noise_A = torch.randn_like(self.noisy_real_A)
        noise_B = torch.randn_like(self.real_B)

        # A -> B -> A  (mask-conditioned G_AB, unconditional G_BA)
        self.fake_B = self._translate(self.netG_AB, self.noisy_real_A, mask_A, t, noise_A)
        self.cyc_A = self._translate(self.netG_BA, self.fake_B, None, t, noise_A)

        # B -> A -> B  (real micrographs carry no mask: G_AB closes the cycle with the empty mask)
        self.fake_A = self._translate(self.netG_BA, self.real_B, None, t, noise_B)
        self.cyc_B = self._translate(self.netG_AB, self.fake_A, None, t, noise_B)

    def _forward_inference(self):
        G_AB = self._get_unwrapped_model(self.netG_AB)
        src = self.noisy_real_A
        cond = self._condition(src, getattr(self, 'mask_A', None))
        with torch.no_grad():
            t_T = torch.full((src.size(0),), self.T - 1, dtype=torch.long, device=self.device)
            x_T, _ = G_AB.q_sample(src, t_T)
            self.fake_B = G_AB.sample(
                x_T,
                condition=cond,
                steps=getattr(self.opt, 'sampling_steps', 20),
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
        # (iv) one Adam step on {eps_theta, eps_phi} and the NCE projection head
        self._optimize_generator()
        # (v) one Adam step on D, with x0_hat detached
        self._optimize_discriminator()
        self.scaler.update()

    def _optimize_generator(self):
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()

        self._compute_generator_losses()

        self.scaler.scale(self.loss_G).backward()
        self.scaler.step(self.optimizer_G)

    def _optimize_discriminator(self):
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()

        # L_D = 1/2 E[(D(y) - 1)^2] + 1/2 E[D(x0_hat)^2]   (least-squares GAN)
        loss_D_real = self.criterionGAN(self.netD(self.real_B), True)
        loss_D_fake = self.criterionGAN(self.netD(self.fake_B.detach()), False)
        self.loss_D = 0.5 * (loss_D_real + loss_D_fake)

        self.scaler.scale(self.loss_D).backward()
        self.scaler.step(self.optimizer_D)

    def _compute_generator_losses(self):
        # L_adv = E[(D(x0_hat) - 1)^2]
        self.loss_G_adv = self.criterionGAN(self.netD(self.fake_B), True)

        # L_cyc = E||G_BA(G_AB(x, m)) - x||_1 + E||G_AB(G_BA(y), empty) - y||_1
        self.loss_cycle_A = self.criterionCycle(self.cyc_A, self.noisy_real_A)
        self.loss_cycle_B = self.criterionCycle(self.cyc_B, self.real_B)

        # L_NCE (mask-guided)
        self.loss_NCE = self._compute_nce_loss() if self.opt.lambda_NCE > 0 else 0.0

        self.loss_G = (
            self.opt.lambda_GAN * self.loss_G_adv +
            self.opt.lambda_cycle * (self.loss_cycle_A + self.loss_cycle_B) +
            self.opt.lambda_NCE * self.loss_NCE
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
        # source features: particle (mask) locations and background (~mask) locations
        pos_k, neg_k, pgrid, ngrid = self.netF(
            feat_k, num_patches=self.opt.num_patches, masks=masks
        )
        # translated-image features at the same locations
        pos_q, neg_q, _, _ = self.netF(
            feat_q, num_patches=self.opt.num_patches,
            pos_grids=pgrid, neg_grids=ngrid, masks=masks
        )

        total_loss = 0.0
        for fq_pos, fq_neg, fk_pos, fk_neg in zip(pos_q, neg_q, pos_k, neg_k):
            # particle queries: positive = same particle patch, negatives = background patches
            l1 = self.criterionNCE(fq_pos, fk_pos, fk_neg).mean()
            # background queries: positive = same background patch, negatives = particle patches
            l2 = self.criterionNCE(fq_neg, fk_neg, fk_pos).mean()
            total_loss += 0.5 * (l1 + l2)

        return total_loss / len(feat_k)

    def _compute_standard_nce(self, feat_q, feat_k):
        pool_k, ids = self.netF(feat_k, self.opt.num_patches, None)
        pool_q, _ = self.netF(feat_q, self.opt.num_patches, ids)

        total_loss = 0.0
        for fq, fk in zip(pool_q, pool_k):
            total_loss += self.criterionNCE(fk, fq).mean()

        return total_loss / len(feat_k)
