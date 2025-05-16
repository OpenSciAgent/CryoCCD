import numpy as np
import torch
import torch.nn.functional as F
from cryoccd.models.base_model import BaseModel
from cryoccd.models import networks
from cryoccd.micrograph import apply_weight_map_and_normalize
from cryoccd.transform import instance_normalize
from cryoccd import utils, losses
from cryoccd.models.networks import define_D

import logging
logger = logging.getLogger(__name__)

class CryoCCDModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(no_dropout=True, no_antialias=True, no_antialias_up=True)
        
        parser.add_argument('--lambda_NCE', type=float, default=10.0,
                            help='Weight for contrastive loss')
        parser.add_argument('--nce_idt', type=utils.str2bool, nargs='?', const=True, default=False,
                            help='Whether to use identity NCE loss')
        parser.add_argument('--nce_layers', type=str, default='1,2,3,4,5',
                            help='Which layers to compute NCE loss on')
        parser.add_argument('--nce_includes_all_negatives_from_minibatch',
                            type=utils.str2bool, nargs='?', const=True, default=False,
                            help='Whether to use all negatives from minibatch')
        parser.add_argument('--netF', type=str, default='mask_sample',
                            choices=['sample','reshape','mlp_sample','mask_sample'],
                            help='Feature projection method')
        parser.add_argument('--netF_nc', type=int, default=256,
                            help='Output channels of the F network')
        parser.add_argument('--nce_T', type=float, default=0.07,
                            help='Temperature for NCE')
        parser.add_argument('--num_patches', type=int, default=256,
                            help='Number of patches to sample per layer')
        parser.add_argument('--flip_equivariance', type=utils.str2bool, nargs='?', const=True, default=False,
                            help='Whether to use flip-equivariance regularization')

        # —— Diffusion —— 
        parser.add_argument('--beta_1', type=float, default=1e-4,
                            help='Initial beta')
        parser.add_argument('--beta_T', type=float, default=0.02,
                            help='Final beta')

        # —— GAN —— 
        parser.add_argument('--lambda_GAN', type=float, default=1.0,
                            help='Weight for GAN loss')

        # —— Window Attention —— 
        parser.add_argument('--use_window_attention', type=utils.str2bool, nargs='?', const=True, default=False,
                            help='Whether to use window attention')
        parser.add_argument('--window_size', type=int, default=8,
                            help='Window size for attention')
        parser.add_argument('--num_heads', type=int, default=4,
                            help='Number of attention heads')

        # —— Cycle consistency —— 
        parser.add_argument('--lambda_cycle', type=float, default=10.0,
                            help='Weight for cycle consistency loss')
                            
        parser.add_argument('--sampler_type', type=str, default='ddpm',
                            choices=['ddpm', 'ddim', 'dpmsolver', 'dpmsolver++', 'unipc'],
                            help='Sampler type')
        parser.add_argument('--solver_order', type=int, default=2,
                            choices=[1, 2, 3],
                            help='Solver order (1, 2, or 3), effective for DPM-Solver type samplers')
        parser.add_argument('--use_corrector', type=utils.str2bool, nargs='?', const=True, default=False,
                            help='Whether to use a corrector (for DPM-Solver++)')

        # Disable image pool
        parser.set_defaults(pool_size=0)
        return parser


    def __init__(self, opt):
        super().__init__(opt)
        self.nce_layers = [int(i) for i in opt.nce_layers.split(',')]
        self.loss_names = [
            'diff_AB', 'diff_BA', 
            'NCE', 'GAN', 
            'cycle_A', 'cycle_B'
        ]
        self.scaler = torch.cuda.amp.GradScaler()
        if self.isTrain:
            self.visual_names = [
                'real_A','clean_real_A','weight_map','noisy_real_A',
                'fake_B','cyc_A',
                'real_B','fake_A','cyc_B','mask_A'
            ]
            self.model_names = ['G_AB','G_BA','F','D']
        else:
            self.visual_names = ['real_A', 'clean_real_A', 'weight_map', 'noisy_real_A', 'fake_B','mask_A']
            self.model_names = ['G_AB']

        # —— A→B UNet —— 
        self.netG_AB = networks.define_diffusion_unet(
            input_nc=1, output_nc=1, ngf=opt.ngf,
            T=opt.T, beta_1=opt.beta_1, beta_T=opt.beta_T,
            norm=opt.norm, use_dropout=not opt.no_dropout,
            init_type=opt.init_type, init_gain=opt.init_gain,
            gpu_ids=self.gpu_ids, opt=opt,
            no_antialias=opt.no_antialias, no_antialias_up=opt.no_antialias_up
        )
        if self.isTrain:
            # —— B→A UNet —— 
            self.netG_BA = networks.define_diffusion_unet(
                input_nc=1, output_nc=1, ngf=opt.ngf,
                T=opt.T, beta_1=opt.beta_1, beta_T=opt.beta_T,
                norm=opt.norm, use_dropout=not opt.no_dropout,
                init_type=opt.init_type, init_gain=opt.init_gain,
                gpu_ids=self.gpu_ids, opt=opt,
                no_antialias=opt.no_antialias, no_antialias_up=opt.no_antialias_up
            )
            self.netF = networks.define_F(
                input_nc=1, netF=opt.netF,
                norm=opt.norm, use_dropout=not opt.no_dropout,
                init_type=opt.init_type, init_gain=opt.init_gain,
                gpu_ids=self.gpu_ids, opt=opt
            )
            self.criterionNCE = (
                losses.MaskAwaredPatchNCELoss(opt).to(self.device)
                if opt.netF=='mask_sample'
                else losses.PatchNCELoss(opt).to(self.device)
            )
            self.netD = define_D(
                input_nc=1, ndf=opt.ndf,
                netD=opt.netD, n_layers_D=opt.n_layers_D,
                norm=opt.norm
            ).to(self.device)
            self.criterionGAN = losses.GANLoss(opt.gan_mode).to(self.device)
            self.criterionCycle = torch.nn.L1Loss().to(self.device)

            self.optimizer_G = torch.optim.Adam(
                list(self.netG_AB.parameters()) +
                list(self.netG_BA.parameters()) +
                list(self.netF.parameters()),
                lr=opt.lr, betas=(opt.beta1, 0.999)
            )
            self.optimizer_D = torch.optim.Adam(
                self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999)
            )
            self.optimizers = [self.optimizer_G, self.optimizer_D]

    def set_input(self, data):
        AtoB = (self.opt.direction=='AtoB')
        self.real_A = data['A' if AtoB else 'B'].to(self.device)
        self.clean_real_A = self.real_A
        self.weight_map = data['weight_map'].to(self.device)
        self.real_A = apply_weight_map_and_normalize(
            self.real_A, self.weight_map, instance_normalize
        )
        if self.opt.mask_dir:
            self.mask_A = data['mask_A'].to(self.device)
        
        if self.isTrain:
            self.real_B = data['B' if AtoB else 'A'].to(self.device)
        else:
            if 'rotations' in data:
                self.rotations = data['rotations']  
            if 'boxes' in data:
                self.boxes = data['boxes']
            if 'centers' in data:
                self.centers = data['centers']
    
    def forward(self):
        # —— A→B diffusion & sample —— 
        G_AB = self.netG_AB.module if hasattr(self.netG_AB,'module') else self.netG_AB
        self.noisy_real_A, _ = G_AB.pre_process(
            self.clean_real_A,
            apply_ctf=False,
            apply_gaussian_noise=True,
            snr=self.opt.snr,
            apix=self.opt.apix
        )
        
        if self.isTrain:
            t = torch.randint(0, self.opt.T, (self.real_A.size(0),), device=self.device)
            noise_B = torch.randn_like(self.real_B)
            x_t_B, _ = G_AB.q_sample(self.real_B, t, noise=noise_B)
            self.pred_noise_AB = G_AB(x_t_B, t, condition=self.noisy_real_A)
            self.target_noise_B = noise_B
            
            with torch.no_grad():
                self.fake_B = G_AB.sample(
                    self.noisy_real_A,
                    condition=self.noisy_real_A,
                    steps=min(100, self.opt.sampling_steps),
                    sampler_type=self.opt.sampler_type,
                    solver_order=self.opt.solver_order,
                    use_corrector=self.opt.use_corrector
                )
        
            # —— B→A diffusion & sample —— 
            G_BA = self.netG_BA.module if hasattr(self.netG_BA,'module') else self.netG_BA
            t2 = torch.randint(0, self.opt.T, (self.real_B.size(0),), device=self.device)
            noise_A = torch.randn_like(self.real_A)
            x_t_A, _ = G_BA.q_sample(self.real_A, t2, noise=noise_A)
            self.pred_noise_BA = G_BA(x_t_A, t2, condition=self.real_B)
            self.target_noise_A = noise_A
            
            with torch.no_grad():
                self.fake_A = G_BA.sample(
                    self.real_B,
                    condition=self.real_B,
                    steps=min(100, self.opt.sampling_steps),
                    sampler_type=self.opt.sampler_type,
                    solver_order=self.opt.solver_order,
                    use_corrector=self.opt.use_corrector
                )
        
            self.cyc_A = G_BA.sample(
                self.fake_B,
                condition=self.fake_B,
                steps=min(100, self.opt.sampling_steps),
                sampler_type=self.opt.sampler_type,
                solver_order=self.opt.solver_order,
                use_corrector=self.opt.use_corrector
            )
            
            self.cyc_B = G_AB.sample(
                self.fake_A,
                condition=self.fake_A,
                steps=min(100, self.opt.sampling_steps),
                sampler_type=self.opt.sampler_type,
                solver_order=self.opt.solver_order,
                use_corrector=self.opt.use_corrector
            )
        else:
            with torch.no_grad():
                self.fake_B = G_AB.sample(
                    self.noisy_real_A,
                    condition=self.noisy_real_A,
                    steps=min(100, self.opt.sampling_steps),
                    sampler_type=self.opt.sampler_type,
                    solver_order=self.opt.solver_order,
                    use_corrector=self.opt.use_corrector
                )
    
    def test(self):
        with torch.no_grad():
            self.forward()


    def optimize_parameters(self):
        self.forward()

        self.optimizer_D.zero_grad()
        loss_D_real = self.criterionGAN(self.netD(self.real_B), True)
        loss_D_fake = self.criterionGAN(self.netD(self.fake_B.detach()), False)
        self.loss_GAN = 0.5*(loss_D_real+loss_D_fake)
        # self.loss_GAN.backward()
        # self.optimizer_D.step()
        self.scaler.scale(self.loss_GAN).backward()
        self.scaler.step(self.optimizer_D)

        self.optimizer_G.zero_grad()
        # (1) diffusion losses
        self.loss_diff_AB = F.mse_loss(self.pred_noise_AB, self.target_noise_B)
        self.loss_diff_BA = F.mse_loss(self.pred_noise_BA, self.target_noise_A)
        # (2) NCE on A→B
        self.loss_NCE = self.calculate_NCE_loss(
            self.noisy_real_A, self.fake_B, masks=getattr(self,'mask_A',None)
        ) if self.opt.lambda_NCE>0 else 0.0
        # (3) cycle losses
        self.loss_cycle_A = self.criterionCycle(self.cyc_A, self.clean_real_A)
        self.loss_cycle_B = self.criterionCycle(self.cyc_B, self.real_B)
        # (4) GAN for G
        loss_GAN_G = self.criterionGAN(self.netD(self.fake_B), True)
        # 总 loss
        self.loss_G = (
            self.loss_diff_AB + self.loss_diff_BA
            + self.opt.lambda_NCE*self.loss_NCE
            + self.opt.lambda_GAN*loss_GAN_G
            + self.opt.lambda_cycle*(self.loss_cycle_A+self.loss_cycle_B)
        )
        # self.loss_G.backward()
        # self.optimizer_G.step()
        self.scaler.scale(self.loss_G).backward()
        self.scaler.step(self.optimizer_G)
        self.scaler.update()

    def calculate_NCE_loss(self, src, tgt, masks=None):
        G_AB = self.netG_AB.module if hasattr(self.netG_AB,'module') else self.netG_AB
        feat_q = G_AB.extract_features(tgt, self.nce_layers, condition=None)
        feat_k = G_AB.extract_features(src, self.nce_layers, condition=None)
        total = 0.0
        if self.opt.netF=='mask_sample':
            pos_k, neg_k, pgrid, ngrid = self.netF(
                feat_k, num_patches=self.opt.num_patches, masks=masks)
            pos_q, neg_q, _, _ = self.netF(
                feat_q, num_patches=self.opt.num_patches,
                pos_grids=pgrid, neg_grids=ngrid, masks=masks)
            for fq_pos,fq_neg,fk_pos,fk_neg in zip(pos_q,neg_q,pos_k,neg_k):
                l1 = self.criterionNCE(fq_pos,fq_neg,fk_pos).mean()
                l2 = self.criterionNCE(fq_neg,fq_pos,fk_neg).mean()
                total += 0.5*(l1+l2)
        else:
            pool_k, ids = self.netF(feat_k,self.opt.num_patches,None)
            pool_q, _   = self.netF(feat_q,self.opt.num_patches,ids)
            for fq,fk in zip(pool_q,pool_k):
                total += self.criterionNCE(fk,fq).mean()
        return total/len(feat_k)

