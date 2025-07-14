"""
CryoCCD Networks Module
======================

This module contains neural network architectures for cryo-electron microscopy 
image generation and processing, including diffusion models, discriminators, 
and feature extractors.
"""

import math
import logging
import functools
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.optim import lr_scheduler
import torch.nn.utils.spectral_norm as spectral_norm

from cryoccd import ctf as ctf_func
from cryoccd.transform import instance_normalize


# Configure logging
logger = logging.getLogger(__name__)


# ====================
# Network Factory Functions
# ====================

def define_F(input_nc, netF, norm='batch', use_dropout=False, 
             init_type='normal', init_gain=0.02, gpu_ids=[], opt=None):
    """Factory function for creating feature extraction networks."""
    if netF == 'mask_sample':
        net = MaskInformedSampleF(
            use_mlp=True, 
            init_type=init_type, 
            init_gain=init_gain, 
            gpu_ids=gpu_ids, 
            nc=opt.netF_nc
        )
    else:
        raise NotImplementedError(f'Projection model name [{netF}] is not recognized')
    
    return init_net(net, init_type, init_gain, gpu_ids)


def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch', 
             init_type='normal', init_gain=0.02, gpu_ids=[], opt=None, 
             no_antialias=True):
    """Factory function for creating discriminator networks."""
    norm_layer = get_norm_layer(norm_type=norm)
    
    if netD == 'basic':
        net = NLayerDiscriminator(
            input_nc, ndf, n_layers=3, 
            norm_layer=norm_layer, no_antialias=no_antialias
        )
    elif netD == 'n_layers':
        net = NLayerDiscriminator(
            input_nc, ndf, n_layers_D, 
            norm_layer=norm_layer, no_antialias=no_antialias
        )
    else:
        raise NotImplementedError(f'Discriminator model name [{netD}] is not recognized')
    
    return init_net(net, init_type, init_gain, gpu_ids,
                    initialize_weights=('stylegan2' not in netD))


def define_diffusion_unet(input_nc, output_nc, ngf, T, beta_1, beta_T, 
                          norm='instance', use_dropout=False, init_type='normal', 
                          init_gain=0.02, gpu_ids=[], opt=None, no_antialias=False, 
                          no_antialias_up=False):
    """Factory function for creating diffusion UNet."""
    norm_layer = get_norm_layer(norm_type=norm)
    net = DiffusionUNet(
        input_nc, output_nc, ngf, T, beta_1, beta_T, norm_layer, 
        use_dropout, no_antialias, no_antialias_up
    )
    return init_net(net, init_type, init_gain, gpu_ids, opt)


# ====================
# Utility Functions
# ====================

def get_norm_layer(norm_type='instance'):
    """Return a normalization layer."""
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x):
            return Identity()
    else:
        raise NotImplementedError(f'Normalization layer [{norm_type}] is not found')
    return norm_layer


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler."""
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l 
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError(f'Learning rate policy [{opt.lr_policy}] is not implemented')
    return scheduler


def get_filter(filt_size=3):
    """Get anti-aliasing filter."""
    if filt_size == 1:
        a = np.array([1.])
    elif filt_size == 2:
        a = np.array([1., 1.])
    elif filt_size == 3:
        a = np.array([1., 2., 1.])
    elif filt_size == 4:
        a = np.array([1., 3., 3., 1.])
    elif filt_size == 5:
        a = np.array([1., 4., 6., 4., 1.])
    elif filt_size == 6:
        a = np.array([1., 5., 10., 10., 5., 1.])
    elif filt_size == 7:
        a = np.array([1., 6., 15., 20., 15., 6., 1.])
    else:
        raise ValueError(f"Unsupported filter size: {filt_size}")

    filt = torch.Tensor(a[:, None] * a[None, :])
    filt = filt / torch.sum(filt)
    return filt


def get_pad_layer(pad_type):
    """Get padding layer."""
    if pad_type in ['refl', 'reflect']:
        PadLayer = nn.ReflectionPad2d
    elif pad_type in ['repl', 'replicate']:
        PadLayer = nn.ReplicationPad2d
    elif pad_type == 'zero':
        PadLayer = nn.ZeroPad2d
    else:
        logger.warning(f'Pad type [{pad_type}] not recognized')
        PadLayer = nn.ZeroPad2d
    return PadLayer


def init_weights(net, init_type='normal', init_gain=0.02, debug=False):
    """Initialize network weights."""
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if debug:
                logger.debug(classname)
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(f'Initialization method [{init_type}] is not implemented')
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    net.apply(init_func)


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[], 
             debug=False, initialize_weights=True):
    """Initialize a network: 1. register CPU/GPU device; 2. initialize the network weights."""
    if len(gpu_ids) > 0:
        assert torch.cuda.is_available()
        net.to(gpu_ids[0])
    
    if initialize_weights:
        init_weights(net, init_type, init_gain=init_gain, debug=debug)
    return net


def apply_physics(img, *, apply_ctf=True, apply_gaussian_noise=True, snr=0.1, apix=1.0):
    """Apply physical effects (CTF and noise) to images."""
    B, _, H, _ = img.shape
    noise = None

    if apply_ctf:
        ctf_params = ctf_func.generate_random_ctf_params(B)
        freqs_mag, ang_rad = ctf_func.compute_safe_freqs(H, apix)
        ctf = ctf_func.compute_ctf(freqs_mag, ang_rad, *ctf_params).reshape(B, 1, H, H)
        ctf = torch.as_tensor(ctf, dtype=img.dtype, device=img.device)

        fourier = ctf_func.torch_fft2_center(img)
        img = ctf_func.torch_ifft2_center(ctf * fourier).real

    if apply_gaussian_noise:
        # σ = sqrt(var(signal) / SNR)
        noise_std = torch.sqrt(torch.var(img, dim=(-2, -1), keepdims=True) / snr)
        noise = torch.randn_like(img) * noise_std.repeat(1, 1, H, H)
        img = img + noise

    return img, noise


def extract(a: torch.Tensor, t: torch.LongTensor, x_shape):
    """Extract values from tensor a at indices t for diffusion models."""
    t = t.to(a.device).long()
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.view(batch_size, *([1] * (len(x_shape) - 1)))


# ====================
# Basic Modules
# ====================

class Identity(nn.Module):
    """Identity layer."""
    def forward(self, x):
        return x


class Normalize(nn.Module):
    """L2 normalization layer."""
    def __init__(self, power=2):
        super().__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm + 1e-7)
        return out


class Downsample(nn.Module):
    """Anti-aliased downsampling layer."""
    def __init__(self, channels, pad_type='reflect', filt_size=3, stride=2, pad_off=0):
        super().__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.stride = stride
        self.off = int((self.stride - 1) / 2.)
        self.channels = channels

        # Calculate padding
        self.pad_sizes = [
            int(1. * (filt_size - 1) / 2), 
            int(np.ceil(1. * (filt_size - 1) / 2)),
            int(1. * (filt_size - 1) / 2), 
            int(np.ceil(1. * (filt_size - 1) / 2))
        ]
        self.pad_sizes = [pad_size + pad_off for pad_size in self.pad_sizes]

        # Register filter
        filt = get_filter(filt_size=self.filt_size)
        self.register_buffer('filt', filt[None, None, :, :].repeat((self.channels, 1, 1, 1)))

        # Padding layer
        self.pad = get_pad_layer(pad_type)(self.pad_sizes)

    def forward(self, inp):
        if self.filt_size == 1:
            if self.pad_off == 0:
                return inp[:, :, ::self.stride, ::self.stride]
            else:
                return self.pad(inp)[:, :, ::self.stride, ::self.stride]
        else:
            return F.conv2d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])


class Upsample(nn.Module):
    """Anti-aliased upsampling layer."""
    def __init__(self, channels, pad_type='repl', filt_size=4, stride=2):
        super().__init__()
        self.filt_size = filt_size
        self.filt_odd = np.mod(filt_size, 2) == 1
        self.pad_size = int((filt_size - 1) / 2)
        self.stride = stride
        self.off = int((self.stride - 1) / 2.)
        self.channels = channels

        # Register filter
        filt = get_filter(filt_size=self.filt_size) * (stride**2)
        self.register_buffer('filt', filt[None, None, :, :].repeat((self.channels, 1, 1, 1)))

        # Padding layer
        self.pad = get_pad_layer(pad_type)([1, 1, 1, 1])
        
    def forward(self, inp):
        ret_val = F.conv_transpose2d(
            self.pad(inp), self.filt, stride=self.stride, 
            padding=1 + self.pad_size, groups=inp.shape[1]
        )[:, :, 1:, 1:]
        
        if self.filt_odd:
            return ret_val
        else:
            return ret_val[:, :, :-1, :-1]


# ====================
# Feature Extraction Networks
# ====================

class MaskInformedSampleF(nn.Module):
    """Mask-informed feature sampling network for contrastive learning."""
    
    def __init__(self, use_mlp=False, init_type='normal', init_gain=0.02, nc=256, gpu_ids=[]):
        super().__init__()
        self.l2norm = Normalize(2)
        self.use_mlp = use_mlp
        self.nc = nc
        self.mlp_init = False
        self.init_type = init_type
        self.init_gain = init_gain
        self.gpu_ids = gpu_ids   

    def create_mlp(self, feats):
        """Create MLP layers for feature processing."""
        for mlp_id, feat in enumerate(feats):
            input_nc = feat.shape[1]
            mlp = nn.Sequential(
                nn.Linear(input_nc, self.nc), 
                nn.ReLU(), 
                nn.Linear(self.nc, self.nc)
            )
            if len(self.gpu_ids) > 0:
                mlp.cuda()
            setattr(self, f'mlp_{mlp_id}', mlp)
        
        init_net(self, self.init_type, self.init_gain, self.gpu_ids)
        self.mlp_init = True
        
    def custom_forward(self, feats, l2_norm=True, use_mlp=True):
        """Forward pass for feature map processing."""
        out = []
        for feat_id, feat in enumerate(feats):
            B, C, Hk, Wk = feat.shape
            feat = feat.permute(0, 2, 3, 1).flatten(0, 1).flatten(0, 1)  # [B*Hk*Wk, C]
            
            if self.use_mlp and use_mlp:
                mlp = getattr(self, f'mlp_{feat_id}')
                feat = mlp(feat)
                
            if l2_norm:
                feat = self.l2norm(feat)
                
            feat = feat.view(B, Hk, Wk, feat.shape[-1])
            out.append(feat)
        return out
        
    def forward(self, feats, num_patches=-1, masks=None, pos_grids=None, 
                neg_grids=None, only_init=False, l2_norm=True, use_mlp=True):
        """Forward pass for positive/negative sampling."""
        pos_feats = []
        neg_feats = []
        
        # Initialize MLPs if needed
        if self.use_mlp and not self.mlp_init:
            self.create_mlp(feats)
            if only_init: 
                return 
        
        if masks is not None:
            masks = masks[:, 0, :, :]
            
        # Process positive samples
        if pos_grids is None:
            pos_grids = self._generate_grids_from_mask(masks, num_patches)
        else:
            num_samples = pos_grids.shape[-2]

        for feat_id, feat in enumerate(feats):
            x_sample = self._sample_features(feat, pos_grids, feat_id, l2_norm, use_mlp)
            pos_feats.append(x_sample)

        # Process negative samples (inverted mask)
        if masks is not None:
            masks = ~masks
            
        if neg_grids is None:
            neg_grids = self._generate_grids_from_mask(masks, num_patches)
        else:
            num_samples = neg_grids.shape[-2]
            
        for feat_id, feat in enumerate(feats):
            x_sample = self._sample_features(feat, neg_grids, feat_id, l2_norm, use_mlp, mode='nearest')
            neg_feats.append(x_sample)
            
        return pos_feats, neg_feats, pos_grids, neg_grids
    
    def _generate_grids_from_mask(self, masks, num_patches):
        """Generate sampling grids from mask."""
        if num_patches != -1:
            num_samples_per_mask = torch.sum(masks.flatten(1, 2), dim=1)
            num_samples = torch.min(num_samples_per_mask).item()
            num_samples = min(num_samples, num_patches)
            
        B, H, W = masks.shape
        grids = []
        
        for b in range(B):
            mask_y, mask_x = torch.where(masks[b])
            mask_y = 2 * (mask_y.float() / (H - 1)) - 1
            mask_x = 2 * (mask_x.float() / (W - 1)) - 1
            grid = torch.stack([mask_x, mask_y], dim=-1)
            
            if num_patches != -1: 
                grid = grid[torch.randint(0, grid.shape[0], (num_samples,))]
            grids.append(grid)  
            
        return torch.stack(grids, dim=0)[:, None, ...]
    
    def _sample_features(self, feat, grids, feat_id, l2_norm, use_mlp, mode='bilinear'):
        """Sample features from feature maps using grids."""
        B, C, Hk, Wk = feat.shape
        x_sample = F.grid_sample(feat, grids, align_corners=True, mode=mode)[:, :, 0, ...]
        x_sample = x_sample.permute(0, 2, 1).flatten(0, 1)  # [B*N, C]
        
        if self.use_mlp and use_mlp:
            mlp = getattr(self, f'mlp_{feat_id}')
            x_sample = mlp(x_sample)
            
        if l2_norm:
            x_sample = self.l2norm(x_sample)
            
        x_sample = x_sample.view(B, -1, x_sample.shape[-1])
        return x_sample


# ====================
# Discriminator Networks
# ====================

class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator."""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, no_antialias=True):
        super().__init__()
        
        # Determine bias usage
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        
        # First layer
        if no_antialias:
            sequence = [
                nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), 
                nn.LeakyReLU(0.2, True)
            ]
        else:
            sequence = [
                nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=1, padding=padw), 
                nn.LeakyReLU(0.2, True), 
                Downsample(ndf)
            ]
        
        # Intermediate layers
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            
            if no_antialias:
                sequence += [
                    nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, 
                             kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                    norm_layer(ndf * nf_mult),
                    nn.LeakyReLU(0.2, True)
                ]
            else:
                sequence += [
                    nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, 
                             kernel_size=kw, stride=1, padding=padw, bias=use_bias),
                    norm_layer(ndf * nf_mult),
                    nn.LeakyReLU(0.2, True),
                    Downsample(ndf * nf_mult)
                ]

        # Penultimate layer
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, 
                     kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        # Output layer
        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
        ]
        
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward pass."""
        return self.model(input)


# ====================
# Attention Modules
# ====================

class WindowAttention(nn.Module):
    """Window-based Multi-Head Self-Attention module."""
    
    def __init__(self, dim: int, window_size: int = 8, num_heads: int = 4, qkv_bias: bool = True):
        super().__init__()
        assert dim % num_heads == 0, "`dim` must be divisible by `num_heads`"
        
        self.ws = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Q K V & output projection
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        # Relative position bias table 
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))
        coords_flat = coords.flatten(1)
        
        rel_coords = (coords_flat[:, :, None] - coords_flat[:, None, :]).permute(1, 2, 0).contiguous()
        rel_coords[:, :, 0] += window_size - 1
        rel_coords[:, :, 1] += window_size - 1
        rel_coords[:, :, 0] *= 2 * window_size - 1
        rel_index = rel_coords.sum(-1)
        
        self.register_buffer("rel_index", rel_index)
        self.rel_bias_table = nn.Parameter(torch.zeros((2 * window_size - 1) ** 2, num_heads))
        nn.init.trunc_normal_(self.rel_bias_table, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        ws = self.ws
        assert H % ws == 0 and W % ws == 0, "Feature map size must be divisible by window size"

        # Partition windows 
        x = (x.view(B, C, H // ws, ws, W // ws, ws)
               .permute(0, 2, 4, 3, 5, 1)
               .contiguous()
               .view(-1, ws * ws, C))

        B_, N, _ = x.shape
        
        # Q K V computation
        qkv = (self.qkv(x)
               .reshape(B_, N, 3, self.num_heads, self.head_dim)
               .permute(2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale

        # Attention computation
        attn = q @ k.transpose(-2, -1)
        rel_bias = self.rel_bias_table[self.rel_index.view(-1)].view(N, N, -1)
        attn = attn + rel_bias.permute(2, 0, 1).unsqueeze(0)
        attn = attn.softmax(dim=-1)

        # Apply attention and project
        out = (attn @ v)
        out = (out.transpose(1, 2).reshape(B_, N, C))
        out = self.proj(out)

        # Merge windows 
        out = (out.view(B, H // ws, W // ws, ws, ws, C)
                   .permute(0, 5, 1, 3, 2, 4)
                   .contiguous()
                   .view(B, C, H, W))
        return out


class SelfAttention(nn.Module):
    """Self-attention module for diffusion models."""
    
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
        
    def forward(self, x):
        b, c, h, w = x.shape
        x_norm = self.norm(x)
        
        qkv = self.qkv(x_norm)
        q, k, v = torch.chunk(qkv, 3, dim=1)
        
        head_dim = c // self.num_heads
        scale = head_dim ** -0.5
        
        q = q.reshape(b, self.num_heads, head_dim, h*w).permute(0, 1, 3, 2)
        k = k.reshape(b, self.num_heads, head_dim, h*w).permute(0, 1, 2, 3)
        v = v.reshape(b, self.num_heads, head_dim, h*w).permute(0, 1, 3, 2)
        
        attn = torch.matmul(q, k) * scale
        attn = F.softmax(attn, dim=-1)
        
        out = torch.matmul(attn, v)
        out = out.permute(0, 1, 3, 2).reshape(b, c, h, w)
        
        return x + self.proj(out)


# ====================
# Diffusion Model Components
# ====================

class TimeEmbedding(nn.Module):
    """Time embedding for diffusion models."""
    
    def __init__(self, T, dim, dim_out):
        super().__init__()
        self.embedding = nn.Embedding(T, dim)
        self.projection = nn.Sequential(
            nn.Linear(dim, dim_out),
            nn.SiLU(),
            nn.Linear(dim_out, dim_out)
        )
        
    def forward(self, t):
        emb = self.embedding(t)
        return self.projection(emb)


class ConditionEncoder(nn.Module):
    """Condition encoder for conditional diffusion."""
    
    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels//4, kernel_size=3, stride=2, padding=1)
        self.norm1 = norm_layer(out_channels//4)
        self.conv2 = nn.Conv2d(out_channels//4, out_channels//2, kernel_size=3, stride=2, padding=1)
        self.norm2 = norm_layer(out_channels//2)
        self.conv3 = nn.Conv2d(out_channels//2, out_channels, kernel_size=3, stride=2, padding=1)
        self.norm3 = norm_layer(out_channels)
        
    def forward(self, x):
        x = F.silu(self.norm1(self.conv1(x)))
        x = F.silu(self.norm2(self.conv2(x)))
        x = F.silu(self.norm3(self.conv3(x)))
        return x


class ResBlock(nn.Module):
    """Residual block for diffusion UNet."""
    
    def __init__(self, in_channels, out_channels, time_channels, 
                 norm_layer=nn.BatchNorm2d, dropout=0.0, use_condition=True):
        super().__init__()
        self.use_condition = use_condition
        
        self.norm1 = norm_layer(in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_channels, out_channels)
        )
        
        if use_condition:
            self.cond_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_channels, out_channels)
            )
        
        self.norm2 = norm_layer(out_channels)
        self.act2 = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        self.shortcut = nn.Identity()
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
            
    def forward(self, x, t_emb, c_emb=None):
        h = self.norm1(x)
        h = self.act1(h)
        h = self.conv1(h)
        
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        
        if self.use_condition and c_emb is not None:
            h = h + self.cond_mlp(c_emb)[:, :, None, None]
            
        h = self.norm2(h)
        h = self.act2(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        return h + self.shortcut(x)


class DownSample(nn.Module):
    """Downsampling layer for diffusion UNet."""
    
    def __init__(self, channels, with_conv=True):
        super().__init__()
        self.with_conv = with_conv
        if with_conv:
            self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)
            
    def forward(self, x, t_emb=None, c_emb=None):
        if self.with_conv:
            return self.conv(x)
        else:
            return F.avg_pool2d(x, kernel_size=2, stride=2)


class UpSample(nn.Module):
    """Upsampling layer for diffusion UNet."""
    
    def __init__(self, channels, with_conv=True):
        super().__init__()
        self.with_conv = with_conv
        if with_conv:
            self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
            
    def forward(self, x, t_emb=None, c_emb=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


# ====================
# Main Diffusion UNet
# ====================

class DiffusionUNet(nn.Module):
    """
    Diffusion UNet model for denoising diffusion probabilistic models.
    
    This model implements a U-Net architecture with attention layers, time embeddings,
    and conditional inputs for generating high-quality cryo-EM images.
    """
    
    def __init__(self, input_nc: int, output_nc: int, ngf: int, T: int, 
                 beta_1: float, beta_T: float, norm_layer=nn.BatchNorm2d,
                 use_dropout: bool = False, init_type=None, init_gain=None, 
                 gpu_ids=None, opt=None, no_antialias=False, no_antialias_up=False):
        super().__init__()
        self.T = T
        dropout = 0.1 if use_dropout else 0.0

        # Register diffusion parameters
        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T))
        alphas = 1. - self.betas
        self.register_buffer('alphas_cumprod', torch.cumprod(alphas, dim=0))
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(self.alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - self.alphas_cumprod))

        # Embeddings
        self.time_embedding = TimeEmbedding(T, ngf, ngf*4)
        self.condition_encoder = ConditionEncoder(input_nc, ngf*4, norm_layer)

        # Input projection
        self.input_conv = nn.Conv2d(input_nc, ngf, 3, padding=1)

        # Architecture configuration
        ch_mult = [1, 2, 2, 4]
        in_ch = ngf
        
        # Encoder (downsampling path)
        self.down_layers = nn.ModuleList()
        for i, mult in enumerate(ch_mult):
            out_ch = ngf * mult
            self.down_layers.append(nn.ModuleList([
                ResBlock(in_ch, out_ch, ngf*4, norm_layer, dropout),
                ResBlock(out_ch, out_ch, ngf*4, norm_layer, dropout),
                WindowAttention(out_ch, 8, 4),
                DownSample(out_ch) if i < len(ch_mult)-1 else nn.Identity()
            ]))
            in_ch = out_ch
            
        # Middle (bottleneck)
        self.middle = nn.ModuleList([
            ResBlock(in_ch, in_ch, ngf*4, norm_layer, dropout),
            nn.Identity(),
            ResBlock(in_ch, in_ch, ngf*4, norm_layer, dropout)
        ])
        
        # Decoder (upsampling path)
        self.up_layers = nn.ModuleList()
        rev_mult = list(reversed(ch_mult[:-1]))
        for idx, mult in enumerate(rev_mult): 
            skip_ch = ngf * mult
            blk1 = ResBlock(in_ch + skip_ch, skip_ch, ngf*4, norm_layer, dropout)
            blk2 = ResBlock(skip_ch, skip_ch, ngf*4, norm_layer, dropout)
            attn = WindowAttention(skip_ch, 8, 4)
            up = UpSample(skip_ch) if idx < len(rev_mult) - 1 else nn.Identity()
            self.up_layers.append(nn.ModuleList([blk1, blk2, attn, up]))
            in_ch = skip_ch
            
        # Output projection
        self.output_block = nn.Sequential(
            norm_layer(in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, output_nc, 3, padding=1)
        )

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None):
        """Forward diffusion process: add noise to clean images."""
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_at = extract(self.sqrt_alphas_cumprod, t, x0.shape)
        sqrt_omt = extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)
        return sqrt_at * x0 + sqrt_omt * noise, noise

    def forward(self, x: torch.Tensor, t: torch.LongTensor, 
                condition: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass of the UNet."""
        # Embeddings
        t_emb = self.time_embedding(t)
        c_emb = self.condition_encoder(condition).mean(dim=[2, 3]) if condition is not None else None

        # Input projection
        h = self.input_conv(x)
        skips = []
        
        # Encoder
        for blk1, blk2, attn, down in self.down_layers:
            h = blk1(h, t_emb, c_emb)
            h = blk2(h, t_emb, c_emb)
            h = attn(h)
            skips.append(h)
            h = down(h)
            
        # Middle
        h = self.middle[0](h, t_emb, c_emb)
        h = self.middle[1](h)
        h = self.middle[2](h, t_emb, c_emb)

        skips.pop()  # Remove last skip connection
        
        # Decoder
        for blk1, blk2, attn, up in self.up_layers:
            skip = skips.pop()
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode='nearest')

            h = torch.cat([h, skip], dim=1)
            h = blk1(h, t_emb, c_emb)
            h = blk2(h, t_emb, c_emb)
            h = attn(h)
            h = up(h)
            
        return self.output_block(h)

    def extract_features(self, x, layers, condition=None):
        """Extract intermediate features from specified layers."""
        features = []
        t = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        t_emb = self.time_embedding(t)
        c_emb = None
        if condition is not None:
            c_emb = self.condition_encoder(condition).mean(dim=[2, 3])
        
        h = self.input_conv(x)
        layer_idx = 1
        
        for block1, block2, attn, downsample in self.down_layers:
            h = block1(h, t_emb, c_emb)
            if layer_idx in layers:
                features.append(h)
            layer_idx += 1
            
            h = block2(h, t_emb, c_emb)
            if layer_idx in layers:
                features.append(h)
            layer_idx += 1
            
            h = attn(h)
            h = downsample(h)
        
        return features

    def pre_process(self, x, *, apply_ctf=True, apply_gaussian_noise=True, snr=0.1, apix=1.0):
        """Apply physical effects and normalization to input images."""
        x, noise = apply_physics(
            x,
            apply_ctf=apply_ctf,
            apply_gaussian_noise=apply_gaussian_noise,
            snr=snr,
            apix=apix,
        )
        x = instance_normalize(x, autocontrast=False)
        return x, noise

    # ====================
    # Sampling Methods
    # ====================

    @torch.no_grad()
    def ddpm_sample(self, x_T: torch.Tensor, condition: Optional[torch.Tensor] = None, 
                    steps: int = 100) -> torch.Tensor:
        """DDPM sampling (standard denoising diffusion)."""
        device, B = x_T.device, x_T.size(0)
        t_seq = torch.linspace(self.T-1, 0, steps, dtype=torch.long, device=device)
        x_t = x_T

        for i, t in enumerate(t_seq):
            t_batch = t.expand(B)
            eps = self.forward(x_t, t_batch, condition)

            beta_t = self.betas[t]
            alpha_t = self.alphas_cumprod[t]
            alpha_prev = self.alphas_cumprod[t-1] if t > 0 else torch.tensor(1.0, device=device)

            x0_pred = (x_t - torch.sqrt(1 - alpha_t) * eps) / torch.sqrt(alpha_t)
            mean = torch.sqrt(alpha_prev) * x0_pred + torch.sqrt(1 - alpha_prev) * eps

            var = beta_t * (1 - alpha_prev) / (1 - alpha_t)
            sigma = torch.sqrt(var)

            if i < steps - 1:
                noise = torch.randn_like(x_t)
                x_t = mean + sigma * noise
            else:
                x_t = mean

        return torch.clamp(x_t, -1, 1)

    @torch.no_grad()
    def ddim_sample(self, x_T, condition=None, steps=100, eta=0.0):
        """DDIM sampling (deterministic variant)."""
        device = x_T.device
        batch_size = x_T.shape[0]
        x_t = x_T
        
        # Create time schedule
        time_steps = torch.linspace(self.T-1, 0, steps+1).to(device).int()
        
        for i in range(steps):
            t = time_steps[i].repeat(batch_size)
            next_t = time_steps[i+1].repeat(batch_size)
            
            # Get alpha values
            at = extract(self.alphas_cumprod, t, x_t.shape)
            next_at = extract(self.alphas_cumprod, next_t, x_t.shape)
            
            # Predict noise
            et = self.forward(x_t, t, condition)
            
            # Predict x0
            x0_pred = (x_t - torch.sqrt(1 - at) * et) / torch.sqrt(at)
            x0_pred = torch.clamp(x0_pred, -1, 1)
            
            # Direction pointing to x_t
            dir_xt = torch.sqrt(1 - next_at) * et
            
            # Add stochasticity (eta controls determinism)
            if eta > 0 and i < steps - 1:
                sigma = eta * torch.sqrt((1 - next_at) / (1 - at)) * torch.sqrt(1 - at / next_at)
                noise = torch.randn_like(x_t)
                x_t = torch.sqrt(next_at) * x0_pred + dir_xt + sigma * noise
            else:
                x_t = torch.sqrt(next_at) * x0_pred + dir_xt
                
        return torch.clamp(x_t, -1, 1)

    @torch.no_grad()
    def dpmsolver_sample(self, x_T, condition=None, steps=20, order=2):
        """DPM-Solver sampling (corrected implementation)."""
        device, B = x_T.device, x_T.size(0)
        x = x_T
        
        # Generate time schedule
        t_seq = torch.linspace(self.T-1, 0, steps+1, device=device, dtype=torch.long)
        
        model_outputs = []
        
        for i in range(steps):
            t_now = t_seq[i].repeat(B)
            t_next = t_seq[i+1].repeat(B)
            
            # Current alpha values
            alpha_now = extract(self.alphas_cumprod, t_now, x.shape)
            alpha_next = extract(self.alphas_cumprod, t_next, x.shape)
            
            # Predict noise
            et = self.forward(x, t_now, condition)
            model_outputs.append(et)
            
            # Predict x0
            x0_pred = (x - torch.sqrt(1 - alpha_now) * et) / torch.sqrt(alpha_now)
            x0_pred = torch.clamp(x0_pred, -1, 1)
            
            if order == 1 or len(model_outputs) < 2:
                # First-order (equivalent to DDIM)
                x = torch.sqrt(alpha_next) * x0_pred + torch.sqrt(1 - alpha_next) * et
            elif order == 2 and len(model_outputs) >= 2:
                # Second-order linear combination
                et_prev = model_outputs[-2]
                t_prev = t_seq[i-1].repeat(B)
                
                # Coefficients for linear combination
                h = self._compute_lambda(t_next) - self._compute_lambda(t_now)
                h_prev = self._compute_lambda(t_now) - self._compute_lambda(t_prev)
                r = h / h_prev
                
                # Linear combination
                et_corrected = (1 + 1/(2*r)) * et - 1/(2*r) * et_prev
                x = torch.sqrt(alpha_next) * x0_pred + torch.sqrt(1 - alpha_next) * et_corrected
            elif order == 3 and len(model_outputs) >= 3:
                # Third-order linear combination
                et_prev1 = model_outputs[-2]
                et_prev2 = model_outputs[-3]
                t_prev1 = t_seq[i-1].repeat(B)
                t_prev2 = t_seq[i-2].repeat(B)
                
                # Compute coefficients
                h = self._compute_lambda(t_next) - self._compute_lambda(t_now)
                h_prev1 = self._compute_lambda(t_now) - self._compute_lambda(t_prev1)
                h_prev2 = self._compute_lambda(t_prev1) - self._compute_lambda(t_prev2)
                
                r1 = h / h_prev1
                r2 = h / (h_prev1 + h_prev2)
                
                # Linear combination
                et_corrected = ((1 + 1/r1 + 1/(r1*r2)) * et 
                               - (1/r1 + 1/(r1*r2)) * et_prev1 
                               + 1/(r1*r2) * et_prev2)
                x = torch.sqrt(alpha_next) * x0_pred + torch.sqrt(1 - alpha_next) * et_corrected
        
        return torch.clamp(x, -1, 1)
    
    def _compute_lambda(self, t):
        """Compute lambda_t = log(alpha_t) - log(sigma_t)."""
        alpha_t = extract(self.alphas_cumprod, t, t.shape + (1, 1, 1))
        return torch.log(alpha_t / (1 - alpha_t))

    @torch.no_grad()
    def dpmsolverpp_sample(self, x_T, condition=None, steps=20, order=2, use_corrector=False):
        """DPM-Solver++ sampling (corrected implementation)."""
        device, B = x_T.device, x_T.size(0)
        x = x_T
        
        # Generate timesteps
        timesteps = torch.linspace(self.T-1, 0, steps+1, device=device, dtype=torch.long)
        
        model_outputs = []
        
        for i in range(steps):
            t_now = timesteps[i].repeat(B)
            t_next = timesteps[i+1].repeat(B)
            
            alpha_now = extract(self.alphas_cumprod, t_now, x.shape)
            alpha_next = extract(self.alphas_cumprod, t_next, x.shape)
            
            # Predict noise
            et = self.forward(x, t_now, condition)
            model_outputs.append(et)
            
            # Predict x0 (data prediction model)
            x0_pred = (x - torch.sqrt(1 - alpha_now) * et) / torch.sqrt(alpha_now)
            x0_pred = torch.clamp(x0_pred, -1.0, 1.0)
            
            if order == 1 or len(model_outputs) < 2:
                # First-order
                x_next = torch.sqrt(alpha_next) * x0_pred + torch.sqrt(1 - alpha_next) * et
            elif order == 2 and len(model_outputs) >= 2:
                # Second-order Adams-Bashforth
                et_prev = model_outputs[-2]
                x0_prev_pred = (x_prev - torch.sqrt(1 - alpha_prev) * et_prev) / torch.sqrt(alpha_prev)
                
                # Interpolation coefficient
                coeff = 0.5
                x0_interpolated = (1 + coeff) * x0_pred - coeff * x0_prev_pred
                x0_interpolated = torch.clamp(x0_interpolated, -1.0, 1.0)
                
                x_next = torch.sqrt(alpha_next) * x0_interpolated + torch.sqrt(1 - alpha_next) * et
            elif order == 3 and len(model_outputs) >= 3:
                # Third-order Adams-Bashforth
                et_prev1 = model_outputs[-2]
                et_prev2 = model_outputs[-3]
                
                # More sophisticated interpolation
                x0_interpolated = (23/12) * x0_pred - (16/12) * x0_prev1_pred + (5/12) * x0_prev2_pred
                x0_interpolated = torch.clamp(x0_interpolated, -1.0, 1.0)
                
                x_next = torch.sqrt(alpha_next) * x0_interpolated + torch.sqrt(1 - alpha_next) * et
            
            # Optional corrector step
            if use_corrector and i < steps - 1:
                et_corrected = self.forward(x_next, t_next, condition)
                x0_corrected = (x_next - torch.sqrt(1 - alpha_next) * et_corrected) / torch.sqrt(alpha_next)
                x0_corrected = torch.clamp(x0_corrected, -1.0, 1.0)
                x_next = torch.sqrt(alpha_next) * x0_corrected + torch.sqrt(1 - alpha_next) * et_corrected
            
            # Store previous values for next iteration
            x_prev, alpha_prev = x, alpha_now
            if len(model_outputs) >= 2:
                x0_prev1_pred = x0_pred
            if len(model_outputs) >= 3:
                x0_prev2_pred = x0_prev1_pred
                
            x = x_next
                
        return torch.clamp(x, -1, 1)

    @torch.no_grad()
    def unipc_sample(self, x_T, condition=None, steps=20, order=2, use_corrector=True):
        """
        UniPC sampling - Unified Predictor-Corrector Framework.
        
        This is a state-of-the-art sampler that can achieve high quality in 5-10 steps.
        Based on: https://arxiv.org/abs/2302.04867
        """
        device, B = x_T.device, x_T.size(0)
        x = x_T
        
        # Generate time schedule
        timesteps = torch.linspace(self.T-1, 0, steps+1, device=device, dtype=torch.long)
        
        # Store model outputs for higher-order methods
        model_outputs = []
        x0_predictions = []
        
        for i in range(steps):
            t_now = timesteps[i].repeat(B)
            t_next = timesteps[i+1].repeat(B)
            
            alpha_now = extract(self.alphas_cumprod, t_now, x.shape)
            alpha_next = extract(self.alphas_cumprod, t_next, x.shape)
            
            lambda_now = self._compute_lambda(t_now)
            lambda_next = self._compute_lambda(t_next)
            h = lambda_next - lambda_now
            
            # Predict noise (Unified Predictor)
            et = self.forward(x, t_now, condition)
            model_outputs.append(et)
            
            # Predict x0
            x0_pred = (x - torch.sqrt(1 - alpha_now) * et) / torch.sqrt(alpha_now)
            x0_pred = torch.clamp(x0_pred, -1, 1)
            x0_predictions.append(x0_pred)
            
            # UniPC Predictor step
            if order == 1 or len(model_outputs) < order:
                # Use lower order when we don't have enough history
                effective_order = min(order, len(model_outputs))
                if effective_order == 1:
                    # First-order (DDIM-like)
                    x_pred = torch.sqrt(alpha_next) * x0_pred + torch.sqrt(1 - alpha_next) * et
                else:
                    # Multi-step predictor
                    x_pred = self._unipc_multistep_predictor(
                        x, model_outputs, x0_predictions, lambda_now, lambda_next, 
                        alpha_next, effective_order
                    )
            else:
                # Full order predictor
                x_pred = self._unipc_multistep_predictor(
                    x, model_outputs, x0_predictions, lambda_now, lambda_next, 
                    alpha_next, order
                )
            
            # UniPC Corrector step (optional)
            if use_corrector and i < steps - 1:
                x_pred = self._unipc_corrector(
                    x_pred, x, t_next, t_now, condition, alpha_now, alpha_next
                )
            
            x = x_pred
            
        return torch.clamp(x, -1, 1)
    
    def _unipc_multistep_predictor(self, x, model_outputs, x0_predictions, 
                                   lambda_now, lambda_next, alpha_next, order):
        """Multi-step predictor for UniPC."""
        if order == 1:
            return torch.sqrt(alpha_next) * x0_predictions[-1] + torch.sqrt(1 - alpha_next) * model_outputs[-1]
        elif order == 2:
            # Linear extrapolation
            coeff = 0.5
            x0_extrap = (1 + coeff) * x0_predictions[-1] - coeff * x0_predictions[-2]
            x0_extrap = torch.clamp(x0_extrap, -1, 1)
            return torch.sqrt(alpha_next) * x0_extrap + torch.sqrt(1 - alpha_next) * model_outputs[-1]
        elif order == 3:
            # Quadratic extrapolation
            x0_extrap = (23/12) * x0_predictions[-1] - (16/12) * x0_predictions[-2] + (5/12) * x0_predictions[-3]
            x0_extrap = torch.clamp(x0_extrap, -1, 1)
            return torch.sqrt(alpha_next) * x0_extrap + torch.sqrt(1 - alpha_next) * model_outputs[-1]
        else:
            # Fallback to first-order
            return torch.sqrt(alpha_next) * x0_predictions[-1] + torch.sqrt(1 - alpha_next) * model_outputs[-1]
    
    def _unipc_corrector(self, x_pred, x_now, t_next, t_now, condition, alpha_now, alpha_next):
        """UniPC corrector step."""
        # Evaluate model at predicted point
        et_pred = self.forward(x_pred, t_next, condition)
        x0_pred_corrected = (x_pred - torch.sqrt(1 - alpha_next) * et_pred) / torch.sqrt(alpha_next)
        x0_pred_corrected = torch.clamp(x0_pred_corrected, -1, 1)
        
        # Corrector formula (simplified)
        correction_weight = 0.5
        x_corrected = x_pred + correction_weight * (
            torch.sqrt(alpha_next) * x0_pred_corrected + torch.sqrt(1 - alpha_next) * et_pred - x_pred
        )
        
        return x_corrected

    @torch.no_grad()
    def lms_sample(self, x_T, condition=None, steps=50, order=4):
        """
        Linear Multi-Step (LMS) sampling.
        
        A numerical ODE solver that uses values from previous timesteps.
        Good balance between speed and quality.
        """
        device, B = x_T.device, x_T.size(0)
        x = x_T
        
        # LMS coefficients for different orders
        lms_coeffs = {
            1: [1],
            2: [3/2, -1/2],
            3: [23/12, -16/12, 5/12],
            4: [55/24, -59/24, 37/24, -9/24]
        }
        
        timesteps = torch.linspace(self.T-1, 0, steps+1, device=device, dtype=torch.long)
        
        # Store derivatives for multi-step
        derivatives = []
        
        for i in range(steps):
            t_now = timesteps[i].repeat(B)
            t_next = timesteps[i+1].repeat(B)
            
            alpha_now = extract(self.alphas_cumprod, t_now, x.shape)
            alpha_next = extract(self.alphas_cumprod, t_next, x.shape)
            
            # Compute derivative (noise prediction)
            et = self.forward(x, t_now, condition)
            x0_pred = (x - torch.sqrt(1 - alpha_now) * et) / torch.sqrt(alpha_now)
            x0_pred = torch.clamp(x0_pred, -1, 1)
            
            # Store derivative
            derivative = et
            derivatives.append(derivative)
            
            # Use appropriate order based on available history
            current_order = min(order, len(derivatives))
            coeffs = lms_coeffs[current_order]
            
            # Compute step
            h = (alpha_next - alpha_now)
            
            if current_order == 1:
                # Euler step
                x_next = torch.sqrt(alpha_next) * x0_pred + torch.sqrt(1 - alpha_next) * et
            else:
                # Multi-step
                weighted_derivative = torch.zeros_like(et)
                for j in range(current_order):
                    weighted_derivative += coeffs[j] * derivatives[-(j+1)]
                
                x_next = torch.sqrt(alpha_next) * x0_pred + torch.sqrt(1 - alpha_next) * weighted_derivative
            
            # Keep only the last `order` derivatives
            if len(derivatives) > order:
                derivatives.pop(0)
            
            x = x_next
            
        return torch.clamp(x, -1, 1)

    @torch.no_grad()
    def heun_sample(self, x_T, condition=None, steps=50):
        """
        Heun's method sampling - a second-order Runge-Kutta method.
        
        More accurate than Euler but requires two function evaluations per step.
        """
        device, B = x_T.device, x_T.size(0)
        x = x_T
        
        timesteps = torch.linspace(self.T-1, 0, steps+1, device=device, dtype=torch.long)
        
        for i in range(steps):
            t_now = timesteps[i].repeat(B)
            t_next = timesteps[i+1].repeat(B)
            
            alpha_now = extract(self.alphas_cumprod, t_now, x.shape)
            alpha_next = extract(self.alphas_cumprod, t_next, x.shape)
            
            # First prediction (Euler step)
            et1 = self.forward(x, t_now, condition)
            x0_pred1 = (x - torch.sqrt(1 - alpha_now) * et1) / torch.sqrt(alpha_now)
            x0_pred1 = torch.clamp(x0_pred1, -1, 1)
            x_euler = torch.sqrt(alpha_next) * x0_pred1 + torch.sqrt(1 - alpha_next) * et1
            
            # Second prediction (at Euler point)
            et2 = self.forward(x_euler, t_next, condition)
            
            # Average the two predictions (Heun's method)
            et_avg = 0.5 * (et1 + et2)
            x0_pred_avg = (x - torch.sqrt(1 - alpha_now) * et_avg) / torch.sqrt(alpha_now)
            x0_pred_avg = torch.clamp(x0_pred_avg, -1, 1)
            
            x = torch.sqrt(alpha_next) * x0_pred_avg + torch.sqrt(1 - alpha_next) * et_avg
            
        return torch.clamp(x, -1, 1)


    @torch.no_grad()
    def sample(self, x_T, condition=None, steps=20, sampler_type='dpmsolver++', 
               solver_order=2, use_corrector=False, eta=0.0):
        """
        Unified sampling interface supporting multiple high-performance samplers.
        
        Args:
            x_T: Initial noise tensor
            condition: Conditional input (optional)
            steps: Number of sampling steps
            sampler_type: Type of sampler to use
            solver_order: Order for solvers that support it (1-3)
            use_corrector: Whether to use corrector steps
            eta: Stochasticity parameter for DDIM (0=deterministic, 1=stochastic)
            
        Available samplers:
            - 'ddpm': Standard DDPM sampling
            - 'ddim': DDIM sampling (deterministic/stochastic)
            - 'dpmsolver': DPM-Solver (fast, high-order)
            - 'dpmsolver++': DPM-Solver++ (improved version)
            - 'unipc': UniPC (state-of-the-art, 5-10 steps)
            - 'lms': Linear Multi-Step (balanced speed/quality)
            - 'heun': Heun's method (second-order, accurate)
        """
        sampler_map = {
            'ddpm': lambda: self.ddpm_sample(x_T, condition, steps),
            'ddim': lambda: self.ddim_sample(x_T, condition, steps, eta),
            'dpmsolver': lambda: self.dpmsolver_sample(x_T, condition, steps, solver_order),
            'dpmsolver++': lambda: self.dpmsolverpp_sample(x_T, condition, steps, solver_order, use_corrector),
            'unipc': lambda: self.unipc_sample(x_T, condition, steps, solver_order, use_corrector),
            'lms': lambda: self.lms_sample(x_T, condition, steps, solver_order),
            'heun': lambda: self.heun_sample(x_T, condition, steps)
        }
        
        if sampler_type not in sampler_map:
            available_samplers = list(sampler_map.keys())
            raise ValueError(f"Unknown sampler type: {sampler_type}. Available: {available_samplers}")
        
        return sampler_map[sampler_type]()