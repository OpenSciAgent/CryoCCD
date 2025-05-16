import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch import Tensor
import functools
from torch.optim import lr_scheduler
import numpy as np
from cryoccd import ctf as ctf_func
from cryoccd.transform import instance_normalize
import math
import torch.nn.utils.spectral_norm as spectral_norm
from typing import List, Optional, Tuple

import logging

logger = logging.getLogger(__name__)

def define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal',
             init_gain=0.02, gpu_ids=[], opt=None, no_antialias=True, no_antialias_up=True):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netG.startswith('unet_'):
        size = int(netG[5:])
        assert size > 0 and (size & (size - 1)) == 0, 'size must be power of 2'
        depth = int(np.log2(size))
        net = UnetGenerator(input_nc, output_nc, depth, ngf, norm_layer=norm_layer, use_dropout=use_dropout)

    elif netG.startswith('resnet_'):
        n_blocks = int(netG[7])
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=n_blocks, opt=opt, no_antialias=no_antialias, no_antialias_up=no_antialias_up)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    
    logger.info(net)
    
    return init_net(net, init_type, init_gain, gpu_ids)

def define_F(input_nc, netF, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[], opt=None):
    if netF == 'global_pool':
        net = PoolingF()
    elif netF == 'reshape':
        net = ReshapeF()
    elif netF == 'sample':
        net = PatchSampleF(use_mlp=False, init_type=init_type, init_gain=init_gain, gpu_ids=gpu_ids, nc=opt.netF_nc)
    elif netF == 'mlp_sample':
        net = PatchSampleF(use_mlp=True, init_type=init_type, init_gain=init_gain, gpu_ids=gpu_ids, nc=opt.netF_nc)
    elif netF == 'strided_conv':
        net = StridedConvF(init_type=init_type, init_gain=init_gain, gpu_ids=gpu_ids)
    elif netF == 'mask_sample':
        net = MaskInformedSampleF(use_mlp=True, init_type=init_type, init_gain=init_gain, gpu_ids=gpu_ids, nc=opt.netF_nc)
    else:
        raise NotImplementedError('projection model name [%s] is not recognized' % netF)
    return init_net(net, init_type, init_gain, gpu_ids)

def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[], opt=None, no_antialias=True):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)
    if netD == 'basic':  # default PatchGAN classifier
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, no_antialias=no_antialias,)
    elif netD == 'n_layers':  # more options
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, no_antialias=no_antialias,)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)
    return init_net(net, init_type, init_gain, gpu_ids,
                    initialize_weights=('stylegan2' not in netD))

class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect', opt=None, no_antialias=True, no_antialias_up=True):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.opt = opt
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            
            if (no_antialias):
                model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                          norm_layer(ngf * mult * 2),
                          nn.ReLU(True)]
                
            else:
                model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=1, padding=1, bias=use_bias),
                    norm_layer(ngf * mult * 2),
                    nn.ReLU(True),
                    Downsample(ngf * mult * 2)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            
            if no_antialias_up:
                model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                             kernel_size=3, stride=2,
                                             padding=1, output_padding=1,
                                             bias=use_bias),
                          norm_layer(int(ngf * mult / 2)),
                          nn.ReLU(True)]
            else:
                model += [Upsample(ngf * mult),
                          nn.Conv2d(ngf * mult, int(ngf * mult / 2),
                                    kernel_size=3, stride=1,
                                    padding=1,  # output_padding=1,
                                    bias=use_bias),
                          norm_layer(int(ngf * mult / 2)),
                          nn.ReLU(True)]
                
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self,                 
                input, 
                nce_layers=[],
                encode_only=False, 
                apply_ctf=True,
                apply_gaussian_noise=True,
                snr=0.1,
                apix=1.0,
        ):
        """Standard forward"""
        batch = input.shape[0]
        sidelen = input.shape[-1]
        
        # apply ctf
        if apply_ctf:
            ctf_params = ctf_func.generate_random_ctf_params(batch)
            freqs_mag, angles_rad = ctf_func.compute_safe_freqs(sidelen, apix)
            ctf = ctf_func.compute_ctf(freqs_mag, angles_rad, *ctf_params).reshape(batch, 1, sidelen, sidelen)
            
            ctf = torch.from_numpy(ctf).to(input.device).float()
            ctf_corrupted_fourier_images = ctf * ctf_func.torch_fft2_center(input)
            input  = ctf_func.torch_ifft2_center(ctf_corrupted_fourier_images).real
            
        # apply gaussian noise
        if apply_gaussian_noise:
            noise_std = torch.sqrt(torch.var(input , axis=(-2, -1), keepdims=True) / snr)
            expand_noise_std = torch.tile(noise_std, (1, sidelen, sidelen))
            input = input + torch.randn_like(input) * expand_noise_std
                
        input = instance_normalize(input, autocontrast=False)
                
        if encode_only:
            feat = input
            feats = []
            for layer_id, layer in enumerate(self.model):
                # logger.info(layer_id, layer)
                feat = layer(feat)
                if layer_id in nce_layers:
                    # logger.info("%d: adding the output of %s %d" % (layer_id, layer.__class__.__name__, feat.size(1)))
                    feats.append(feat)
                else:
                    # logger.info("%d: skipping %s %d" % (layer_id, layer.__class__.__name__, feat.size(1)))
                    pass
                if layer_id == nce_layers[-1] and encode_only:
                    # logger.info('encoder only return features')
                    return feats, None  # return intermediate features alone; stop in the last layers

            return feats, input 
        else:
            """Standard forward"""
            fake = self.model(input)
            return fake, input

class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out
    
class MaskInformedSampleF(nn.Module):
    def __init__(self, use_mlp=False, init_type='normal', init_gain=0.02, nc=256, gpu_ids=[]):
        # potential issues: currently, we use the same patch_ids for multiple images in the batch
        super(MaskInformedSampleF, self).__init__()
        self.l2norm = Normalize(2)
        self.use_mlp = use_mlp
        self.nc = nc  # hard-coded
        self.mlp_init = False
        self.init_type = init_type
        self.init_gain = init_gain
        self.gpu_ids = gpu_ids   

    def create_mlp(self, feats):
        for mlp_id, feat in enumerate(feats):
            input_nc = feat.shape[1]
            mlp = nn.Sequential(*[nn.Linear(input_nc, self.nc), nn.ReLU(), nn.Linear(self.nc, self.nc)])
            if len(self.gpu_ids) > 0:
                mlp.cuda()
            setattr(self, 'mlp_%d' % mlp_id, mlp)
        init_net(self, self.init_type, self.init_gain, self.gpu_ids)
        self.mlp_init = True
        
    def custom_forward(self, feats, l2_norm=True, use_mlp=True):
        out = []
        for feat_id, feat in enumerate(feats):
            B, C, Hk, Wk = feat.shape
            feat = feat.permute(0, 2, 3, 1).flatten(0, 1).flatten(0, 1) # [B*Hk*Wk, C]
            if self.use_mlp and use_mlp:
                mlp = getattr(self, 'mlp_%d' % feat_id)
                feat = mlp(feat)
            if l2_norm:
                feat = self.l2norm(feat)
            feat = feat.view(B, Hk, Wk, feat.shape[-1])
            out.append(feat)
        return out
        
    def forward(self, 
            feats, 
            num_patches=-1, 
            masks=None, 
            pos_grids=None, 
            neg_grids=None,
            only_init=False,
            l2_norm=True,
            use_mlp=True,
    ):
        pos_feats = []
        neg_feats = []
        if self.use_mlp and not self.mlp_init:
            self.create_mlp(feats)
            if only_init: return 
        if masks is not None:
            masks = masks[:, 0, :, :]
        if pos_grids is None:
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
                if num_patches != -1: grid = grid[torch.randint(0, grid.shape[0], (num_samples,))]
                grids.append(grid)  
            pos_grids = torch.stack(grids, dim=0)[:, None, ...]
        else:
            num_samples = pos_grids.shape[-2]
        
        for feat_id, feat in enumerate(feats):
            B, C, Hk, Wk = feat.shape
            x_sample = F.grid_sample(feat, pos_grids, align_corners=True, mode='bilinear')[:, :, 0, ...] # [B, C, N]
            # x_sample = F.grid_sample(masks[None,...].float(), pos_grids, align_corners=True, mode='bilinear')[:, :, 0, ...] # [B, C, N]
            x_sample = x_sample.permute(0, 2, 1).flatten(0, 1) # [B*N, C]
            if self.use_mlp and use_mlp:
                mlp = getattr(self, 'mlp_%d' % feat_id)
                x_sample = mlp(x_sample)
            if l2_norm:
                x_sample = self.l2norm(x_sample)
            x_sample = x_sample.view(B, -1, x_sample.shape[-1])
            pos_feats.append(x_sample)

        # bool inverse
        if masks is not None:
            masks = ~masks
        if neg_grids is None:
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
                if num_patches != -1: grid = grid[torch.randint(0, grid.shape[0], (num_samples,))]
                grids.append(grid)  
            neg_grids = torch.stack(grids, dim=0)[:, None, ...]
        else:
            num_samples = neg_grids.shape[-2]
        for feat_id, feat in enumerate(feats):
            B, C, Hk, Wk = feat.shape
            x_sample = F.grid_sample(feat, neg_grids, align_corners=True, mode='nearest')[:, :, 0, ...] # [B, C, N]
            x_sample = x_sample.permute(0, 2, 1).flatten(0, 1) # [B*N, C]

            if self.use_mlp and use_mlp:
                mlp = getattr(self, 'mlp_%d' % feat_id)
                x_sample = mlp(x_sample)
            if l2_norm:
                x_sample = self.l2norm(x_sample)
            x_sample = x_sample.view(B, num_samples, -1).view(B, -1, x_sample.shape[-1])
            neg_feats.append(x_sample)
            
        return pos_feats, neg_feats, pos_grids, neg_grids

class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetGenerator, self).__init__()
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, 
                input, 
                nce_layers=[],
                encode_only=False, 
                apply_ctf=True,
                apply_gaussian_noise=True,
                snr=0.1,
                apix=1.0,
                return_features=False,
        ):
        """Standard forward"""
        batch, _, sidelen, _ = input.shape
        # apply ctf
        if apply_ctf:
            ctf_params = ctf_func.generate_random_ctf_params(batch)
            freqs_mag, angles_rad = ctf_func.compute_safe_freqs(sidelen, apix)
            ctf = ctf_func.compute_ctf(freqs_mag, angles_rad, *ctf_params).reshape(batch, 1, sidelen, sidelen)
            ctf = torch.from_numpy(ctf).to(input.device).float()
            ctf_corrupted_fourier_images = ctf * ctf_func.torch_fft2_center(input)
            input  = ctf_func.torch_ifft2_center(ctf_corrupted_fourier_images).real
            
        # apply gaussian noise
        if apply_gaussian_noise:
            noise_std = torch.sqrt(torch.var(input , axis=(-2, -1), keepdims=True) / snr)
            expand_noise_std = torch.tile(noise_std, (1, 1, sidelen, sidelen))
            input = input + torch.randn_like(input) * expand_noise_std
                
        noisy_input = input
        
        input = instance_normalize(
            input,
            autocontrast=False
        )
            
        if encode_only:
            output, features = self.model(input)
            features = features[::-1]
            feats = []
            if nce_layers:
                for layer in nce_layers:
                    feats.append(features[layer])
            else:
                feats = features
            return feats, noisy_input
            
        elif return_features:
            # return both input features and output features
            output, features = self.model(input)
            features = features[::-1]
            feats = []
            if nce_layers:
                for layer in nce_layers:
                    feats.append(features[layer])
            input_features = feats
            _, features = self.model(output)
            features = features[::-1]
            feats = []
            if nce_layers:
                for layer in nce_layers:
                    feats.append(features[layer])
            output_features = feats
            return input_features, output_features, output, noisy_input
        
        else:
            output, _ = self.model(input)
                
            return output, noisy_input
        
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        self.innermost = innermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
                
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                up = up + [nn.Dropout(0.5)]

        self.down = nn.Sequential(*down)
        self.submodule = submodule
        self.up = nn.Sequential(*up)

    def forward(self, x):
        
        downsample_output = self.down(x)
        if self.submodule is not None:
            output, features = self.submodule(downsample_output)
        else: # inner most
            features = []
            output = downsample_output
            
        output = self.up(output)
            
        features.append(downsample_output)
                
        if self.outermost:
            features.append(x)
            return output, features
        else:
            return torch.cat([x, output], 1), features
        
class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, no_antialias=True):
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        if(no_antialias):
            sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        else:
            sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=1, padding=padw), nn.LeakyReLU(0.2, True), Downsample(ndf)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            if(no_antialias):
                sequence += [
                    nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                    norm_layer(ndf * nf_mult),
                    nn.LeakyReLU(0.2, True)
                ]
            else:
                sequence += [
                    nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
                    norm_layer(ndf * nf_mult),
                    nn.LeakyReLU(0.2, True),
                    Downsample(ndf * nf_mult)]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)
    
def get_filter(filt_size=3):
    if(filt_size == 1):
        a = np.array([1., ])
    elif(filt_size == 2):
        a = np.array([1., 1.])
    elif(filt_size == 3):
        a = np.array([1., 2., 1.])
    elif(filt_size == 4):
        a = np.array([1., 3., 3., 1.])
    elif(filt_size == 5):
        a = np.array([1., 4., 6., 4., 1.])
    elif(filt_size == 6):
        a = np.array([1., 5., 10., 10., 5., 1.])
    elif(filt_size == 7):
        a = np.array([1., 6., 15., 20., 15., 6., 1.])

    filt = torch.Tensor(a[:, None] * a[None, :])
    filt = filt / torch.sum(filt)

    return filt

def get_pad_layer(pad_type):
    if(pad_type in ['refl', 'reflect']):
        PadLayer = nn.ReflectionPad2d
    elif(pad_type in ['repl', 'replicate']):
        PadLayer = nn.ReplicationPad2d
    elif(pad_type == 'zero'):
        PadLayer = nn.ZeroPad2d
    else:
        logger.info('Pad type [%s] not recognized' % pad_type)
    return PadLayer

def apply_physics(img, *, apply_ctf=True, apply_gaussian_noise=True,
                  snr=0.1, apix=1.0):
    B, _, H, _ = img.shape
    noise = None

    # -------- ① CTF --------
    if apply_ctf:
        ctf_params = ctf_func.generate_random_ctf_params(B)
        freqs_mag, ang_rad = ctf_func.compute_safe_freqs(H, apix)
        ctf = ctf_func.compute_ctf(freqs_mag, ang_rad, *ctf_params)\
                      .reshape(B, 1, H, H)
        ctf = torch.as_tensor(ctf, dtype=img.dtype, device=img.device)

        fourier = ctf_func.torch_fft2_center(img)
        img     = ctf_func.torch_ifft2_center(ctf * fourier).real

    # -------- ② Gaussian noise --------
    if apply_gaussian_noise:
        # σ = sqrt(var(signal) / SNR)
        noise_std = torch.sqrt(torch.var(img, dim=(-2, -1), keepdims=True) / snr)
        noise     = torch.randn_like(img) * noise_std.repeat(1, 1, H, H)
        img = img + noise

    return img, noise

class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x):
            return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions.
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
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
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler

class WindowAttention(nn.Module):
    def __init__(self, dim: int, window_size: int = 8, num_heads: int = 4,
                 qkv_bias: bool = True):
        super().__init__()
        assert dim % num_heads == 0, "`dim` must be divisible by `num_heads`"
        self.ws        = window_size
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.scale     = self.head_dim ** -0.5          # 1 / √d

        # Q K V & output projection
        self.qkv  = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        # ---------------- relative-position bias table ----------------
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords   = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))   # (2, ws, ws)
        coords_flat = coords.flatten(1)                                             # (2, ws²)
        # (2, ws², ws²) → (ws², ws², 2)
        rel_coords = (coords_flat[:, :, None] - coords_flat[:, None, :]).permute(1, 2, 0).contiguous()
        # shift to ≥0
        rel_coords[:, :, 0] += window_size - 1
        rel_coords[:, :, 1] += window_size - 1
        rel_coords[:, :, 0] *= 2 * window_size - 1
        rel_index = rel_coords.sum(-1)                                              # (ws², ws²)
        self.register_buffer("rel_index", rel_index)
        self.rel_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) ** 2, num_heads)
        )
        nn.init.trunc_normal_(self.rel_bias_table, std=0.02)

    # -----------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:            # (B, C, H, W)
        B, C, H, W = x.shape
        ws = self.ws
        assert H % ws == 0 and W % ws == 0, "Feature map size must be divisible by window size"

        # -------- 1) partition windows --------
        x = (x.view(B, C, H // ws, ws, W // ws, ws)            # (B,C, H/ws,ws, W/ws,ws)
               .permute(0, 2, 4, 3, 5, 1)                      # (B, H/ws, W/ws, ws,ws,C)
               .contiguous()
               .view(-1, ws * ws, C))                          # (B*nWin, N, C)

        B_, N, _ = x.shape                                     # nWin*B, N=ws²
        # -------- 2) Q K V --------
        qkv = (self.qkv(x)
               .reshape(B_, N, 3, self.num_heads, self.head_dim)
               .permute(2, 0, 3, 1, 4))                        # (3, B_, heads, N, d)
        q, k, v = qkv[0], qkv[1], qkv[2]                       # each: (B_, heads, N, d)
        q = q * self.scale

        # -------- 3) attention --------
        attn = q @ k.transpose(-2, -1)                         # (B_, heads, N, N)
        rel_bias = self.rel_bias_table[self.rel_index.view(-1)].view(N, N, -1)  # (N,N,heads)
        attn = attn + rel_bias.permute(2, 0, 1).unsqueeze(0)   # broadcast to (1,heads,N,N)
        attn = attn.softmax(dim=-1)

        # -------- 4) projection --------
        out = (attn @ v)                                       # (B_, heads, N, d)
        out = (out.transpose(1, 2)                             # (B_, N, heads, d)
                    .reshape(B_, N, C))                        # (B_, N, C)
        out = self.proj(out)                                   # (B_, N, C)

        # -------- 5) merge windows --------
        out = (out.view(B, H // ws, W // ws, ws, ws, C)
                   .permute(0, 5, 1, 3, 2, 4)                  # (B,C,H/ws,ws,W/ws,ws)
                   .contiguous()
                   .view(B, C, H, W))
        return out

def init_weights(net, init_type='normal', init_gain=0.02, debug=False):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
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
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    net.apply(init_func)  # apply the initialization function <init_func>
    
def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[], debug=False, initialize_weights=True):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        # net.to(f"cuda:{gpu_ids[0]}")
        net.to(gpu_ids[0])
        # net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs for non-AMP training
        # if not amp:
    if initialize_weights:
        init_weights(net, init_type, init_gain=init_gain, debug=debug)
    return net

class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm + 1e-7)
        return out

class PoolingF(nn.Module):
    def __init__(self):
        super(PoolingF, self).__init__()
        model = [nn.AdaptiveMaxPool2d(1)]
        self.model = nn.Sequential(*model)
        self.l2norm = Normalize(2)

    def forward(self, x):
        return self.l2norm(self.model(x))


class ReshapeF(nn.Module):
    def __init__(self):
        super(ReshapeF, self).__init__()
        model = [nn.AdaptiveAvgPool2d(4)]
        self.model = nn.Sequential(*model)
        self.l2norm = Normalize(2)

    def forward(self, x):
        x = self.model(x)
        x_reshape = x.permute(0, 2, 3, 1).flatten(0, 2)
        return self.l2norm(x_reshape)
    
class Downsample(nn.Module):
    def __init__(self, channels, pad_type='reflect', filt_size=3, stride=2, pad_off=0):
        super(Downsample, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int(1. * (filt_size - 1) / 2), int(np.ceil(1. * (filt_size - 1) / 2)), int(1. * (filt_size - 1) / 2), int(np.ceil(1. * (filt_size - 1) / 2))]
        self.pad_sizes = [pad_size + pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride - 1) / 2.)
        self.channels = channels

        filt = get_filter(filt_size=self.filt_size)
        self.register_buffer('filt', filt[None, None, :, :].repeat((self.channels, 1, 1, 1)))

        self.pad = get_pad_layer(pad_type)(self.pad_sizes)

    def forward(self, inp):
        if(self.filt_size == 1):
            if(self.pad_off == 0):
                return inp[:, :, ::self.stride, ::self.stride]
            else:
                return self.pad(inp)[:, :, ::self.stride, ::self.stride]
        else:
            return F.conv2d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])
        
class Upsample(nn.Module):
    def __init__(self, channels, pad_type='repl', filt_size=4, stride=2):
        super(Upsample, self).__init__()
        self.filt_size = filt_size
        self.filt_odd = np.mod(filt_size, 2) == 1
        self.pad_size = int((filt_size - 1) / 2)
        self.stride = stride
        self.off = int((self.stride - 1) / 2.)
        self.channels = channels

        filt = get_filter(filt_size=self.filt_size) * (stride**2)
        self.register_buffer('filt', filt[None, None, :, :].repeat((self.channels, 1, 1, 1)))

        self.pad = get_pad_layer(pad_type)([1, 1, 1, 1])
        
    def forward(self, inp):
        ret_val = F.conv_transpose2d(self.pad(inp), self.filt, stride=self.stride, padding=1 + self.pad_size, groups=inp.shape[1])[:, :, 1:, 1:]
        if(self.filt_odd):
            return ret_val
        else:
            return ret_val[:, :, :-1, :-1]
        
class StridedConvF(nn.Module):
    def __init__(self, init_type='normal', init_gain=0.02, gpu_ids=[]):
        super().__init__()
        # self.conv1 = nn.Conv2d(256, 128, 3, stride=2)
        # self.conv2 = nn.Conv2d(128, 64, 3, stride=1)
        self.l2_norm = Normalize(2)
        self.mlps = {}
        self.moving_averages = {}
        self.init_type = init_type
        self.init_gain = init_gain
        self.gpu_ids = gpu_ids

    def create_mlp(self, x):
        C, H = x.shape[1], x.shape[2]
        n_down = int(np.rint(np.log2(H / 32)))
        mlp = []
        for i in range(n_down):
            mlp.append(nn.Conv2d(C, max(C // 2, 64), 3, stride=2))
            mlp.append(nn.ReLU())
            C = max(C // 2, 64)
        mlp.append(nn.Conv2d(C, 64, 3))
        mlp = nn.Sequential(*mlp)
        init_net(mlp, self.init_type, self.init_gain, self.gpu_ids)
        return mlp

    def update_moving_average(self, key, x):
        if key not in self.moving_averages:
            self.moving_averages[key] = x.detach()

        self.moving_averages[key] = self.moving_averages[key] * 0.999 + x.detach() * 0.001

    def forward(self, x, use_instance_norm=False):
        C, H = x.shape[1], x.shape[2]
        key = '%d_%d' % (C, H)
        if key not in self.mlps:
            self.mlps[key] = self.create_mlp(x)
            self.add_module("child_%s" % key, self.mlps[key])
        mlp = self.mlps[key]
        x = mlp(x)
        self.update_moving_average(key, x)
        x = x - self.moving_averages[key]
        if use_instance_norm:
            x = F.instance_norm(x)
        return self.l2_norm(x)


class PatchSampleF(nn.Module):
    def __init__(self, use_mlp=False, init_type='normal', init_gain=0.02, nc=256, gpu_ids=[]):
        # potential issues: currently, we use the same patch_ids for multiple images in the batch
        super(PatchSampleF, self).__init__()
        self.l2norm = Normalize(2)
        self.use_mlp = use_mlp
        self.nc = nc  # hard-coded
        self.mlp_init = False
        self.init_type = init_type
        self.init_gain = init_gain
        self.gpu_ids = gpu_ids

    def create_mlp(self, feats):
        for mlp_id, feat in enumerate(feats):
            input_nc = feat.shape[1]
            mlp = nn.Sequential(*[nn.Linear(input_nc, self.nc), nn.ReLU(), nn.Linear(self.nc, self.nc)])
            if len(self.gpu_ids) > 0:
                mlp.cuda()
            setattr(self, 'mlp_%d' % mlp_id, mlp)
        init_net(self, self.init_type, self.init_gain, self.gpu_ids)
        self.mlp_init = True

    def forward(self, feats, num_patches=64, patch_ids=None):
        return_ids = []
        return_feats = []
        if self.use_mlp and not self.mlp_init:
            self.create_mlp(feats)
        for feat_id, feat in enumerate(feats):
            B, H, W = feat.shape[0], feat.shape[2], feat.shape[3]
            feat_reshape = feat.permute(0, 2, 3, 1).flatten(1, 2)
            if num_patches > 0:
                if patch_ids is not None:
                    patch_id = patch_ids[feat_id]
                else:
                    # torch.randperm produces cudaErrorIllegalAddress for newer versions of PyTorch. https://github.com/taesungp/contrastive-unpaired-translation/issues/83
                    #patch_id = torch.randperm(feat_reshape.shape[1], device=feats[0].device)
                    patch_id = np.random.permutation(feat_reshape.shape[1])
                    patch_id = patch_id[:int(min(num_patches, patch_id.shape[0]))]  # .to(patch_ids.device)
                patch_id = torch.tensor(patch_id, dtype=torch.long, device=feat.device)
                x_sample = feat_reshape[:, patch_id, :].flatten(0, 1)  # reshape(-1, x.shape[1])
            else:
                x_sample = feat_reshape
                patch_id = []
            if self.use_mlp:
                mlp = getattr(self, 'mlp_%d' % feat_id)
                x_sample = mlp(x_sample)
            return_ids.append(patch_id)
            x_sample = self.l2norm(x_sample)

            if num_patches == 0:
                x_sample = x_sample.permute(0, 2, 1).reshape([B, x_sample.shape[-1], H, W])
            
            x_sample = x_sample.view(B, -1, x_sample.shape[-1])
            return_feats.append(x_sample)
        return return_feats, return_ids


def define_diffusion_unet(input_nc, output_nc, ngf, T, beta_1, beta_T, norm='instance',
                          use_dropout=False, init_type='normal', init_gain=0.02,
                          gpu_ids=[], opt=None, no_antialias=False, no_antialias_up=False):
    norm_layer = get_norm_layer(norm_type=norm)
    net = DiffusionUNet(input_nc, output_nc, ngf, T, beta_1, beta_T, norm_layer, 
                       use_dropout, no_antialias, no_antialias_up)
    return init_net(net, init_type, init_gain, gpu_ids, opt)


class TimeEmbedding(nn.Module):
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


class SelfAttention(nn.Module):
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
        
        q = q.reshape(b, self.num_heads, head_dim, h*w).permute(0, 1, 3, 2)  # (b, heads, h*w, head_dim)
        k = k.reshape(b, self.num_heads, head_dim, h*w).permute(0, 1, 2, 3)  # (b, heads, head_dim, h*w)
        v = v.reshape(b, self.num_heads, head_dim, h*w).permute(0, 1, 3, 2)  # (b, heads, h*w, head_dim)
        
        attn = torch.matmul(q, k) * scale  # (b, heads, h*w, h*w)
        attn = F.softmax(attn, dim=-1)
        
        out = torch.matmul(attn, v)  # (b, heads, h*w, head_dim)
        out = out.permute(0, 1, 3, 2).reshape(b, c, h, w)  # (b, c, h, w)
        
        return x + self.proj(out)


class DownSample(nn.Module):
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


class DiffusionUNet(nn.Module):
    def __init__(
        self,
        input_nc:       int,
        output_nc:      int,
        ngf:            int,
        T:              int,
        beta_1:         float,
        beta_T:         float,
        norm_layer=     nn.BatchNorm2d,
        use_dropout:    bool = False,
        init_type=None, init_gain=None, gpu_ids=None, opt=None,
        no_antialias=False, no_antialias_up=False
    ):
        super().__init__()
        self.T = T
        dropout = 0.1 if use_dropout else 0.0

        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T))
        alphas = 1. - self.betas
        self.register_buffer('alphas_cumprod', torch.cumprod(alphas, dim=0))
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(self.alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - self.alphas_cumprod))

        self.time_embedding = TimeEmbedding(T, ngf, ngf*4)
        self.condition_encoder = ConditionEncoder(input_nc, ngf*4, norm_layer)

        self.input_conv = nn.Conv2d(input_nc, ngf, 3, padding=1)

        # ch_mult = [1, 2, 2, 4]
        ch_mult = [1, 2, 2, 4]
        in_ch = ngf
        
        self.down_layers = nn.ModuleList()
        for i, mult in enumerate(ch_mult):
            out_ch = ngf * mult
            self.down_layers.append(nn.ModuleList([
                ResBlock(in_ch,  out_ch, ngf*4, norm_layer, dropout),
                ResBlock(out_ch, out_ch, ngf*4, norm_layer, dropout),
                WindowAttention(out_ch, 8, 4),
                DownSample(out_ch) if i < len(ch_mult)-1 else nn.Identity()
            ]))
            in_ch = out_ch
            
        self.middle = nn.ModuleList([
            ResBlock(in_ch, in_ch, ngf*4, norm_layer, dropout),
            nn.Identity(),
            ResBlock(in_ch, in_ch, ngf*4, norm_layer, dropout)
        ])
        
        self.up_layers = nn.ModuleList()
        rev_mult = list(reversed(ch_mult[:-1]))
        for idx, mult in enumerate(rev_mult): 
            skip_ch = ngf * mult
            blk1 = ResBlock(in_ch + skip_ch, skip_ch, ngf*4, norm_layer, dropout)
            blk2 = ResBlock(skip_ch,          skip_ch, ngf*4, norm_layer, dropout)
            attn = WindowAttention(skip_ch, 8, 4)
            up   = UpSample(skip_ch) if idx < len(rev_mult) - 1 else nn.Identity()
            self.up_layers.append(nn.ModuleList([blk1, blk2, attn, up]))
            in_ch = skip_ch
            
        self.output_block = nn.Sequential(
            norm_layer(in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, output_nc, 3, padding=1)
        )

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor,
                 noise: Optional[torch.Tensor] = None):
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_at  = extract(self.sqrt_alphas_cumprod, t, x0.shape)
        sqrt_omt = extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)
        return sqrt_at * x0 + sqrt_omt * noise, noise

    def forward(self, x: torch.Tensor, t: torch.LongTensor, condition: Optional[torch.Tensor]=None) -> torch.Tensor:
        t_emb = self.time_embedding(t)
        c_emb = self.condition_encoder(condition).mean(dim=[2,3]) if condition is not None else None

        h = self.input_conv(x)
        skips = []
        # down
        for blk1, blk2, attn, down in self.down_layers:
            h = blk1(h, t_emb, c_emb)
            h = blk2(h, t_emb, c_emb)
            h = attn(h)
            skips.append(h)
            h = down(h)
        # middle
        h = self.middle[0](h, t_emb, c_emb)
        h = self.middle[1](h)
        h = self.middle[2](h, t_emb, c_emb)

        skips.pop()  
        # up
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

    def _encode(self, x: torch.Tensor, t_emb: torch.Tensor,
                c_emb: torch.Tensor | None, nce_layers: list[int]):
        feats, layer_id = [], 0
        h = self.input_conv(x)

        for b1, b2, attn, down in self.down_layers:
            h = b1(h, t_emb, c_emb)
            if layer_id in nce_layers: feats.append(h); layer_id += 1
            h = b2(h, t_emb, c_emb)
            if layer_id in nce_layers: feats.append(h); layer_id += 1
            h = attn(h)
            feats.append(h) if layer_id in nce_layers else None
            h = down(h)

        return feats, h

        
    def pre_process(self, x, *, apply_ctf=True, apply_gaussian_noise=True,
                    snr=0.1, apix=1.0):
        x, noise = apply_physics(
            x,
            apply_ctf=apply_ctf,
            apply_gaussian_noise=apply_gaussian_noise,
            snr=snr,
            apix=apix,
        )
        x = instance_normalize(x, autocontrast=False)

        return x, noise
    
    def extract_features(self, x, layers, condition=None):
        features = []
        t = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        t_emb = self.time_embedding(t)
        c_emb = None
        if condition is not None:
            c_emb = self.condition_encoder(condition).mean(dim=[2,3])
        
        h = self.input_conv(x)
        hs = [h]
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
            hs.append(h)
            h = downsample(h)
        
        return features
    
    def ddim_sample(self, x_T, condition=None, steps=100):
        """DDIM"""
        device = x_T.device
        batch_size = x_T.shape[0]
        x_t = x_T
        
        time_steps = torch.linspace(self.T-1, 0, steps+1).to(device)
        time_steps = time_steps.int()
        
        for i in range(steps):
            t = time_steps[i].repeat(batch_size)
            next_t = time_steps[i+1].repeat(batch_size)
            at = extract(self.alphas_cumprod, t, x_t.shape)
            next_at = extract(self.alphas_cumprod, next_t, x_t.shape)
            
            et = self.forward(x_t, t, condition)
            
            x0_t = (x_t - et * torch.sqrt(1 - at)) / torch.sqrt(at)
            
            x0_t = torch.clamp(x0_t, -1, 1)
            
            c1 = torch.sqrt(1 - next_at)
            c2 = torch.sqrt(next_at)
            xt_next = c2 * x0_t + c1 * et
            
            if i < steps - 1:
                noise = torch.randn_like(x_t)
                sigma = 0.0
                xt_next = xt_next + sigma * noise
                
            x_t = xt_next
            
        return torch.clamp(x_t, -1, 1)


    @torch.no_grad()
    def ddpm_sample(self, x_T: torch.Tensor, condition: Optional[torch.Tensor] = None, steps: int = 100) -> torch.Tensor:
        device, B = x_T.device, x_T.size(0)
        t_seq = torch.linspace(self.T-1, 0, steps, dtype=torch.long, device=device)
        x_t = x_T

        for i, t in enumerate(t_seq):
            t_batch = t.expand(B)  # (B,)
            # 1) ε_θ(x_t, t)
            eps = self.forward(x_t, t_batch, condition)  

            # 2) β_t, ᾱ_t, ᾱ_{t-1}
            beta_t     = self.betas[t]
            alpha_t    = self.alphas_cumprod[t]
            alpha_prev = self.alphas_cumprod[t-1] if t > 0 else torch.tensor(1.0, device=device)

            # 3) x0_pred = (x_t - sqrt(1-α_t)*eps) / sqrt(α_t)
            x0_pred = (x_t - torch.sqrt(1 - alpha_t) * eps) / torch.sqrt(alpha_t)

            # 4) mean = sqrt(alpha_prev) * x0_pred + sqrt(1-alpha_prev) * eps
            mean = torch.sqrt(alpha_prev) * x0_pred + torch.sqrt(1 - alpha_prev) * eps

            # 5) σ_t^2 = β_t * (1-α_{t-1})/(1-α_t)
            var = beta_t * (1 - alpha_prev) / (1 - alpha_t)
            sigma = torch.sqrt(var)

            if i < steps - 1:
                noise = torch.randn_like(x_t)
                x_t = mean + sigma * noise
            else:
                x_t = mean

        return torch.clamp(x_t, -1, 1) 
        
    @torch.no_grad()
    def dpmsolver_sample(self, x_T, condition=None, steps=20, order=2):
        device, B = x_T.device, x_T.size(0)
        
        t_T = self.T - 1
        t_0 = 0

        def compute_alpha_products(t):
            alpha_cumprod_t = extract(self.alphas_cumprod, t, x_T.shape)
            return alpha_cumprod_t
        
        def compute_lambda(t):
            alpha_cumprod = compute_alpha_products(t)
            log_alpha_t = torch.log(alpha_cumprod)
            log_1_min_alpha_t = torch.log(1 - alpha_cumprod)
            return log_alpha_t - log_1_min_alpha_t
        
        timesteps = torch.linspace(1, 0, steps + 1, device=device)[:-1]
        timesteps = torch.flip(timesteps, [0])  # 从T到0
        
        x = x_T
        
        model_prev = None
        model_prev_prev = None
        lambda_prev = None
        lambda_prev_prev = None
        
        for i, t_scale in enumerate(timesteps):
            t_cur = torch.full((B,), t_scale * t_T, device=device, dtype=torch.long)
            t_next = torch.full((B,), timesteps[min(i + 1, len(timesteps) - 1)] * t_T, 
                               device=device, dtype=torch.long)
            lambda_t = compute_lambda(t_cur)
            lambda_next = compute_lambda(t_next)
            h = lambda_next - lambda_t
            et = self.forward(x, t_cur, condition)
            if order == 1 or model_prev is None:
                x0_t = (x - torch.sqrt(1 - compute_alpha_products(t_cur)) * et) / torch.sqrt(compute_alpha_products(t_cur))
                x0_t = torch.clamp(x0_t, -1, 1)
                
                alpha_next = compute_alpha_products(t_next)
                x = torch.sqrt(alpha_next) * x0_t + torch.sqrt(1 - alpha_next) * et
                
            elif order == 2 and model_prev is not None:
                r = h / (lambda_t - lambda_prev)
                D1 = (x - model_prev) / (lambda_t - lambda_prev)
                x = x + h * ((1 + 0.5 * r) * D1)
                
            elif order == 3 and model_prev is not None and model_prev_prev is not None:
                r1 = h / (lambda_t - lambda_prev)
                r2 = h / (lambda_t - lambda_prev_prev)
                
                D1 = (x - model_prev) / (lambda_t - lambda_prev)
                D2 = (model_prev - model_prev_prev) / (lambda_prev - lambda_prev_prev)
                x = x + h * (D1 + 0.5 * r1 * (D1 - D2))
            
            model_prev_prev = model_prev
            model_prev = x.clone()
            lambda_prev_prev = lambda_prev
            lambda_prev = lambda_t.clone()
        
        return torch.clamp(x, -1, 1)        
        

    @torch.no_grad()
    def dpmsolverpp_sample(self, x_T, condition=None, steps=20, order=2, use_corrector=False):
        device, B = x_T.device, x_T.size(0)
        
        x = x_T
        model_outputs = []
        time_steps = []
        t_T = self.T - 1
        t_0 = 0
        lambda_T = torch.log(self.betas[t_T] / (1 - self.alphas_cumprod[t_T]))
        lambda_0 = torch.log(self.betas[t_0] / (1 - self.alphas_cumprod[t_0]))
        timesteps = torch.linspace(lambda_T.item(), lambda_0.item(), steps+1, device=device)
        timesteps = torch.flip(torch.softmax(timesteps, dim=0), dims=[0]) * steps
        for i in range(steps+1):
            t_idx = min(t_T, max(t_0, int(timesteps[i].item())))
            time_steps.append(t_idx)
        history = {}
        for i in range(steps):
            t_now = torch.tensor([time_steps[i]], device=device).repeat(B)
            t_next = torch.tensor([time_steps[i+1]], device=device).repeat(B)
            alpha_now = extract(self.alphas_cumprod, t_now, x.shape)
            alpha_next = extract(self.alphas_cumprod, t_next, x.shape)
            lambda_now = torch.log(alpha_now / (1 - alpha_now))
            lambda_next = torch.log(alpha_next / (1 - alpha_next))
            h = lambda_next - lambda_now
            et = self.forward(x, t_now, condition)
            model_outputs.append(et)
            x_0_pred = (x - torch.sqrt(1 - alpha_now) * et) / torch.sqrt(alpha_now)
            x_0_pred = torch.clamp(x_0_pred, -1.0, 1.0)
            if order == 1 or len(model_outputs) < 2:
                x_next = torch.sqrt(alpha_next) * x_0_pred + torch.sqrt(1 - alpha_next) * et
            elif order == 2 and len(model_outputs) >= 2:
                et_prev = model_outputs[-2]
                t_prev = torch.tensor([time_steps[i-1]], device=device).repeat(B)
                alpha_prev = extract(self.alphas_cumprod, t_prev, x.shape)
                lambda_prev = torch.log(alpha_prev / (1 - alpha_prev))
                
                h_last = lambda_now - lambda_prev
                r = h / h_last
                et_interp = (1 + 1/r) * et - (1/r) * et_prev
                x_next = torch.sqrt(alpha_next) * x_0_pred + torch.sqrt(1 - alpha_next) * et_interp
            elif order == 3 and len(model_outputs) >= 3:
                et_prev1 = model_outputs[-2]
                et_prev2 = model_outputs[-3]
                
                t_prev1 = torch.tensor([time_steps[i-1]], device=device).repeat(B)
                t_prev2 = torch.tensor([time_steps[i-2]], device=device).repeat(B)
                
                alpha_prev1 = extract(self.alphas_cumprod, t_prev1, x.shape)
                alpha_prev2 = extract(self.alphas_cumprod, t_prev2, x.shape)
                
                lambda_prev1 = torch.log(alpha_prev1 / (1 - alpha_prev1))
                lambda_prev2 = torch.log(alpha_prev2 / (1 - alpha_prev2))
                
                h_last1 = lambda_now - lambda_prev1
                h_last2 = lambda_prev1 - lambda_prev2
                r1 = h / h_last1
                r2 = h / (h_last1 + h_last2)
                
                et_interp = ((1 + 1/r1 + 1/r2) * et - 
                             (1/r1 + 1/r2**2) * et_prev1 + 
                             (1/r2) * et_prev2)
                x_next = torch.sqrt(alpha_next) * x_0_pred + torch.sqrt(1 - alpha_next) * et_interp
            
            if use_corrector and i < steps - 1:
                et_next = self.forward(x_next, t_next, condition)
                x_0_next_pred = (x_next - torch.sqrt(1 - alpha_next) * et_next) / torch.sqrt(alpha_next)
                x_0_next_pred = torch.clamp(x_0_next_pred, -1.0, 1.0)
                
                x_next = torch.sqrt(alpha_next) * x_0_next_pred + torch.sqrt(1 - alpha_next) * et_next
            
            x = x_next
                
        return torch.clamp(x, -1, 1)
    
    @torch.no_grad()
    def sample(self, x_T, condition=None, steps=20, sampler_type='dpmsolver++', 
               solver_order=2, use_corrector=False):
        if sampler_type == 'ddpm':
            return self.ddpm_sample(x_T, condition, steps)
        elif sampler_type == 'ddim':
            return self.ddim_sample(x_T, condition, steps)
        elif sampler_type == 'dpmsolver':
            return self.dpmsolver_sample(x_T, condition, steps, order=solver_order)
        elif sampler_type == 'dpmsolver++':
            return self.dpmsolverpp_sample(x_T, condition, steps, 
                                           order=solver_order, 
                                           use_corrector=use_corrector)
        else:
            raise ValueError(f"{sampler_type}")


def extract(a: torch.Tensor, t: torch.LongTensor, x_shape):
    t = t.to(a.device).long()
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.view(batch_size, *([1] * (len(x_shape)-1)))