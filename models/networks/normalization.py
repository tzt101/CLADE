"""
Copyright (C) University of Science and Technology of China.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.networks.sync_batchnorm import SynchronizedBatchNorm2d
import torch.nn.utils.spectral_norm as spectral_norm


# Returns a function that creates a normalization function
# that does not condition on semantic map
def get_nonspade_norm_layer(opt, norm_type='instance'):
    # helper function to get # output channels of the previous layer
    def get_out_channel(layer):
        if hasattr(layer, 'out_channels'):
            return getattr(layer, 'out_channels')
        return layer.weight.size(0)

    # this function will be returned
    def add_norm_layer(layer):
        nonlocal norm_type
        if norm_type.startswith('spectral'):
            layer = spectral_norm(layer)
            subnorm_type = norm_type[len('spectral'):]

        if subnorm_type == 'none' or len(subnorm_type) == 0:
            return layer

        # remove bias in the previous layer, which is meaningless
        # since it has no effect after normalization
        if getattr(layer, 'bias', None) is not None:
            delattr(layer, 'bias')
            layer.register_parameter('bias', None)

        if subnorm_type == 'batch':
            norm_layer = nn.BatchNorm2d(get_out_channel(layer), affine=True)
        elif subnorm_type == 'sync_batch':
            norm_layer = SynchronizedBatchNorm2d(get_out_channel(layer), affine=True)
        elif subnorm_type == 'instance':
            norm_layer = nn.InstanceNorm2d(get_out_channel(layer), affine=False)
        else:
            raise ValueError('normalization layer %s is not recognized' % subnorm_type)

        return nn.Sequential(layer, norm_layer)

    return add_norm_layer


# Creates SPADE normalization layer based on the given configuration
# SPADE consists of two steps. First, it normalizes the activations using
# your favorite normalization method, such as Batch Norm or Instance Norm.
# Second, it applies scale and bias to the normalized output, conditioned on
# the segmentation map.
# The format of |config_text| is spade(norm)(ks), where
# (norm) specifies the type of parameter-free normalization.
#       (e.g. syncbatch, batch, instance)
# (ks) specifies the size of kernel in the SPADE module (e.g. 3x3)
# Example |config_text| will be spadesyncbatch3x3, or spadeinstance5x5.
# Also, the other arguments are
# |norm_nc|: the #channels of the normalized activations, hence the output dim of SPADE
# |label_nc|: the #channels of the input semantic map, hence the input dim of SPADE
class SPADE(nn.Module):
    def __init__(self, config_text, norm_nc, label_nc):
        super().__init__()

        assert config_text.startswith('spade')
        parsed = re.search('spade(\D+)(\d)x\d', config_text)
        param_free_norm_type = str(parsed.group(1))
        ks = int(parsed.group(2))

        if param_free_norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'syncbatch':
            self.param_free_norm = SynchronizedBatchNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        else:
            raise ValueError('%s is not a recognized param-free norm type in SPADE'
                             % param_free_norm_type)

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 128

        pw = ks // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)

    def forward(self, x, segmap, input_dist=None):

        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out

class SPADELight(nn.Module):
    def __init__(self, config_text, norm_nc, label_nc, no_instance=True, add_dist=False):
        super().__init__()
        self.no_instance = no_instance
        self.add_dist = add_dist
        assert config_text.startswith('spade')
        parsed = re.search('spade(\D+)(\d)x\d', config_text)
        param_free_norm_type = str(parsed.group(1))
        ks = int(parsed.group(2))

        if param_free_norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'syncbatch':
            self.param_free_norm = SynchronizedBatchNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        else:
            raise ValueError('%s is not a recognized param-free norm type in SPADE'
                             % param_free_norm_type)
        self.class_specified_affine = ClassAffine(label_nc, norm_nc, add_dist)

        if not no_instance:
            self.inst_conv = nn.Conv2d(1, 1, kernel_size=1, padding=0)

    def forward(self, x, segmap, input_dist=None):

        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. scale the segmentation mask
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')

        if not self.no_instance:
            inst_map = torch.unsqueeze(segmap[:,-1,:,:],1)
            segmap = segmap[:,:-1,:,:]

        # Part 3. class affine with noise
        out = self.class_specified_affine(normalized, segmap, input_dist)

        if not self.no_instance:
            inst_feat = self.inst_conv(inst_map)
            out = torch.cat((out, inst_feat), dim=1)

        return out

class ClassAffine(nn.Module):
    def __init__(self, label_nc, affine_nc, add_dist=False):
        super(ClassAffine, self).__init__()
        self.add_dist = add_dist
        self.affine_nc = affine_nc
        self.label_nc = label_nc
        self.weight = nn.Parameter(torch.Tensor(self.label_nc, self.affine_nc))
        self.bias = nn.Parameter(torch.Tensor(self.label_nc, self.affine_nc))
        nn.init.uniform_(self.weight)
        nn.init.zeros_(self.bias)
        if add_dist:
            self.dist_conv_w = nn.Conv2d(2, 1, kernel_size=1, padding=0)
            nn.init.zeros_(self.dist_conv_w.weight)
            nn.init.zeros_(self.dist_conv_w.bias)
            self.dist_conv_b = nn.Conv2d(2, 1, kernel_size=1, padding=0)
            nn.init.zeros_(self.dist_conv_b.weight)
            nn.init.zeros_(self.dist_conv_b.bias)

    def affine_gather(self, input, mask):
        n, c, h, w = input.shape
        # process mask
        mask2 = torch.argmax(mask, 1) # [n, h, w]
        mask2 = mask2.view(n, h*w).long() # [n, hw]
        mask2 = mask2.unsqueeze(1).expand(n, self.affine_nc, h*w) # [n, nc, hw]
        # process weights
        weight2 = torch.unsqueeze(self.weight, 2).expand(self.label_nc, self.affine_nc, h*w) # [cls, nc, hw]
        bias2 = torch.unsqueeze(self.bias, 2).expand(self.label_nc, self.affine_nc, h*w) # [cls, nc, hw]
        # torch gather function
        class_weight = torch.gather(weight2, 0, mask2).view(n, self.affine_nc, h, w)
        class_bias = torch.gather(bias2, 0, mask2).view(n, self.affine_nc, h, w)
        return class_weight, class_bias

    def affine_einsum(self, mask):
        class_weight = torch.einsum('ic,nihw->nchw', self.weight, mask)
        class_bias = torch.einsum('ic,nihw->nchw', self.bias, mask)
        return class_weight, class_bias

    def affine_embed(self, mask):
        arg_mask = torch.argmax(mask, 1).long() # [n, h, w]
        class_weight = F.embedding(arg_mask, self.weight).permute(0, 3, 1, 2) # [n, c, h, w]
        class_bias = F.embedding(arg_mask, self.bias).permute(0, 3, 1, 2) # [n, c, h, w]
        return class_weight, class_bias

    def forward(self, input, mask, input_dist=None):
        # class_weight, class_bias = self.affine_gather(input, mask)
        # class_weight, class_bias = self.affine_einsum(mask) 
        class_weight, class_bias = self.affine_embed(mask)
        if self.add_dist:
            input_dist = F.interpolate(input_dist, size=input.size()[2:], mode='nearest')
            class_weight = class_weight * (1 + self.dist_conv_w(input_dist))
            class_bias = class_bias * (1 + self.dist_conv_b(input_dist))
        x = input * class_weight + class_bias
        return x
