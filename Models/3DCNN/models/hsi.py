# -*- coding: utf-8 -*-
"""
Created on Fri May 20 11:46:01 2022

@author: Blues
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch 
import torch.nn as nn
import numpy as np
from itertools import repeat
import collections.abc
import torch.nn.functional as F
from torchsummary import summary
import math
import warnings

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple

def _no_grad_trunc_normal_(tensor, mean=0, std=1, a=-2., b=2.):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        tensor.uniform_(2 * l - 1, 2 * u - 1)

        tensor.erfinv_()

        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        tensor.clamp_(min=a, max=b)
        return tensor

def torch_SAD(cen_idx, nei_idx):
    cen_idx = torch.sigmoid(cen_idx)
    nei_idx = torch.sigmoid(nei_idx)
    cos_alpha = cen_idx @ nei_idx / (torch.sqrt(torch.sum(torch.pow(cen_idx, 2))) * torch.sqrt(torch.sum(torch.pow(nei_idx, 2))))
    print(cos_alpha)
    return torch.acos(cos_alpha)

class hybrid_position_embedding(nn.Module):
    def __init__(self, keep_prob=0.9, block_size=3, beta=0.9):
        super(hybrid_position_embedding, self).__init__()
        self.patch_size = 1
        self.beta = beta
        self.keep_prob = keep_prob
        self.block_size = block_size
        self.gelu = nn.GELU()
        self.softmax = nn.Softmax(dim=1)

    def normalize(self, input):
        min_c, max_c = input.min(1, keepdim=True)[0], input.max(1, keepdim=True)[0]
        input_norm = (input - min_c) / (max_c - min_c + 1e-8)
        return input_norm

    def cal_SAD(self, cen_idx, nei_idx):
        cos_alpha = cen_idx @ nei_idx / (np.sqrt(np.sum(np.power(cen_idx, 2))) * np.sqrt(np.sum(np.power(nei_idx, 2))))
        eps = 1e-6
        if 1.0 < cos_alpha < 1.0 + eps:
            cos_alpha = 1.0
        elif -1.0 - eps < cos_alpha < -1.0:
            cos_alpha = -1.0
        return np.arccos(cos_alpha)

    def cal_SID(self, y_true, y_pred):
        y_true = y_true + 1e-7
        y_pred = y_pred + 1e-7
        p_n = y_true / np.sum(y_true, axis=-1, keepdims=True)
        q_n = y_pred / np.sum(y_pred, axis=-1, keepdims=True)
        return np.sum(p_n * np.log(p_n / q_n)) + np.sum(q_n * np.log(q_n / p_n))

    def forward(self, input):
        input_norm = self.normalize(input).detach().cpu().numpy()
        cen_idx = int((input_norm.shape[2] - 1) / 2)
        hybrid_matrix = np.zeros([input_norm.shape[0], input_norm.shape[2]])
        for i in range(input_norm.shape[0]):
            for j in range(input_norm.shape[2]):
                hybrid_matrix[i, j] = 1 - (self.cal_SID(input_norm[i, :, cen_idx], input_norm[i, :, j]) * np.tan(self.cal_SAD(input_norm[i, :, cen_idx], input_norm[i, :, j])))
        hybrid_matrix = torch.from_numpy(hybrid_matrix).long().to(input.device, dtype=input.dtype)
        hybrid_matrix = hybrid_matrix.view(input.size()[0], 1, input.size()[2])
        Hmap = F.conv1d(hybrid_matrix, torch.ones((1, 1, self.patch_size)).to(device=input.device, dtype=input.dtype), padding=self.patch_size // 2, groups=1)
        Out = self.softmax(Hmap)
        return Out

def measure_pos_embd(inputs):
    print(inputs)
    B, C, P = inputs.size()
    cen_idx = int((P - 1) / 2)
    spc_matrix = torch.zeros([B, P])
    for i in range(B):
        for j in range(P):
            spc_matrix[i, j] = torch_SAD(inputs[i, :, cen_idx], inputs[i, :, j])
    print(spc_matrix)
    device = torch.device('cuda:0')
    spc_matrix = spc_matrix.to(device)
    return spc_matrix

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

class Linear_projection(nn.Module):
    def __init__(self, input_size=(99, 144), patch_size=3, in_chans=200, embed_dim=28):
        super(Linear_projection, self).__init__()
        self.input_size = input_size
        self.conv = nn.Conv3d(in_channels=1, out_channels=int((in_chans - 7) / 2 + 1), kernel_size=(patch_size, patch_size, patch_size), stride=(1, 1, 1), padding=(0, 1, 1))
        self.hybrid = hybrid_position_embedding()
        self.proj = nn.Conv3d(in_channels=1, out_channels=embed_dim, kernel_size=(7, 1, 1), stride=(7, 1, 1))

    def forward(self, x):
        B, C, D, H, W = x.size()

        # Ensure input size matches expected input size for convolution
        assert H == self.input_size[0] and W == self.input_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.input_size[0]}*{self.input_size[1]})."

        # Perform convolution
        x = self.conv(x)
        
        # Compute output dimensions after convolution
        Cout = x.size(1)
        
        # Reshape tensor for further processing
        x = x.view(B, Cout, D, H, W)
        
        # Apply hybrid position embedding
        hybrid_embd = self.hybrid(x)
        
        # Concatenate tensors
        x = torch.cat([x, hybrid_embd], dim=1)
        
        # Perform projection
        x = self.proj(x)
        
        return x




def Spectral_Mixer_Layer(dim, depth, kernel_size=(7, 1, 1), patch_size=7):
    return nn.Sequential(
        *[nn.Sequential(
            Residual(nn.Sequential(
                nn.Conv3d(dim, dim, kernel_size, groups=dim, padding=(int(kernel_size[0] / 2), 0, 0)),
                nn.GELU(),
                nn.BatchNorm3d(dim)
            )),
            nn.Conv3d(dim, dim, kernel_size=1, padding=(0, 0, 0)),
            nn.GELU(),
            nn.BatchNorm3d(dim)
        ) for i in range(depth)]
    )

def Spatial_Mixer_Layer(dim, depth, kernel_size=(1, 7, 7), patch_size=7):
    return nn.Sequential(
        *[nn.Sequential(
            Residual(nn.Sequential(
                nn.Conv3d(dim, dim, kernel_size, groups=dim, dilation=(1, 2, 2), padding=(0, 2, 2)),
                nn.GELU(),
                nn.BatchNorm3d(dim)
            )),
            nn.Conv3d(dim, dim, kernel_size=1, padding=(0, 0, 0)),
            nn.GELU(),
            nn.BatchNorm3d(dim)
        ) for i in range(depth)]
    )

class Spectral_Spatial_Mixer_Block(nn.Module):
    '''For building Spectral Mixer Block
    '''
    def __init__(self, dim, depth, spec_kernel_size=(7, 1, 1), spat_kernel_size=(1, 7, 7), patch_size=7):
        super(Spectral_Spatial_Mixer_Block, self).__init__()
        self.Spectral_layers = Spectral_Mixer_Layer(dim, depth, kernel_size=spec_kernel_size, patch_size=patch_size)
        self.Spatial_layers = Spatial_Mixer_Layer(dim, depth, kernel_size=spat_kernel_size, patch_size=patch_size)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.Spectral_layers(x)
        x = self.Spatial_layers(x)
        return x

class HSI_Mixer_Net(nn.Module):
    def __init__(self, num_classes, img_size, patch_size, in_chans, embed_dim, depth=12, mlp_ratio=4.0, norm_layer=nn.LayerNorm, **kwargs):
        super(HSI_Mixer_Net, self).__init__()

        self.input_size = img_size
        self.embed_dim = embed_dim

        # Initialize components
        self.Linear_projection = Linear_projection(input_size=(patch_size, patch_size), patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        self.SSMB = Spectral_Spatial_Mixer_Block(dim=embed_dim, depth=depth)
        self.norm = norm_layer(embed_dim)

        # Classification head
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.Linear_projection(x)
        x = self.SSMB(x)
        x = torch.mean(x, dim=[2, 3, 4])  # Global average pooling over spatial dimensions
        x = self.norm(x)
        x = self.head(x)
        return x

def hsi(dataset, patch_size):
    model = None
    if dataset == 'sa':
        model = HSI_Mixer_Net(num_classes = 16, img_size = 12, patch_size = patch_size, in_chans = 204, embed_dim=28)
    elif dataset == 'pu':
        model = CNN3D(input_channels=103, n_classes=9, patch_size=patch_size)
    elif dataset == 'hsn':
        model = CNN3D(input_channels=144, n_classes=15, patch_size=patch_size)
    elif dataset == 'hrl':
        model = CNN3D(input_channels=176, n_classes=14, patch_size=patch_size)
    return model

if __name__ == '__main__':
    
    input = torch.randn(16, 1, 176, 9, 9)
    #input = torch.randn(1, 3, 224, 224)
    #net = HSI_Mixer_Net(num_classes = 16, img_size = 224, patch_size = 16, in_chans = 3, embed_dim=768)
    net = HSI_Mixer_Net(num_classes = 16, img_size = input.size()[3], patch_size = 3, in_chans = input.size()[1], embed_dim=28)    
    net.cuda()
    summary(net, (1,176,9,9))



"""
if __name__ == '__main__':
    t = torch.randn(size=(3, 1, 204, 7, 7))
    print("input shape:", t.shape)
    net = cnn3d(dataset='sa', patch_size=7)
    print("output shape:", net(t).shape)
"""

