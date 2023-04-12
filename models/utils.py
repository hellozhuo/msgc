from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def conv2d_out_dim(dim, kernel_size, padding=0, stride=1, dilation=1, ceil_mode=False):
    if ceil_mode:
        return int(math.ceil((dim + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1))
    else:
        return int(math.floor((dim + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1))

class GumbelSoftmax(nn.Module):
    '''
        gumbel softmax gate.
    '''
    def __init__(self, eps=1):
        super(GumbelSoftmax, self).__init__()
        self.eps = eps
        self.sigmoid = nn.Sigmoid()
    
    def gumbel_sample(self, template_tensor, eps=1e-8):
        uniform_samples_tensor = template_tensor.clone().uniform_()
        gumble_samples_tensor = torch.log(uniform_samples_tensor+eps)-torch.log(
                                          1-uniform_samples_tensor+eps)
        return gumble_samples_tensor
    
    def gumbel_softmax(self, logits):
        """ Draw a sample from the Gumbel-Softmax distribution"""
        gsamples = self.gumbel_sample(logits.data)
        logits = logits + gsamples
        soft_samples = self.sigmoid(logits / self.eps)
        return soft_samples, logits
    
    def forward(self, logits):
        if not self.training:
            out_hard = (logits>=0).float()
            return out_hard
        out_soft, prob_soft = self.gumbel_softmax(logits)
        out_hard = ((out_soft >= 0.5).float() - out_soft).detach() + out_soft
        return out_hard

class MaskGen(nn.Module):
    '''
        Decision Mask.
    '''
    def __init__(self, inplanes, outplanes, heads=4, eps=0.66667, bias=-1, squeeze_rate=4, pool=False):
        super(MaskGen, self).__init__()
        # Parameter
        self.bottleneck = inplanes // squeeze_rate 
        self.inplanes, self.outplanes, self.heads = inplanes, outplanes, heads

        # channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1) if pool else None
        self.conv = nn.Sequential(
            nn.Conv2d(inplanes, self.bottleneck, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(self.bottleneck),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.bottleneck, outplanes * heads, kernel_size=1, stride=1, bias=bias>=0),
        )
        if bias>=0:
            nn.init.constant_(self.conv[3].bias, bias)
        # Gate
        self.gate = GumbelSoftmax(eps=eps)

        self.flops = (inplanes + 1) * self.bottleneck + self.bottleneck * outplanes * heads
    
    def forward(self, x):
        batch = x.size(0)
        if self.avg_pool is not None:
            context = self.avg_pool(x) # [N, C, 1, 1] 
        else:
            context = x
        # transform
        mask = self.conv(context).view(batch, self.heads, self.outplanes) # [N, heads, C_out]
        # channel gate
        mask = self.gate(mask) # [N, heads, C_out]

        return mask

class AttGen(nn.Module):
    '''
        Attention Maps
    '''
    def __init__(self, inplanes, outplanes, heads=4, squeeze_rate=4, pool=False):
        super(AttGen, self).__init__()
        # Parameter
        self.bottleneck = inplanes // squeeze_rate 
        self.inplanes, self.outplanes, self.heads = inplanes, outplanes, heads

        # channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1) if pool else None
        self.conv = nn.Sequential(
            nn.Conv2d(inplanes, self.bottleneck, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(self.bottleneck),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.bottleneck, outplanes * heads, kernel_size=1, stride=1, bias=True),
        )
        nn.init.constant_(self.conv[3].weight, 0)
        nn.init.constant_(self.conv[3].bias, 1)

        self.flops = (inplanes + 1) * self.bottleneck + self.bottleneck * outplanes * heads
    
    def forward(self, x):
        batch = x.size(0)
        if self.avg_pool is not None:
            context = self.avg_pool(x) # [N, C, 1, 1] 
        else:
            context = x
        # transform
        att = self.conv(context).view(batch, self.heads, self.outplanes) # [N, heads, C_out]

        return att
