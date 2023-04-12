from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.utils import load_state_dict_from_url
import math

from .utils import MaskGen, AttGen, conv2d_out_dim
from .net_config import Config_densenet

class _Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=1, 
                 stride=1, padding=0, groups=1):
        super(_Conv, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(in_channels))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(in_channels * groups, out_channels,
                                          kernel_size=kernel_size,
                                          stride=stride,
                                          padding=padding, bias=False,
                                          groups=groups))

class _DyDenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate, bottleneck, input_size, config):
        super(_DyDenseLayer, self).__init__()

        h, w = input_size
        self.attention = config.attention

        hidden_dim = bottleneck * growth_rate

        ## 1x1 conv i --> b*k
        self.conv_1 = _Conv(in_channels, hidden_dim)

        ## 3x3 conv b*k --> k
        self.conv_2 = _Conv(hidden_dim, growth_rate, 
                kernel_size=3, padding=1, groups=config.heads)
        self.flops_dgc1 = in_channels * hidden_dim * h * w
        self.flops_conv1_relu =  in_channels * h * w
        if self.attention:
            self.flops_dgc2 = hidden_dim * (9 * growth_rate + config.heads + 1) * h * w # +1: for relu
            self.flops_original_extra = config.heads * hidden_dim * h * w
        else:
            self.flops_dgc2 = hidden_dim * (9 * growth_rate + 1) * h * w
            self.flops_original_extra = 0

        self.flops_dgc = self.flops_dgc1 + self.flops_dgc2

        squeeze_rate = 8 if in_channels >= 200 else 4
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        ## mask generator
        self.maskgen1 = MaskGen(in_channels, in_channels, 1, config.eps, config.bias, squeeze_rate)
        self.maskgen2 = MaskGen(in_channels, hidden_dim, config.heads, config.eps, config.bias, squeeze_rate)
        self.flops_mask = self.maskgen1.flops + self.maskgen2.flops + h * w * in_channels

        ## attention generator
        self.flops_att = 0
        if self.attention:
            self.attgen2 = AttGen(in_channels, hidden_dim, config.heads, squeeze_rate)
            self.flops_att = self.attgen2.flops

    def get_others(self, mask1, mask2, others):
        flops_dgc_, bonus_ = others
        flops_dgc1 = mask1.mean() * self.flops_dgc1
        flops_dgc2 = mask2.mean() * self.flops_dgc2

        mask2_d = mask2.detach()
        heads = mask2_d.size(1)
        mask_bonus = 1.0 - mask2_d[:, 0, :]
        for i in range(heads):
            mask_bonus = mask_bonus * (1.0 - mask2_d[:, i, :])
        bonus = mask_bonus.mean() * flops_dgc1.detach()

        flops_dgc = flops_dgc_ + flops_dgc1 + flops_dgc2
        bonus = bonus_ + bonus

        return flops_dgc, bonus

    def forward(self, x_others):
        x, others = x_others
        x_ = x

        x = self.conv_1.relu(self.conv_1.norm(x))

        x_pool = self.avg_pool(x)
        mask1 = self.maskgen1(x_pool)
        mask2 = self.maskgen2(x_pool)
        in_channels = mask1.size(2)
        b, heads, hidden_dim = mask2.size()
        if self.attention:
            att2 = self.attgen2(x_pool)

        x = self.conv_1.conv(x * mask1.view(b, in_channels, 1, 1))
        x = self.conv_2.relu(self.conv_2.norm(x))

        xcat = []
        for i in range(heads):
            if self.attention:
                xmask = x * mask2[:, i, :].view(b, hidden_dim, 1, 1) * att2[:, i, :].view(b, hidden_dim, 1, 1)
            else:
                xmask = x * mask2[:, i, :].view(b, hidden_dim, 1, 1)
            xcat.append(xmask)
        x = torch.cat(xcat, dim=1)
        x = self.conv_2.conv(x)
        x = torch.cat([x_, x], 1)

        flops_dgc, bonus = self.get_others(mask1, mask2, others)
        return x, [flops_dgc, bonus]


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, in_channels, growth_rate, 
            bottleneck=4, input_size=None, config=None):
        super(_DenseBlock, self).__init__()
        assert config is not None, 'config should not be None'
        assert input_size is not None, 'input size should not be None'
        self.flops_dgc = 0
        self.flops_mask = 0
        self.flops_conv1_relu = 0
        self.flops_original_extra = 0
        for i in range(num_layers):
            layer = _DyDenseLayer(in_channels + i * growth_rate, growth_rate, 
                    bottleneck, input_size, config)
            self.add_module('denselayer_%d' % (i + 1), layer)
            self.flops_dgc += layer.flops_dgc
            self.flops_mask += layer.flops_mask + layer.flops_att
            self.flops_conv1_relu += layer.flops_conv1_relu
            self.flops_original_extra += layer.flops_original_extra

class _Transition(nn.Module):
    def __init__(self, in_channels):
        super(_Transition, self).__init__()
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x_others):
        x, others = x_others
        x = self.pool(x)
        return x, others

class InitConv(nn.Conv2d):

    def forward(self, x_others):
        x, others = x_others
        x = super(InitConv, self).forward(x)
        return x, others

class DyDenseNet(nn.Module):
    def __init__(self, config):

        super(DyDenseNet, self).__init__()

        num_classes = config.num_classes
        self.stages = config.stages
        self.growth = config.growth
        assert len(self.stages) == len(self.growth)

        if 'cifar' in config.data:
            self.init_stride = 1
            self.pool_size = 8
        else:
            self.init_stride = 2
            self.pool_size = 7
        h, w = config.input_size

        self.features = nn.Sequential()
        ### Initial nChannels should be 3
        self.num_features = 2 * self.growth[0]
        ### Dense-block 1 (224x224)
        self.features.add_module('init_conv', InitConv(3, self.num_features,
                                                        kernel_size=3,
                                                        stride=self.init_stride,
                                                        padding=1,
                                                        bias=False))
        h = conv2d_out_dim(h, 3, 1, self.init_stride)
        w = conv2d_out_dim(w, 3, 1, self.init_stride)
        self.flops_init = 9 * 3 * self.num_features * h * w
        print('Init Conv: h {}, w {}, flops {}'.format(h, w, self.flops_init))

        self.flops_dgc = 0
        self.flops_mask = 0
        self.flops_original_extra = 0
        self.flops_pool = 0
        self.flops_block_relu = 0

        self.features_last = nn.Sequential()
        for i in range(len(self.stages)):
            ### Dense-block i
            h, w = self.add_block(i, config, (h, w))
        ### Linear layer
        self.classifier = nn.Linear(self.num_features, num_classes)

        self.flops_classifier = self.num_features * num_classes

        ### initialize
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
        for m in self.modules():
            if isinstance(m, AttGen):
                nn.init.constant_(m.conv[3].weight, 0)
                nn.init.constant_(m.conv[3].bias, 1)

    def add_block(self, i, config, input_size):
        ### Check if ith is the last one
        h, w = input_size
        last = (i == len(self.stages) - 1)
        block = _DenseBlock(
            num_layers=self.stages[i],
            in_channels=self.num_features,
            growth_rate=self.growth[i],
            input_size=(h, w),
            config=config
        )
        self.features.add_module('denseblock_%d' % (i + 1), block)
        self.num_features += self.stages[i] * self.growth[i]
        self.flops_dgc += block.flops_dgc
        self.flops_mask += block.flops_mask
        self.flops_block_relu += block.flops_conv1_relu
        self.flops_original_extra += block.flops_original_extra
        print('Block: h {}, w {}, flops {}'.format(h, w, block.flops_dgc))
        if not last:
            trans = _Transition(in_channels=self.num_features)
            self.features.add_module('transition_%d' % (i + 1), trans)
            h = conv2d_out_dim(h, 2, 0, 2)
            w = conv2d_out_dim(w, 2, 0, 2)
            flops_pool = 4 * self.num_features * h * w
            self.flops_pool += flops_pool
            print('Pool: h {}, w {}, flops {}'.format(h, w, flops_pool))
        else:
            self.features_last.add_module('norm_last',
                                     nn.BatchNorm2d(self.num_features))
            self.features_last.add_module('relu_last',
                                     nn.ReLU(inplace=True))
            self.features_last.add_module('pool_last',
                                     nn.AvgPool2d(self.pool_size))
            flops_pool = self.pool_size ** 2 * self.num_features * 2 # relu and pool
            self.flops_pool += flops_pool
            print('AvgPool: h {}, w {}, flops {}'.format(h, w, flops_pool))

        return (h, w)

    def get_flops(self):
        flops_main = self.flops_init + self.flops_pool + self.flops_classifier + self.flops_block_relu
        flops = flops_main + self.flops_dgc - self.flops_original_extra
        flops_possible = flops_main + self.flops_dgc * 0.25 + self.flops_mask
        return flops, flops_possible, flops_main, self.flops_dgc, self.flops_mask

    def forward(self, x):
        x = [x, [0, 0]]
        features, others = self.features(x)
        features = self.features_last(features)

        out = features.view(features.size(0), -1)
        out = self.classifier(out)
        return out, others

def msgc_condensenet(args):
    config = Config_densenet(args)
    model = DyDenseNet(config)
    if not args.scratch:
        url = 'https://github.com/hellozhuo/msgc/releases/download/v0.1/pretrained_densenet74.pth'
        pretrained_dict = load_state_dict_from_url(url, progress=True)
        model_dict = model.state_dict()
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print('load pretrained model successfully')

    return model
