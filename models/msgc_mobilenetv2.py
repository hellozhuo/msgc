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
from .net_config import Config_mobilenetv2

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class ConvBNReLU_1st(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU_1st, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True))
    
    def forward(self, x_others):
        x, others = x_others
        x = super(ConvBNReLU_1st, self).forward(x)
        return x, others


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )
        
class DyInvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, input_size, config):
        super(DyInvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        h, w = input_size
        self.attention = config.attention

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        self.conv = nn.ModuleList()
        self.flops_mask = inp * h * w
        ## pw
        self.flops_pw = 0
        if expand_ratio != 1:
            self.conv.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
            self.flops_pw = inp * hidden_dim * h * w + hidden_dim * h * w
        ## dw
        self.conv.append(ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim))
        h = conv2d_out_dim(h, 3, 1, stride)
        w = conv2d_out_dim(w, 3, 1, stride)
        self.output_size = (h, w)
        self.flops_dw = 9 * hidden_dim * h * w + hidden_dim * h * w

        ## dynamic group conv with bn
        self.conv.append(nn.Conv2d(config.heads * hidden_dim, oup, 1, 1, 0, groups=config.heads, bias=False))
        self.conv.append(nn.BatchNorm2d(oup))
        if self.attention:
            self.flops_dgc = hidden_dim * (oup + config.heads) * h * w # the 2nd term is for att
            self.flops_original_extra = config.heads * hidden_dim * h * w
        else:
            self.flops_dgc = hidden_dim * oup * h * w
            self.flops_original_extra = 0

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        ## mask generator
        self.maskgen = MaskGen(inp, hidden_dim, config.heads, config.eps, config.bias)
        self.flops_mask += self.maskgen.flops

        ## attention generator
        self.flops_att = 0
        if self.attention:
            self.attgen = AttGen(inp, hidden_dim, config.heads)
            self.flops_att = self.attgen.flops

    def get_others(self, mask, others):
        flops_dgc_, bonus_ = others
        flops_dgc = mask.mean() * self.flops_dgc
        mask_d = mask.detach()
        mask_bonus = 1.0 - mask_d[:, 0, :]
        for i in range(1, mask_d.size(1)):
            mask_bonus = mask_bonus * (1.0 - mask_d[:, i, :]) # b, hidden_dim
        bonus = mask_bonus.mean() * (self.flops_pw + self.flops_dw)
        return flops_dgc_ + flops_dgc, bonus_ + bonus

    def forward(self, x_others):
        x, others = x_others
        x_ = x

        x_pool = self.avg_pool(x)
        mask = self.maskgen(x_pool) # b, heads, hidden_dim
        b, heads, hidden_dim = mask.size()

        if self.attention:
            att = self.attgen(x_pool)

        #x = self.conv(x) # b, hidden_dim, h, w
        for layer in self.conv[:-2]:
            x = layer(x)

        xcat = []
        for i in range(heads):
            if self.attention:
                xmask = x * mask[:, i, :].view(b, hidden_dim, 1, 1) * att[:, i, :].view(b, hidden_dim, 1, 1)
            else:
                xmask = x * mask[:, i, :].view(b, hidden_dim, 1, 1)
            xcat.append(xmask)
        x = torch.cat(xcat, dim=1) # b, heads*hidden_dim, h, w
        for layer in self.conv[-2:]:
            x = layer(x)

        if self.use_res_connect:
            x = x_ + x

        flops_dgc, bonus = self.get_others(mask, others)

        return x, [flops_dgc, bonus]

class DyMobileNetV2(nn.Module):
    def __init__(self,
                 config,
                 inverted_residual_setting=None,
                 round_nearest=8,
                 block=None):
        """
        MobileNet V2 main class
        Args:
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet
        """
        super(DyMobileNetV2, self).__init__()
        width_mult = config.width_mult
        
        if block is None:
            block = DyInvertedResidual
        input_channel = 32
        last_channel = 1280
        h, w = config.input_size

        if inverted_residual_setting is None:
            if "cifar" in config.data:
                inverted_residual_setting = [
                    # t, c, n, s
                    [1, 16, 1, 1],
                    [6, 24, 2, 1],
                    [6, 32, 3, 2],
                    [6, 64, 4, 1],
                    [6, 96, 3, 2],
                    [6, 160, 3, 1],
                    [6, 320, 1, 1],
                ]
                init_stride = 1
            else:
                inverted_residual_setting = [
                    # t, c, n, s
                    [1, 16, 1, 1],
                    [6, 24, 2, 2],
                    [6, 32, 3, 2],
                    [6, 64, 4, 2],
                    [6, 96, 3, 1],
                    [6, 160, 3, 2],
                    [6, 320, 1, 1],
                ]
                init_stride = 2

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features = [ConvBNReLU_1st(3, input_channel, stride=init_stride)]
        h = conv2d_out_dim(h, 3, 1, init_stride)
        w = conv2d_out_dim(w, 3, 1, init_stride)
        self.flops_1st = 3 * input_channel * 9 * h * w + input_channel * h * w
        print('Conv 1st: h {}, w {}, flops {}'.format(h, w, self.flops_1st))
        # building inverted residual blocks
        self.flops_pwdw = 0
        self.flops_dgc = 0
        self.flops_mask = 0
        self.flops_original_extra = 0
        for k, [t, c, n, s] in enumerate(inverted_residual_setting):
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                the_block = block(input_channel, output_channel, stride, expand_ratio=t, input_size=(h, w), config=config)
                features.append(the_block)
                input_channel = output_channel
                h, w = the_block.output_size
                flops_pwdw = the_block.flops_pw + the_block.flops_dw
                print('Block {}-{}: h {}, w {}, pwdw flops {}, dgc flops {}'.format(
                    k, i, h, w, flops_pwdw, the_block.flops_dgc))
                self.flops_pwdw += flops_pwdw
                self.flops_dgc += the_block.flops_dgc
                self.flops_mask += the_block.flops_mask + the_block.flops_att
                self.flops_original_extra += the_block.flops_original_extra

        # building last several layers
        features.append(ConvBNReLU_1st(input_channel, self.last_channel, kernel_size=1))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0),
            nn.Linear(self.last_channel, config.num_classes),
        )
        self.flops_last = h * w * input_channel * self.last_channel + self.last_channel * h * w +\
                self.last_channel * config.num_classes + self.last_channel * h * w
        print('Conv last and classifier: h {}, w {}, flops {}'.format(h, w, self.flops_last))

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                #if m.bias is not None:
                #    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.out_features == config.num_classes:
                    nn.init.zeros_(m.bias)
        for m in self.modules():
            if isinstance(m, AttGen):
                nn.init.constant_(m.conv[3].weight, 0)
                nn.init.constant_(m.conv[3].bias, 1)

    def get_flops(self):
        flops_main = self.flops_1st + self.flops_pwdw + self.flops_last
        flops = flops_main + self.flops_dgc - self.flops_original_extra
        flops_possible = flops_main + self.flops_dgc * 0.25 + self.flops_mask
        return flops, flops_possible, flops_main, self.flops_dgc, self.flops_mask


    def forward(self, x):
        x_others = [x, [0, 0]]
        x, others = self.features(x_others)
        x = x.mean([2, 3])
        x = self.classifier(x)
        return x, others

def msgc_mobilenetv2(args):
    config = Config_mobilenetv2(args)
    model = DyMobileNetV2(config)
    if not args.scratch:
        url = 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth'
        pretrained_dict = load_state_dict_from_url(url, progress=True)
        model_dict = model.state_dict()
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print('load pretrained model successfully')

    return model
