import torch
from torch import Tensor
import torch.nn as nn
from torchvision.models.utils import load_state_dict_from_url
from typing import Type, Any, Callable, Union, List, Optional

from .utils import MaskGen, AttGen, conv2d_out_dim
from .net_config import Config_resnet

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-b627a593.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-0676ba61.pth',
}


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class DyBasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        input_size: tuple = (None, None),
        config: any = None
    ) -> None:
        super(DyBasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.downsample = downsample
        self.stride = stride
        self.attention = config.attention

        h, w = input_size
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        if self.attention:
            self.flops_dgc1 = inplanes * h * w # this term is for att
            self.flops_original_extra = inplanes * h * w
        else:
            self.flops_dgc1 = 0
            self.flops_original_extra = 0
        self.flops_mask = h * w * inplanes

        h = conv2d_out_dim(h, 3, 1, stride)
        w = conv2d_out_dim(w, 3, 1, stride)
        self.output_size = (h, w)
        self.flops_dgc1 += 9 * inplanes * planes * h * w + planes * h * w

        # dynamic conv
        self.conv2 = conv3x3(planes * config.heads, planes, groups=config.heads)
        self.bn2 = norm_layer(planes)
        if self.attention:
            self.flops_dgc2 = (9 * planes + 1) * planes * h * w \
                    + config.heads * planes * h * w # the 2nd term is for att
            self.flops_original_extra += config.heads * planes * h * w
        else:
            self.flops_dgc2 = (9 * planes + 1) * planes * h * w

        self.flops_dgc = self.flops_dgc1 + self.flops_dgc2

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # mask generator
        self.maskgen1 = MaskGen(inplanes, inplanes, 1, config.eps, config.bias)
        self.maskgen2 = MaskGen(inplanes, planes, config.heads, config.eps, config.bias)
        self.flops_mask += self.maskgen1.flops + self.maskgen2.flops 

        # attention generator
        self.flops_att = 0
        if self.attention:
            self.attgen1 = AttGen(inplanes, inplanes, 1)
            self.attgen2 = AttGen(inplanes, planes, config.heads)
            self.flops_att = self.attgen1.flops + self.attgen2.flops

    def get_others(self, mask1, mask2, others):
        flops_dgc_, bonus_ = others
        flops_dgc1 = mask1.mean() * self.flops_dgc1
        flops_dgc2 = mask2.mean() * self.flops_dgc2

        mask_d = mask2.detach()
        mask_bonus = 1.0 - mask_d[:, 0, :]
        for i in range(1, mask_d.size(1)):
            mask_bonus = mask_bonus * (1.0 - mask_d[:, i, :]) # b, planes
        bonus = mask_bonus.mean() * flops_dgc1.detach()
        return flops_dgc_ + flops_dgc1 + flops_dgc2, bonus_ + bonus

    def forward(self, x_others):
        x, others = x_others
        identity = x
        
        x_pool = self.avg_pool(x)
        mask1 = self.maskgen1(x_pool) # b, 1, inplanes
        mask2 = self.maskgen2(x_pool) # b, heads, planes
        _, _, inplanes = mask1.size()
        b, heads, planes = mask2.size()

        if self.attention:
            att1 = self.attgen1(x_pool) # b, 1, inplanes
            att2 = self.attgen2(x_pool) # b, heads, planes
            out = self.conv1(x * mask1.view(b, inplanes, 1, 1) * att1.view(b, inplanes, 1, 1))
        else:
            out = self.conv1(x * mask1.view(b, inplanes, 1, 1))

        out = self.bn1(out)
        out = self.relu(out)

        outcat = []
        for i in range(heads):
            if self.attention:
                outmask = out * mask2[:, i, :].view(b, planes, 1, 1) * att2[:, i, :].view(b, planes, 1, 1)
            else:
                outmask = out * mask2[:, i, :].view(b, planes, 1, 1)
            outcat.append(outmask)
        out = torch.cat(outcat, dim=1) # b, heads*planes, h, w

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        flops_dgc, bonus = self.get_others(mask1, mask2, others)

        return out, [flops_dgc, bonus]


class DyBottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        input_size: tuple = (None, None),
        config: any = None
    ) -> None:
        super(DyBottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        self.downsample = downsample
        self.stride = stride
        self.attention = config.attention

        h, w = input_size
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.flops_dgc1 = inplanes * width * h * w + width * h * w
        if self.attention:
            self.flops_dgc2 = config.heads * width * h * w
            self.flops_original_extra = config.heads * width * h * w
        else:
            self.flops_dgc2 = 0
            self.flops_original_extra = 0
        self.flops_mask = h * w * inplanes # for the avgpool in the MaskGen

        self.conv2 = conv3x3(width * config.heads, width, stride, groups=config.heads)
        self.bn2 = norm_layer(width)
        h = conv2d_out_dim(h, 3, 1, stride)
        w = conv2d_out_dim(w, 3, 1, stride)
        self.output_size = (h, w)
        self.flops_dgc2 += 9 * width * width * h * w + width * h * w

        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.flops_dgc3 = (width + 1) * planes * self.expansion * h * w

        self.flops_dgc = self.flops_dgc1 + self.flops_dgc2 + self.flops_dgc3

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # mask generator 
        self.maskgen1 = MaskGen(inplanes, inplanes, 1, config.eps, config.bias, 8)
        self.maskgen2 = MaskGen(inplanes, width, config.heads, config.eps, config.bias, 8)
        self.maskgen3 = MaskGen(inplanes, width, 1, config.eps, config.bias, 8)
        self.flops_mask += self.maskgen1.flops + self.maskgen2.flops + self.maskgen3.flops

        # attention generator
        self.flops_att = 0
        if self.attention:
            self.attgen2 = AttGen(inplanes, width, config.heads, 8)
            self.flops_att = self.attgen2.flops

        self.relu = nn.ReLU(inplace=True)

    def get_others(self, mask1, mask2, mask3, others):
        flops_dgc_, bonus_ = others
        flops_dgc1 = mask1.mean() * self.flops_dgc1
        flops_dgc2 = mask2.mean() * self.flops_dgc2
        flops_dgc3 = mask3.mean() * self.flops_dgc3

        mask2_d = mask2.detach()
        heads = mask2_d.size(1)
        mask2_bonus = 1.0 - mask2_d[:, 0, :]
        for i in range(1, heads):
            mask2_bonus = mask2_bonus * (1.0 - mask2_d[:, i, :])
        bonus2 = mask2_bonus.mean() * flops_dgc1.detach()

        mask3_d = mask3.detach()
        mask3_bonus = 1.0 - mask3_d[:, 0, :]
        head_width = mask3_d.size(2) // heads
        bonus3 = 0.0
        for i in range(heads):
            start_ = i * head_width
            end_ = start_ + head_width
            mask3_bonus_head = mask3_bonus[:, start_: end_]
            bonus3 += mask2_d[:, i, :].mean() * self.flops_dgc2 / heads * mask3_bonus_head.mean()

        flops_dgc = flops_dgc_ + flops_dgc1 + flops_dgc2 + flops_dgc3
        bonus = bonus_ + bonus2 + bonus3
        return flops_dgc, bonus

    def forward(self, x_others):
        x, others = x_others
        identity = x

        x_pool = self.avg_pool(x)
        mask1 = self.maskgen1(x_pool)
        mask2 = self.maskgen2(x_pool)
        mask3 = self.maskgen3(x_pool)
        inplanes = mask1.size(2)
        b, heads, width = mask2.size()

        if self.attention:
            att2 = self.attgen2(x_pool)

        out = self.conv1(x * mask1.view(b, inplanes, 1, 1))
        out = self.bn1(out)
        out = self.relu(out)

        outcat = []
        for i in range(heads):
            if self.attention:
                outmask = out * mask2[:, i, :].view(b, width, 1, 1) * att2[:, i, :].view(b, width, 1, 1)
            else:
                outmask = out * mask2[:, i, :].view(b, width, 1, 1)
            outcat.append(outmask)
        out = torch.cat(outcat, dim=1)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out * mask3.view(b, width, 1, 1))
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        flops_dgc, bonus = self.get_others(mask1, mask2, mask3, others)

        return out, [flops_dgc, bonus]


class ResNet(nn.Module):

    def __init__(
        self,
        block: Type[Union[DyBasicBlock, DyBottleneck]],
        layers: List[int],
        config,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.config = config

        self.inplanes = 64
        self.dilation = 1
        h, w = config.input_size
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        h = conv2d_out_dim(h, 7, 3, 2)
        w = conv2d_out_dim(w, 7, 3, 2)
        self.flops_conv1 = 49 * 3 * self.inplanes * h * w
        print('Conv 1st: h {}, w {}, flops {}'.format(h, w, self.flops_conv1))
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.flops_conv1 += self.inplanes * h * w
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        h = conv2d_out_dim(h, 3, 1, 2)
        w = conv2d_out_dim(w, 3, 1, 2)
        self.flops_conv1 += 4 * self.inplanes * h * w
        self.flops_dgc = 0
        self.flops_mask = 0
        self.flops_original_extra = 0
        self.flops_downsample = 0
        self.layer1, (h, w) = self._make_layer(block, 64, layers[0], input_size=(h, w))
        self.layer2, (h, w) = self._make_layer(block, 128, layers[1], stride=2, input_size=(h, w),
                                       dilate=replace_stride_with_dilation[0])
        self.layer3, (h, w) = self._make_layer(block, 256, layers[2], stride=2, input_size=(h, w),
                                       dilate=replace_stride_with_dilation[1])
        self.layer4, (h, w) = self._make_layer(block, 512, layers[3], stride=2, input_size=(h, w),
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, config.num_classes)
        self.flops_fc = 512 * block.expansion * (h * w + config.num_classes)
        print('classifier: flops {}'.format(self.flops_fc))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        for m in self.modules():
            if isinstance(m, AttGen):
                nn.init.constant_(m.conv[3].weight, 0)
                nn.init.constant_(m.conv[3].bias, 1)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, DyBottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, DyBasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: Type[Union[DyBasicBlock, DyBottleneck]], planes: int, blocks: int,
            stride: int = 1, input_size: tuple = (None, None), dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        h, w = input_size
        print('input h and w: {} {}'.format(h, w))
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
            hd = conv2d_out_dim(h, 1, 0, stride)
            wd = conv2d_out_dim(w, 1, 0, stride)
            self.flops_downsample += self.inplanes * planes * block.expansion * hd * wd

        layers = []
        the_block = block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, 
                            input_size=(h, w), config=self.config)
        layers.append(the_block)
        h, w = the_block.output_size
        print('Block: h {}, w {}, flops {}'.format(h, w, the_block.flops_dgc))
        self.flops_dgc += the_block.flops_dgc
        self.flops_mask += the_block.flops_mask + the_block.flops_att
        self.flops_original_extra += the_block.flops_original_extra
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            the_block = block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, input_size=(h, w), config=self.config)
            layers.append(the_block)
            h, w = the_block.output_size
            print('Block: h {}, w {}, flops {}'.format(h, w, the_block.flops_dgc))
            self.flops_dgc += the_block.flops_dgc
            self.flops_mask += the_block.flops_mask + the_block.flops_att
            self.flops_original_extra += the_block.flops_original_extra

        return nn.Sequential(*layers), (h, w)

    def get_flops(self):
        flops_main = self.flops_conv1 + self.flops_downsample + self.flops_fc
        flops = flops_main + self.flops_dgc - self.flops_original_extra
        flops_possible = flops_main + self.flops_dgc * 0.25 + self.flops_mask
        return flops, flops_possible, flops_main, self.flops_dgc, self.flops_mask

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = [x, [0, 0]]
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x, others = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x, others

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def msgc_resnet18(args):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    """
    config = Config_resnet(args)
    model = ResNet(DyBasicBlock, [2, 2, 2, 2], config)
    if not args.scratch:
        pretrained_dict = load_state_dict_from_url(model_urls['resnet18'], progress=True)
        model_dict = model.state_dict()
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print('load pretrained model successfully')

    return model

def msgc_resnet50(args):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    """
    config = Config_resnet(args)
    model = ResNet(DyBottleneck, [3, 4, 6, 3], config)
    if not args.scratch:
        pretrained_dict = load_state_dict_from_url(model_urls['resnet50'], progress=True)
        model_dict = model.state_dict()
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print('load pretrained model successfully')

    return model
