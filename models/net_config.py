from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

class Config_densenet:
    def __init__(self, args):

        self.bottleneck = 4
        self.stages = [4, 6, 8, 10, 8]
        self.growth = [8, 16, 32, 64, 128]

        self.heads = args.heads
        self.num_classes = args.num_classes
        self.eps = 2.0 / 3
        self.bias = 2.0
        self.input_size = None
        self.attention = args.attention
        if 'imagenet' in args.data:
            self.input_size = (224, 224)
        elif 'cifar' in args.data:
            self.input_size = (32, 32)
        self.data = args.data

class Config_mobilenetv2:
    def __init__(self, args):

        self.heads = args.heads
        self.num_classes = args.num_classes
        self.width_mult = args.width_mul
        self.eps = 2.0 / 3
        self.bias = 2.0
        self.input_size = None
        self.attention = args.attention
        if 'imagenet' in args.data:
            self.input_size = (224, 224)
        elif 'cifar' in args.data:
            self.input_size = (32, 32)
        self.data = args.data

class Config_resnet:
    def __init__(self, args):

        self.heads = args.heads
        self.num_classes = args.num_classes
        self.eps = 2.0 / 3
        self.bias = 2.0
        self.input_size = None
        self.attention = args.attention
        if 'imagenet' in args.data:
            self.input_size = (224, 224)
        elif 'cifar' in args.data:
            self.input_size = (32, 32)
        self.data = args.data
