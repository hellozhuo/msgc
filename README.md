# Middle Spectrum Grouped Convolution (MSGC)

This repository contains the PyTorch implementation for 
[Boosting Convolutional Neural Networks with Middle Spectrum Grouped Convolution] (https://arxiv.org/abs/2304.06305)
by 
[Zhuo Su](https://zhuogege1943.com/homepage/), 
[Jiehua Zhang](https://scholar.google.com/citations?user=UIbAv3wAAAAJ&hl=en&oi=sra), 
Tianpeng Liu,
Zhen Liu,
Shuanghui Zhang,
[Matti Pietikäinen](https://en.wikipedia.org/wiki/Matti_Pietik%C3%A4inen_(academic)), 
and [Li Liu](http://lilyliliu.com/) (corresponding author).


## Running environment

Ubuntu 18.04 system + cuda 11.1 and cudnn 8.2.1 + Pytorch 1.9 + python 3.9

*Other versions may also work~ :)*


## Performance

The performances of MSGC equipped models (on ImageNet) are listed below. The checkpoints of our trained models can be downloaded at [link to our trained models](https://github.com/hellozhuo/msgc/releases/download/v1.0/trained_models_imagenet.zip). For evaluation, 
please unzip the checkpoints to folder [checkpoints](checkpoints). 
The evaluation scripts to reproduce the following results can be found in [scripts.sh](scripts.sh).

| Model | Attention | Top-1 (%) | Top-5 (%) | MAC | Training script | Training log |
|-------|-------|-------|-------|-----|-------------|-------------|
| ResNet-18 | - | 69.76 | 89.08 | 1817 M | - | - |
| ResNet-18 + MSGC | &cross; | 70.30 | 89.27 | 883 M | [script](https://github.com/hellozhuo/msgc/blob/092f46e4e115bfdcbc73546c309267996fa86dd2/scripts.sh#L32) | [log](logs/msgc_resnet18_noatt_log.txt) |
| ResNet-18 + MSGC | &check; | 71.51 | 90.21 | 885 M | [script](https://github.com/hellozhuo/msgc/blob/092f46e4e115bfdcbc73546c309267996fa86dd2/scripts.sh#L35) | [log](logs/msgc_resnet18_att_log.txt) |
| ResNet-18 + MSGC | &check; | 72.33 | 90.53 | 1630 M | [script](https://github.com/hellozhuo/msgc/blob/092f46e4e115bfdcbc73546c309267996fa86dd2/scripts.sh#L53) | [log](logs/msgc_resnet18_noatt_tau0_9_log.txt) |
| | | | | | |
| ResNet-50 | - | 76.13 | 92.86 | 4099 M | - | - |
| ResNet-50 + MSGC | &cross; | 77.20 | 93.37 | 1886 M | [script](https://github.com/hellozhuo/msgc/blob/092f46e4e115bfdcbc73546c309267996fa86dd2/scripts.sh#L38) | [log](logs/msgc_resnet50_noatt_log.txt) |
| ResNet-50 + MSGC | &check; | 76.76 | 92.99 | 1892 M | [script](https://github.com/hellozhuo/msgc/blob/092f46e4e115bfdcbc73546c309267996fa86dd2/scripts.sh#L41) | [log](logs/msgc_resnet50_att_log.txt) |
| | | | | | |
| MobileNetV2 | - | 71.88 | 90.27 | 307 M | - | - |
| MobileNetV2 + MSGC | &cross; | 72.10 | 90.41 | 198 M | [script](https://github.com/hellozhuo/msgc/blob/092f46e4e115bfdcbc73546c309267996fa86dd2/scripts.sh#L44) | [log](logs/msgc_mobilenetv2_noatt_log.txt) |
| MobileNetV2 + MSGC | &check; | 72.59 | 90.82 | 197 M | [script](https://github.com/hellozhuo/msgc/blob/092f46e4e115bfdcbc73546c309267996fa86dd2/scripts.sh#L47) | [log](logs/msgc_mobilenetv2_att_log.txt) |
| | | | | | |
| CondenseNet | - | 73.80 | 91.70 | 529 M | - | - |
| CondenseNet + MSGC | &cross; | 74.81 | 92.17 | 523 M | [script](https://github.com/hellozhuo/msgc/blob/092f46e4e115bfdcbc73546c309267996fa86dd2/scripts.sh#L50) | [log](logs/msgc_condensenet_noatt_log.txt) |

## Training

An example script for training on two gpus is:
```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
--master_port=12345 \
--nproc_per_node=2 \
main_dist.py \
--model msgc_resnet18 \
--attention \
-j 16 \
--data imagenet \
--datadir /to/imagenet/dataset \
--savedir ./results \
--resume \
--target 0.5
```

The above script trains the MSGC equipped ResNet-18 architecture with a target MAC reduction of 50%.<br>
Other training scripts can be seen in [scripts.sh](scripts.sh). 

For more detailed illustraion of the training set up, please refer to [main\_dist.py](main_dist.py), or run:
```bash
python main_dist.py -h
```

## Acknowledgement

The coding is inspired by:

- [Pixel Difference Convolution](https://github.com/zhuoinoulu/pidinet)
- [Dynamic Grouped Convolution](https://github.com/hellozhuo/dgc)
- [Detectron2](https://github.com/facebookresearch/detectron2)

