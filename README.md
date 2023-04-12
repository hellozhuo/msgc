# Middle Spectrum Grouped Convolution (MSGC)

This repository contains the PyTorch implementation for 
"Boosting Convolutional Neural Networks with Middle Spectrum Grouped Convolution" 
by 
[Zhuo Su](https://zhuogege1943.com/homepage/), 
[Jiehua Zhang](https://scholar.google.com/citations?user=UIbAv3wAAAAJ&hl=en&oi=sra), 
Tianpeng Liu,
Zhen Liu,
shuanghui zhang,
[Matti Pietikäinen](https://en.wikipedia.org/wiki/Matti_Pietik%C3%A4inen_(academic)) and 
[Li Liu](http://lilyliliu.com/)\*
(\* Corresponding author).


## Running environment

Training: Pytorch 1.9 with cuda 11.1 and cudnn 7.5 in an Ubuntu 18.04 system <br>

*Other versions may also work~ :)*


## Performance

The performances of MSGC equipped models (on ImageNet) are listed below. The checkpoints of our pretrained models can be downloaded at [link to our trained models](). For evaluation, 
please unzip the checkpoints to folder [checkpoints](checkpoints). 
The evaluation scripts to reproduce the following results can be found in [scripts.sh](scripts.sh).

| Model | Attention | Top-1 (%) | Top-5 (%) | MAC | Training logs |
|-------|-------|-------|-------|-----|-------------|
| ResNet-18 | - | 69.76 | 89.08 | 1817 M | - |
| ResNet-18 + MSGC | &cross; | 70.30 | 89.27 | 883 M | [log]() |
| ResNet-18 + MSGC | &check; | 71.51 | 90.21 | 885 M | [log]() |
| ResNet-18 + MSGC | &check; | 72.33 | 90.53 | 1630 M | [log]() |
| | | | | | |
| ResNet-50 | - | 76.13 | 92.86 | 4099 M | - |
| ResNet-50 + MSGC | &cross; | 77.20 | 93.37 | 1886 M | [log]() |
| ResNet-50 + MSGC | &check; | 76.76 | 92.99 | 1892 M | [log]() |
| | | | | | |
| MobileNetV2 | - | 71.88 | 90.27 | 307 M | - |
| MobileNetV2 + MSGC | &cross; | 72.10 | 90.41 | 198 M | [log]() |
| MobileNetV2 + MSGC | &check; | 72.59 | 90.82 | 197 M | [log]() |
| | | | | | |
| CondenseNet | - | 73.80 | 91.70 | 529 M | - |
| CondenseNet + MSGC | &cross; | 74.81 | 92.17 | 523 M | [log]() |

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

