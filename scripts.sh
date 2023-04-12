
## Evaluation

# ResNet-18 with MSGC w/o attention (Top-1 = 70.3%, MAC = 883 M)
#CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --master_port=12345 --nproc_per_node=1 main_dist.py --model msgc_resnet18 --data imagenet --datadir /to/imagenet/dataset --savedir ./results --evaluate checkpoints/msgc_resnet18_noatt.pth

# ResNet-18 with MSGC w/ attention (Top-1 = 71.5%, MAC = 885 M)
#CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --master_port=12345 --nproc_per_node=1 main_dist.py --model msgc_resnet18 --attention --data imagenet --datadir /to/imagenet/dataset --savedir ./results --evaluate checkpoints/msgc_resnet18_att.pth

# ResNet-50 with MSGC w/o attention (Top-1 = 77.2%, MAC = 1886 M)
#CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --master_port=12345 --nproc_per_node=1 main_dist.py --model msgc_resnet50 --data imagenet --datadir /to/imagenet/dataset --savedir ./results --evaluate checkpoints/msgc_resnet50_noatt.pth

# ResNet-50 with MSGC w/ attention (Top-1 = 76.8%, MAC = 1892 M)
#CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --master_port=12345 --nproc_per_node=1 main_dist.py --model msgc_resnet50 --attention --data imagenet --datadir /to/imagenet/dataset --savedir ./results --evaluate checkpoints/msgc_resnet50_att.pth

# MobileNetV2 with MSGC w/o attention (Top-1 = 72.1%, MAC = 198 M)
#CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --master_port=12345 --nproc_per_node=1 main_dist.py --model msgc_mobilenetv2 --data imagenet --datadir /to/imagenet/dataset --savedir ./results --evaluate checkpoints/msgc_mobilenetv2_noatt.pth

# MobileNetV2 with MSGC w/ attention (Top-1 = 72.6%, MAC = 197 M)
#CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --master_port=12345 --nproc_per_node=1 main_dist.py --model msgc_mobilenetv2 --attention --data imagenet --datadir /to/imagenet/dataset --savedir ./results --evaluate checkpoints/msgc_mobilenetv2_att.pth

# CondenseNet with MSGC w/o attention (Top-1 = 74.8%, MAC = 523 M)
#CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --master_port=12345 --nproc_per_node=1 main_dist.py --model msgc_condensenet --data imagenet --datadir /to/imagenet/dataset --savedir ./results --evaluate checkpoints/msgc_condensenet.pth

# ResNet-18 with MSGC w/o attention, \tau_{end} = 0.9 (Top-1 = 72.3%, MAC = 1631 M)
#CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --master_port=12345 --nproc_per_node=1 main_dist.py --model msgc_resnet18 --data imagenet --datadir /to/imagenet/dataset --savedir ./results --evaluate checkpoints/msgc_resnet18_noatt_tau0.9.pth


## Training

# ResNet-18 with MSGC w/o attention
#CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --master_port=12345 --nproc_per_node=2 main_dist.py --model msgc_resnet18 -j 16 --data imagenet --datadir /to/imagenet/dataset --savedir ./results --resume --target 0.49

# ResNet-18 with MSGC w/ attention
#CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --master_port=12345 --nproc_per_node=2 main_dist.py --model msgc_resnet18 --attention -j 16 --data imagenet --datadir /to/imagenet/dataset --savedir ./results --resume --target 0.488

# ResNet-50 with MSGC w/o attention
#CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --master_port=12345 --nproc_per_node=2 main_dist.py --model msgc_resnet50 -j 16 --data imagenet --datadir /to/imagenet/dataset --savedir ./results --resume --target 0.457

# ResNet-50 with MSGC w/ attention
#CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --master_port=12345 --nproc_per_node=2 main_dist.py --model msgc_resnet50 --attention -j 16 --data imagenet --datadir /to/imagenet/dataset --savedir ./results --resume --target 0.457

# MobileNetV2 with MSGC w/o attention
#CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --master_port=12345 --nproc_per_node=2 main_dist.py --model msgc_mobilenetv2 -j 16 --data imagenet --datadir /to/imagenet/dataset --savedir ./results --epochs 150 --resume --target 0.65

# MobileNetV2 with MSGC w/ attention
#CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --master_port=12345 --nproc_per_node=2 main_dist.py --model msgc_mobilenetv2 --attention -j 16 --data imagenet --datadir /to/imagenet/dataset --savedir ./results --epochs 150 --resume --target 0.628

# CondenseNet with MSGC w/o attention
#CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --master_port=12345 --nproc_per_node=2 main_dist.py --model msgc_condensenet -j 16 --data imagenet --datadir /to/imagenet/dataset --savedir ./results --resume --target 0.258

# ResNet-18 with MSGC w/o attention, \tau_{end} = 0.9
#CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --master_port=12345 --nproc_per_node=2 main_dist.py --model msgc_resnet18 -j 16 --data imagenet --datadir /to/imagenet/dataset --savedir ./results --resume --target 0.9
