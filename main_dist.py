from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import argparse
import os
import time
import models
from utils import *

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import torch.nn.parallel
import torch.distributed as dist 

parser = argparse.ArgumentParser(description='PyTorch Middle Spectrum Grouped Convolution')

### dirs
parser.add_argument('--data', type=str, default='imagenet', 
        help='name of dataset', choices=['imagenet'])
parser.add_argument('--datadir', type=str, default='../data', 
        help='dir to the dataset or the validation set')
parser.add_argument('--savedir', type=str, default='results/savedir', 
        help='path to save result and checkpoint')

### model
parser.add_argument('--model', type=str, default='dgc_densenet86', 
        help='model to train the dataset')
parser.add_argument('--heads', type=int, default=4, 
        help='number of heads')
parser.add_argument('--width-mul', type=float, default=1.0, 
        help='width mutiplier for mobilenetv2')
parser.add_argument('--attention', action='store_true', 
        help='use attention in model')
parser.add_argument('--scratch', action='store_true', 
        help='load pretrained model from pytorch repository')
parser.add_argument('--checkinfo', action='store_true', 
        help='only check the information of model')

### training
parser.add_argument('--epochs', type=int, default=120, 
        help='number of total epochs to run')
parser.add_argument('--warm-epoch', default=0, type=int, 
        help='epoch number to warm up')
parser.add_argument('-b', '--batch-size', type=int, default=128, 
        help='mini-batch size')
parser.add_argument('--btest', type=int, default=100, 
        help='mini-batch size for testing')
parser.add_argument('--opt', type=str, default='sgd', 
        help='optimizer [adam, sgd]')
parser.add_argument('--momentum', type=float, default=0.9, 
        help='momentum (default: 0.9)')
parser.add_argument('--lr', type=float, default=0.075, 
        help='initial learning rate for mask weights')
parser.add_argument('--lr-mul', type=float, default=0.2, 
        help='initial learning rate scale for pretrained weights')
parser.add_argument('--lr-type', type=str, default='cosine', 
        help='learning rate strategy [cosine, multistep]')
parser.add_argument('--lr-steps', type=str, default=None, 
        help='steps for multistep learning rate')
parser.add_argument('--wd', type=float, default=1e-4, 
        help='weight decay for all weights')
parser.add_argument('--label-smooth', type=float, default=0.1, 
        help='label smoothing')
parser.add_argument('--lmd', type=float, default=30, 
        help='lambda for calculating disperity loss')
parser.add_argument('--target', type=float, default=0.5, 
        help='target flops rate for DGC convolutions')
parser.add_argument('--pstart', type=float, default=0, 
        help='start pruning progress')
parser.add_argument('--pstop', type=float, default=0.5, 
        help='stop pruning progress')
parser.add_argument('--seed', type=int, default=None, 
        help='random seed (default: None)')
parser.add_argument('--print-freq', type=int, default=10, 
        help='print frequency (default: 10)')

# gpu and cpu
parser.add_argument('--gpu', type=str, default='', 
        help='gpus available')
parser.add_argument('--nocudnnbm', action='store_true', 
        help='set cudnn benchmark to False')
parser.add_argument('-j', '--workers', type=int, default=4, 
        help='number of data loading workers')

### checkpoint
parser.add_argument('--save-freq', type=int, default=5, 
        help='save frequency (default: 10)')
parser.add_argument('--resume', action='store_true', 
        help='use latest checkpoint if have any')
parser.add_argument('--resume-from', type=str, default=None,
        help='give a checkpoint path for resuming')
parser.add_argument('--evaluate', type=str, default=None, 
        help="full path to checkpoint to be evaluated or 'best'")

parser.add_argument('--local_rank', type=int)

args = parser.parse_args()

torch.distributed.init_process_group(backend='nccl')
torch.cuda.set_device(args.local_rank)
device = torch.device('cuda', args.local_rank)

best_prec1 = 0

def main(running_file):

    global args, best_prec1

    ### Refine args
    if args.seed is None:
        args.seed = int(time.time())
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.lr_steps is not None: 
        args.lr_steps = list(map(int, args.lr_steps.split('-'))) 
        args.lr_gammas = [0.1 for _ in args.lr_steps]

    if args.btest is None:
        args.btest = args.batch_size

    if args.data == 'cifar10':
        args.num_classes = 10
        R = 32
    elif args.data == 'cifar100':
        args.num_classes = 100
        R = 32
    elif 'imagenet' in args.data:
        args.num_classes = 1000
        R = 224
    else:
        raise ValueError('unrecognized data')

    ### Create model
    model = getattr(models, args.model)(args)

    flops_ori, flops_possible, flops_main, flops_dgc, flops_mask = model.get_flops()
    flops_target = args.target * flops_ori
    args.flops_ori, args.flops_main, args.flops_dgc, args.flops_mask = \
            flops_ori, flops_main, flops_dgc, flops_mask
    print(args)
    flopsinfo = 'Flops of {}: original {} M, target {} M, possible {} M, dgc {} M, mask {} M\n'.format(
        args.model, flops_ori / 1e6, flops_target / 1e6, flops_possible / 1e6, flops_dgc / 1e6, flops_mask / 1e6)
    print(flopsinfo)

    if args.checkinfo:
        running_file.write(flopsinfo)
        running_file.flush()
        return

    ### Define optimizer
    param_dict = dict(model.named_parameters())

    p_conv = []
    pname_conv = []

    p_mask = []
    pname_mask = []

    p_bn = []
    pname_bn = []

    BN_name_pool = []
    for m_name, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            BN_name_pool.append(m_name + '.weight')
            BN_name_pool.append(m_name + '.bias')

    for key, value in param_dict.items():
        if 'mask' in key:
            pname_mask.append(key)
            p_mask.append(value)
        elif key in BN_name_pool:
            pname_bn.append(key)
            p_bn.append(value)
        else:
            pname_conv.append(key)
            p_conv.append(value)
    params = [{'params': p_mask, 'lr': args.lr, 'weight_decay': 0.}, 
              {'params': p_bn, 'lr': args.lr * args.lr_mul, 'weight_decay': 0.}, 
              {'params': p_conv, 'lr': args.lr * args.lr_mul, 'weight_decay': args.wd}]
    args.lr_list = [g['lr'] for g in params]
    optimizer = torch.optim.SGD(params, momentum=args.momentum, nesterov=True)

    ### Transfer to cuda devices
    model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, 
            device_ids=[args.local_rank], output_device=args.local_rank)
    print('cuda is used, with %d gpu devices' % torch.cuda.device_count())

    ### Define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()
    criterion_smooth = CrossEntropyLabelSmooth(args.num_classes, args.label_smooth)

    cudnn.benchmark = not args.nocudnnbm

    ### Data loading 
    traindir = os.path.join(args.datadir, 'train')
    valdir = os.path.join(args.datadir, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_set = datasets.ImageFolder(traindir, transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]))

    val_set = datasets.ImageFolder(valdir, transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]))

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=args.btest, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    ### Optionally resume from a checkpoint
    args.start_epoch = 0
    if args.resume or (args.resume_from is not None) or (args.evaluate is not None):
        checkpoint = load_checkpoint(args, running_file)
        if checkpoint is not None:
            model.load_state_dict(checkpoint['state_dict'])
            ## Evaluate directly if required
            if args.evaluate is not None:
                validate(val_loader, model, criterion, args)
                print('##########Time########## %s' % (time.strftime('%Y-%m-%d %H:%M:%S')))
                return
            args.start_epoch = checkpoint['epoch'] + 1
            best_prec1 = checkpoint['best_prec1']
            optimizer.load_state_dict(checkpoint['optimizer'])


    ### Train
    saveID = None
    print('current best: {}'.format(best_prec1))
    with open(args.log_file, 'a') as f:
        f.write('Flops of {}: original {} M, target {} M, possible {} M, main {} M, dgc {} M, mask {} M\n'.format(
            args.model, flops_ori/1e6, flops_target/1e6, flops_possible/1e6, flops_main/1e6, flops_dgc/1e6, flops_mask/1e6))

    for epoch in range(args.start_epoch, args.epochs):

        if epoch >= args.epochs - 5:
            args.save_freq = 1

        train_sampler.set_epoch(epoch)

        # adjust learning rate and progress
        lr_str = adjust_learning_rate(optimizer, epoch, args, method=args.lr_type)

        # train
        tr_prec1, tr_prec5, loss, tr_flops, tr_dgc, tr_bonus = \
            train(train_loader, model, criterion_smooth, optimizer, epoch, 
                    running_file, lr_str, args)

        val_prec1, val_prec5, val_flops, val_dgc, val_bonus = \
            validate(val_loader, model, criterion, args)

        is_best = val_prec1 >= best_prec1
        best_prec1 = max(val_prec1, best_prec1)

        log = ("Epoch %03d/%03d: (%.4f %.4f) | %.4f M (%.4f -%.4f)" + \
               " || train (%.4f %.4f) | %.4f M (%.4f -%.4f)| loss %.4f" + \
               " || lr %s | Time %s\n") \
            % (epoch, args.epochs, val_prec1, val_prec5, val_flops, val_dgc, val_bonus, \
            tr_prec1, tr_prec5, tr_flops, tr_dgc, tr_bonus, loss, \
            lr_str, time.strftime('%Y-%m-%d %H:%M:%S'))
        with open(args.log_file, 'a') as f:
            f.write(log)

        if args.local_rank == 0:
            print('checkpoint saving in local rank 0')
            running_file.write('checkpoint saving in local rank 0\n')
            running_file.flush()
            saveID = save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': optimizer.state_dict(),
                }, epoch, args.savedir, is_best, 
                saveID, keep_freq=args.save_freq)

    return


def train(train_loader, model, criterion, optimizer, epoch, 
        running_file, running_lr, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()

    losses = AverageMeter()
    losses_sparse = AverageMeter()
    dgces = AverageMeter()
    bonuses = AverageMeter()
    flopses = AverageMeter()

    top1 = AverageMeter('sum')
    top5 = AverageMeter('sum')

    ## Switch to train mode
    model.train()

    running_file.write('\n%s\n' % str(args))
    running_file.flush()

    wD = len(str(len(train_loader)))
    wE = len(str(args.epochs))

    end = time.time()
    for i, (input, label) in enumerate(train_loader):

        ## Calculate progress
        progress = float(epoch * len(train_loader) + i) / (args.epochs * len(train_loader))
        start, stop = args.pstart, args.pstop
        target = (progress - start) / (stop - start) * (args.target - 1) + 1
        target = max(target, args.target) if progress > start else 1.0

        ## Measure data loading time
        data_time.update(time.time() - end)

        input = input.cuda(non_blocking=True)
        label = label.cuda(non_blocking=True)

        ## Compute output
        output, [dgc, bonus] = model(input)

        loss = criterion(output, label)
        flops = args.flops_main + dgc + args.flops_mask - bonus
        if flops.item() / args.flops_ori >= target:
            #loss_sparse = args.lmd * (flops / args.flops_ori - target) ** 2
            loss_sparse = args.lmd * (flops / args.flops_ori - target)
            losses_sparse.update(loss_sparse.item(), input.size(0))
        else:
            loss_sparse = 0
            losses_sparse.update(0, input.size(0))

        ## Measure accuracy and record losses

        prec1, prec5 = accuracy(output.data, label, topk=(1, 5))

        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))
        flopses.update(flops.item()/1e6, input.size(0))
        dgces.update(dgc.item()/1e6, input.size(0))
        bonuses.update(bonus.item()/1e6, input.size(0))

        losses.update(loss.item(), input.size(0))

        ## Compute gradient and do SGD step
        loss = loss + loss_sparse
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ## Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        ## Record
        if i % args.print_freq == 0:
            runinfo = str(('GPU %d Epoch: [{0:0%dd}/{1:0%dd}][{2:0%dd}/{3:0%dd}] | ' \
                      % (args.local_rank, wE, wE, wD, wD) + \
                      'Time {batch_time.val:.3f} | ' + \
                      'Data {data_time.val:.3f} | ' + \
                      'Loss ({loss.val:.4f} {loss_sparse.val:.4f}) | ' + \
                      'Flops ({flops.val:.4f} M {dgc.val:.4f} M {bonus.val:.4f} M) | ' + \
                      'Prec@1 {top1.val100:.3f} | ' + \
                      'Prec@5 {top5.val100:.3f} | ' + \
                      'lr {lr}').format(
                          epoch, args.epochs, i, len(train_loader), 
                          batch_time=batch_time, data_time=data_time, 
                          loss=losses, loss_sparse=losses_sparse, 
                          flops=flopses, dgc=dgces, bonus=bonuses,
                          top1=top1, top5=top5, lr=running_lr))
            print(runinfo)
            if i % (args.print_freq * 20) == 0 and args.local_rank == 0:
                running_file.write('%s\n' % runinfo)
                running_file.flush()

    return top1.avg100, top5.avg100, losses.avg, flopses.avg, dgces.avg, bonuses.avg


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter()

    dgces = AverageMeter()
    bonuses = AverageMeter()
    flopses = AverageMeter()

    top1 = AverageMeter('sum')
    top5 = AverageMeter('sum')

    ## Switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, label) in enumerate(val_loader):
        with torch.no_grad():
            label = label.cuda()
            input = input.cuda()

            ## Compute output
            output, [dgc, bonus] = model(input)

        ## Measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, label, topk=(1, 5))
        flops = args.flops_main + dgc + args.flops_mask - bonus

        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))
        flopses.update(flops.item()/1e6, input.size(0))
        dgces.update(dgc.item()/1e6, input.size(0))
        bonuses.update(bonus.item()/1e6, input.size(0))

        ## Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        ## Record
        if i % args.print_freq == 0:
            print(('Test: [{0}/{1}]\t' + \
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' + \
                  'Flops ({flops.val:.4f} M {dgc.val:.4f} M {bonus.val:.4f} M) | ' + \
                  'Prec@1 {top1.val100:.3f} ({top1.avg100:.3f})\t' + \
                  'Prec@5 {top5.val100:.3f} ({top5.avg100:.3f})').format(
                      i, len(val_loader), batch_time=batch_time, 
                      flops=flopses, dgc=dgces, bonus=bonuses,
                      top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg100:.3f} | Prec@5 {top5.avg100:.3f} | '
            'Flops {flops.avg:.4f} M'.format(
            top1=top1, top5=top5, flops=flopses))

    return top1.avg100, top5.avg100, flopses.avg, dgces.avg, bonuses.avg


if __name__ == '__main__':

    os.makedirs(args.savedir, exist_ok=True)
    args.log_file = os.path.join(args.savedir, '%s_log.txt' % args.model)
    running_file = os.path.join(args.savedir, '%s_running-%s.txt' % (args.model, time.strftime('%Y-%m-%d-%H-%M-%S')))

    with open(running_file, 'w') as f:
        main(f)
