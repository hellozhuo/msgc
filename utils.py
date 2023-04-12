from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import os
import shutil
import math
import time

import torch
import torch.nn as nn


######################################
#         basic functions            #
######################################


class CrossEntropyLabelSmooth(nn.Module):
    """
        label smooth
    """
    def __init__(self, num_classes, epsilon):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss


def load_checkpoint(args, running_file):

    model_dir = os.path.join(args.savedir, 'save_models')
    latest_filename = os.path.join(model_dir, 'latest.txt')
    model_filename = ''

    if args.evaluate is not None:
        model_filename = args.evaluate
    elif args.resume_from is not None:
        model_filename = args.resume_from
    elif os.path.exists(latest_filename):
        with open(latest_filename, 'r') as fin:
            model_filename = fin.readlines()[0].strip()
    loadinfo = "=> loading checkpoint from '{}'".format(model_filename)
    print(loadinfo)

    state = None
    if os.path.exists(model_filename):
        state = torch.load(model_filename, map_location='cpu')
        loadinfo2 = "=> loaded checkpoint '{}' successfully".format(model_filename)
    else:
        loadinfo2 = "no checkpoint loaded"
    print(loadinfo2)
    running_file.write('%s\n%s\n' % (loadinfo, loadinfo2))
    running_file.flush()

    return state


def save_checkpoint(state, epoch, root, is_best, saveID, keep_freq=10):

    filename = 'checkpoint_%03d.pth.tar' % epoch
    model_dir = os.path.join(root, 'save_models')
    model_filename = os.path.join(model_dir, filename)
    latest_filename = os.path.join(model_dir, 'latest.txt')

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # write new checkpoint 
    torch.save(state, model_filename)
    with open(latest_filename, 'w') as fout:
        fout.write(model_filename)
    print("=> saved checkpoint '{}'".format(model_filename))

    # update best model 
    if is_best:
        best_filename = os.path.join(model_dir, 'model_best.pth.tar')
        shutil.copyfile(model_filename, best_filename)

    # remove old model
    if saveID is not None and (saveID + 1) % keep_freq != 0:
        filename = 'checkpoint_%03d.pth.tar' % saveID
        model_filename = os.path.join(model_dir, filename)
        if os.path.exists(model_filename):
            os.remove(model_filename)
            print('=> removed checkpoint %s' % model_filename)

    print('##########Time##########', time.strftime('%Y-%m-%d %H:%M:%S'))
    return epoch


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, accum='mean'):
        self.reset()
        self.accum = accum

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if self.accum == 'mean':
            self.sum += val * n
            self.val = val
        elif self.accum == 'sum':
            self.sum += val
            self.val = val / n
        self.count += n
        self.avg = self.sum / self.count
        self.avg100 = self.sum / self.count * 100
        self.val100 = self.val * 100

def adjust_learning_rate(optimizer, epoch, args, method='cosine'):
    if method == 'cosine':
        T_total = float(args.epochs)
        T_cur = float(epoch)
        lr_multi = 0.5 * (1 + math.cos(math.pi * T_cur / T_total))
    elif method == 'multistep':
        lr_multi = 1.0
        for epoch_step, lr_gamma in zip(args.lr_steps, args.lr_gammas):
            if epoch >= epoch_step:
                lr_multi = lr_multi * lr_gamma
    if epoch < args.warm_epoch:
        lr_multi = (epoch + 1) / args.warm_epoch

    _lr = []
    for i, param_group in enumerate(optimizer.param_groups):
        param_group['lr'] = args.lr_list[i] * lr_multi
        _lr.append('{:.6f}'.format(param_group['lr']))
    return '-'.join(_lr)


def accuracy(output, target, topk=(1,)):
    """
        Computes the precision@k for the specified values of k
    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        #res.append(correct_k.mul_(100.0 / batch_size))
        res.append(correct_k)
    return res



######################################
#         debug functions            #
######################################

def change_checkpoint(state):
    """
        an interface to modify the checkpoint
    """
    state_new = dict()
    for k, v in state.items():
        if 'binary_conv' in k:
            state_new[k.replace('binary_conv', 'bconv')] = v
        elif 'bn1' in k:
            state_new[k.replace('bn1', 'bn')] = v
        else:
            state_new[k] = v
    return state_new

def visualize(checkpoint, img_dir):

    from matplotlib import pyplot as plt
    import numpy as np

    state = checkpoint['state_dict']
    epoch = checkpoint['epoch']
    os.makedirs(img_dir, exist_ok=True)
    img_file = os.path.join(img_dir, 'img_epoch_%03d.png' % epoch)
    print('processing %s' % img_file)

    data = []
    for k, v in state.items():
        if 'bconv' in k and 'weights' in k:
            data.append(v.data.view(-1))

    data = torch.cat(data).cpu().numpy()

    bins = list(np.linspace(-1.5, 1.5, 200))
    plt.hist(data, bins)
    plt.savefig(img_file)
    plt.close()

    print('done')


