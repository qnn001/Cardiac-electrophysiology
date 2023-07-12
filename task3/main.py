import os
import pdb
import time 
import pickle
import random
import shutil
import argparse
import numpy as np  
from copy import deepcopy
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset

import torch
import torch.optim
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import torchvision.models as models
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data.sampler import SubsetRandomSampler

from data_dssi import read_data, Custom_dataset
from models import SqueezeNet, Transformer1d



parser = argparse.ArgumentParser(description='PyTorch Evaluation Tickets')

##################################### general setting #################################################
parser.add_argument('--seed', default=42, type=int, help='random seed')
parser.add_argument('--save_dir', help='The directory used to save the trained models', default='output', type=str)
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')

##################################### training setting #################################################
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay')
parser.add_argument('--epochs', default=10, type=int, help='number of total epochs to run')
parser.add_argument('--warmup', default=0, type=int, help='warm up epochs')
parser.add_argument('--print_freq', default=100, type=int, help='print frequency')
parser.add_argument('--decreasing_lr', default='91,136', help='decreasing strategy')

def main():
    global args
    args = parser.parse_args()
    print(args)

    torch.cuda.set_device(int(args.gpu))
    os.makedirs(args.save_dir, exist_ok=True)
    if args.seed:
        setup_seed(args.seed)

    # Construct Dataset
    raw_data = read_data()
    whole_data = Custom_dataset(raw_data)
    num_examples = len(whole_data)
    permutation_index = np.random.permutation(num_examples)
    train_examples = int(num_examples * 0.90)
    eval_examples = int(num_examples * 0.05)
    test_examples = num_examples - train_examples - eval_examples

    train_set = Subset(whole_data, list(permutation_index[:train_examples]))
    val_set = Subset(whole_data, list(permutation_index[train_examples:train_examples + eval_examples]))
    test_set = Subset(whole_data, list(permutation_index[-test_examples:]))

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=False, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    print('### Loading Dataset ###')
    print('### Training Examples {}, Validation Examples {}, Test Examples {} ###'.format(train_examples, eval_examples, test_examples))

    # Define Model
    model = SqueezeNet('1_1', 75, 0)
    # model = Transformer1d(
    #     n_classes = 75,
    #     nlength = 500,
    #     d_model = 192,
    #     n_layers = 6,
    #     nhead = 3,
    #     dim_feedforward = 768,
    #     dropout = 0.1,
    #     activation = nn.ReLU
    # )

    parameter_cnt = 0
    for p in model.parameters():
        parameter_cnt += p.nelement()
    print('Model Parameters = {:.4f} MB'.format(parameter_cnt / 1024 / 1024))
    model.cuda()

    criterion = nn.MSELoss()
    # decreasing_lr = list(map(int, args.decreasing_lr.split(',')))
    # optimizer = torch.optim.SGD(model.parameters(), args.lr,
    #                             momentum=args.momentum,
    #                             weight_decay=args.weight_decay)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=decreasing_lr, gamma=0.1)

    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)


    # evaluate on validation set
    trainloss = validate(train_loader, model, criterion)
    # evaluate on validation set
    vloss = validate(val_loader, model, criterion)
    # evaluate on test set
    tloss = validate(test_loader, model, criterion)

    all_result = {
        'train': [trainloss],
        'val': [vloss],
        'test': [tloss]
    }
    start_epoch = 0
    for epoch in range(start_epoch, args.epochs):

        print(optimizer.state_dict()['param_groups'][0]['lr'])
        loss = train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        vloss = validate(val_loader, model, criterion)
        # evaluate on test set
        tloss = validate(test_loader, model, criterion)

        scheduler.step()

        all_result['train'].append(loss)
        all_result['val'].append(vloss)
        all_result['test'].append(tloss)

        plt.plot(all_result['train'], label='train_loss')
        plt.plot(all_result['val'], label='val_loss')
        plt.plot(all_result['test'], label='test_loss')
        plt.legend()
        plt.savefig(os.path.join(args.save_dir, 'net_train.png'))
        plt.close()

    best_val_epoch = np.argmin(all_result['val'])
    print('best epoch: {}, Train MSE: {}, Val MSE: {}, Test MSE: {}'.format(epoch, all_result['train'][best_val_epoch], all_result['val'][best_val_epoch], all_result['test'][best_val_epoch]))



def train(train_loader, model, criterion, optimizer, epoch):
    
    losses = AverageMeter()
    # switch to train mode
    model.train()

    start = time.time()
    for i, (input, target) in enumerate(train_loader):

        if epoch < args.warmup:
            warmup_lr(epoch, i+1, optimizer, one_epoch_step=len(train_loader))

        input = input.cuda()
        target = target.cuda()

        output = model(input)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss = loss.float()
        losses.update(loss.item(), target.size(0))

        if i % args.print_freq == 0:
            end = time.time()
            print('Epoch: [{0}][{1}/{2}]\t'
                'Loss {loss.val:.6f} ({loss.avg:.6f})\t'
                'Time {3:.2f}'.format(
                    epoch, i, len(train_loader), end-start, loss=losses))
            start = time.time()
    print('train_loss {loss.avg:.6f}'.format(loss=losses))
    return losses.avg


def validate(val_loader, model, criterion):
    """
    Run evaluation
    """
    losses = AverageMeter()

    # switch to evaluate mode
    model.eval()

    for i, (input, target) in enumerate(val_loader):
        
        input = input.cuda()
        target = target.cuda()

        # compute output
        with torch.no_grad():
            output = model(input)
            loss = criterion(output, target)

        loss = loss.float()
        losses.update(loss.item(), target.size(0))

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                'Loss {loss.val:.6f} ({loss.avg:.6f})\t'.format(
                    i, len(val_loader), loss=losses))
    print('valid_loss {loss.avg:.6f}'
        .format(loss=losses))
    return losses.avg

def warmup_lr(epoch, step, optimizer, one_epoch_step):

    overall_steps = args.warmup*one_epoch_step
    current_steps = epoch*one_epoch_step + step 

    lr = args.lr * current_steps/overall_steps
    lr = min(lr, args.lr)

    for p in optimizer.param_groups:
        p['lr']=lr

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def setup_seed(seed): 
    torch.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed) 
    np.random.seed(seed) 
    random.seed(seed) 
    torch.backends.cudnn.deterministic = True 

if __name__ == '__main__':
    main()
