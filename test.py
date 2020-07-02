from __future__ import print_function

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from torch.autograd import Variable
from utils import AIGS_Dataset

parser = argparse.ArgumentParser(description='PyTorch Testing')
parser.add_argument('--checkpoint', required=True)
parser.add_argument('--dataset', required=True)
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
best_acc = 0
start_epoch = 0

print('==> Preparing data...')
means = (0.4914, 0.4822, 0.4465)
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(means, (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(means, (0.2023, 0.1994, 0.2010)),
])

if args.dataset == 'CIFAR':
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
elif args.dataset == 'AIGS':
    _, testset = AIGS_Dataset(transform_train, transform_test)

testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

print('==> Resuming from checkpoint...')
assert os.path.isdir(args.checkpoint), 'Error: no checkpoint directory found!'
checkpoint = torch.load(os.path.join(args.checkpoint, 'ckpt.t7'))
net = checkpoint['net']
best_acc = checkpoint['acc']
start_epoch = checkpoint['epoch']

if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        print(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    acc = 100. * correct / total
    print(acc)

test(0)
