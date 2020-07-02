from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import MultiStepLR

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models.resnet import *
from torch.autograd import Variable
from utils import AIGS_Dataset, mixup_data, mixup_criterion

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--resume', action='store_true', help='resume from checkpoint')
parser.add_argument('--mixup', action='store_true', help='apply the Mixup method')
parser.add_argument('--transfer_learning', action='store_true', help='apply the Transfer Learning')
parser.add_argument('--n_epochs', default=250, type=int)
parser.add_argument('--checkpoint', required=True)
parser.add_argument('--dataset', required=True)
parser.add_argument('--model', required=True)
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
best_acc = 0
start_epoch = 0
n_epochs = args.n_epochs

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
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
elif args.dataset == 'AIGS':
    trainset, testset = AIGS_Dataset(transform_train, transform_test)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

if args.resume:
    print('==> Resuming from checkpoint...')
    assert os.path.isdir(args.checkpoint), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(os.path.join(args.checkpoint, 'ckpt.t7'))
    net = checkpoint['net']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
else:
    print('==> Building model...')
    if args.model == 'ResNet18':
        if args.transfer_learning:
            net = torchvision.models.resnet18(pretrained=True)
        else:
            net = ResNet18()
    elif args.model == 'ResNet34':
        if args.transfer_learning:
            net = torchvision.models.resnet34(pretrained=True)
        else:
            net = ResNet34()
    elif args.model == 'ResNet50':
        if args.transfer_learning:
            net = torchvision.models.resnet50(pretrained=True)
        else:
            net = ResNet50()
    if args.transfer_learning:
        num_ftrs = net.fc.in_features
        net.fc = nn.Linear(num_ftrs, 10)

if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        if args.mixup:
            inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, 1.0, use_cuda)
            inputs, targets_a, targets_b = map(Variable, (inputs, targets_a, targets_b))
            outputs = net(inputs)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            _, predicted = torch.max(outputs.data, 1)
            correct += lam * predicted.eq(targets_a.data).cpu().sum().float()
            correct += (1 - lam) * predicted.eq(targets_b.data).cpu().sum().float()
        else:
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            _, predicted = torch.max(outputs.data, 1)
            correct += predicted.eq(targets.data).cpu().sum()

        total += targets.size(0)
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        
        if args.transfer_learning:
            if batch_idx >= len(trainloader) - 2:
                break

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
    if acc > best_acc:
        print('Saving...')
        state = {
            'net': net,
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir(args.checkpoint):
            os.mkdir(args.checkpoint)
        torch.save(state, os.path.join(args.checkpoint, 'ckpt.t7'))
        best_acc = acc

milestones = [40, 90, 140, 180, 220]
scheduler = MultiStepLR(optimizer, milestones, gamma=0.1)
for epoch in range(start_epoch):
    scheduler.step()

for epoch in range(start_epoch, n_epochs):
    scheduler.step()
    train(epoch)
    test(epoch)
