'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

import os
import argparse
import numpy as np

from MyNet import my_FCNet_0 as my_ConvNet
from utils import progress_bar, matplotlib_imshow, plot_classes_preds

# Train for 1 epoch
def train(epoch):

    print('\nEpoch: %d' % epoch)
    net.train()

    # Variable initialization
    total_images = len(trainloader.dataset)
    trained_images = 0
    running_loss = 0
    correct = 0
    total = 0
    prev_batch = -1
    brief_step = 100

    # Iterate through the entire dataset
    for batch, (inputs, targets) in enumerate(trainloader):

        inputs, targets = inputs.to(device), targets.to(device)

        # Forward pass, Backward pass, Weight update
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        # Training statistics to be printed
        running_loss += loss.item()     # Add all computed losses together
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        trained_images += len(inputs)

        # To actually print the training statistics
        if batch % brief_step == brief_step-1 or batch == len(trainloader)-1:

            avg_running_loss = running_loss/(batch-prev_batch) if batch == len(trainloader)-1 else running_loss/brief_step
            print('Average loss over {} batches: {}; Images trained: {}/{};'.format(batch-prev_batch,
                                                                                    avg_running_loss,
                                                                                    trained_images,
                                                                                    total_images))
            writer.add_scalar('training loss',
                              avg_running_loss,
                              # epoch * len(trainloader) + batch
                              epoch * total_images + trained_images)
            running_loss = 0
            prev_batch = batch
    return correct/total

# Test for 1 epoch
def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if batch_idx == 0:
                writer.add_figure('predictions vs. actuals on test set',
                                  plot_classes_preds(net, inputs, targets, classes, len(inputs)),
                                  global_step=epoch * len(trainloader) + batch_idx)

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'% (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {'net':   net.state_dict(),
                 'acc':   acc,
                 'epoch': epoch }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc

    return correct/total

if __name__ == '__main__':

    writer = SummaryWriter('runs/cifar10_result_1')

    # Get arguments from command line
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--training_bs', default=32, type=int, help='training batch size')
    parser.add_argument('--testing_bs', default=32, type=int, help='testing batch size')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("using ", device)
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # Prepare Cifar10 Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize((0.1307,), (0.3081,)) ])
    transform_test = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.1307,), (0.3081,)) ])

    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.training_bs, shuffle=True, num_workers=0)

    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.testing_bs, shuffle=True, num_workers=0)

    classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

    # Visualize a batch of training data on tensorboard
    dataiter = iter(trainloader)
    images, labels = dataiter.next()
    img_grid = torchvision.utils.make_grid(images, padding=0, pad_value=0)
    img_grid = matplotlib_imshow(img_grid, one_channel=True)
    img_grid = np.expand_dims(img_grid, 0)
    writer.add_image('a batch of minist images', img_grid)

    # Model
    print('==> Building model..')

    net = my_ConvNet()
    writer.add_graph(net, images)
    writer.close()
    net = net.to(device)

    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/ckpt.pth')
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    criterion = nn.CrossEntropyLoss()   # Loss function
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)    # Optimizer
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)    # Learning rate decay scheduler

    # Train and Test
    for epoch in range(start_epoch, start_epoch+200):
        train_acc = train(epoch)
        test_acc = test(epoch)
        scheduler.step()
        writer.add_scalars('training & testing accuracy',
                           {'training': train_acc, 'testing': test_acc},
                           epoch)
