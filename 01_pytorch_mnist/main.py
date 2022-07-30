## 라이브러리 추가하기
import os
import argparse

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from torchvision import transforms, datasets
from util import load, save
from model import Network


## Training
def train(args):
    lr = args.lr
    batch_size = args.batch_size
    num_epoch = args.num_epoch
    ckpt_dir = args.ckpt_dir
    log_dir = args.log_dir
    data_dir = args.data_dir

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ## MNIST 데이터 셋 불러오기
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,))])

    dataset = datasets.MNIST(download=True, root=data_dir, train=True, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    num_data = len(loader.dataset)
    num_batch = np.ceil(num_data / batch_size)

    ## 네트워크와 손실 함수 설정
    net = Network().to(device)
    params = net.parameters()

    fn_loss = nn.CrossEntropyLoss().to(device)
    fn_pred = lambda output: torch.softmax(output, dim=1)
    fn_acc = lambda pred, label: ((pred.max(dim=1)[1] == label).type(torch.float)).mean()

    optim = torch.optim.Adam(params, lr=lr)

    for epoch in range(1, num_epoch + 1):
        net.train()

        loss_arr = []
        acc_arr = []

        for batch, (input, label) in enumerate(loader, 1):
            input = input.to(device)
            label = label.to(device)

            output = net(input)
            pred = fn_pred(output)

            optim.zero_grad()

            loss = fn_loss(output, label)
            acc = fn_acc(pred, label)

            loss.backward()

            optim.step()

            loss_arr += [loss.item()]
            acc_arr += [acc.item()]

            print('TRAIN: EPOCH %04d/%04d | BATCH %04d/%04d | LOSS %.4f | ACC %.4f' %
                  (epoch, num_epoch, batch, num_batch, np.mean(loss_arr), np.mean(acc_arr)))

        save(ckpt_dir=ckpt_dir, net=net, optim=optim, epoch=epoch)

def eval(args):
    lr = args.lr
    batch_size = args.batch_size
    num_epoch = args.num_epoch
    ckpt_dir = args.ckpt_dir
    log_dir = args.log_dir
    data_dir = args.data_dir

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # MNIST 데이터 셋 불러오기
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,))])

    dataset = datasets.MNIST(download=True, root='./', train=False, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    num_data = len(loader.dataset)
    num_batch = np.ceil(num_data / batch_size)

    # 네트워크와 손실 함수 설정
    net = Network().to(device)
    params = net.parameters()

    fn_loss = nn.CrossEntropyLoss().to(device)
    fn_pred = lambda output: torch.softmax(output, dim=1)
    fn_acc = lambda pred, label: ((pred.max(dim=1)[1] == label).type(torch.float)).mean()

    optim = torch.optim.Adam(params, lr=lr)

    net, optim = load(ckpt_dir=ckpt_dir, net=net, optim=optim)

    # Training
    with torch.no_grad():
        net.eval()

        loss_arr = []
        acc_arr = []

        for batch, (input, label) in enumerate(loader, 1):
            input = input.to(device)
            label = label.to(device)

            output = net(input)
            pred = fn_pred(output)

            loss = fn_loss(output, label)
            acc = fn_acc(pred, label)

            loss_arr += [loss.item()]
            acc_arr += [acc.item()]

            print('TEST: BATCH %04d/%04d | LOSS %.4f | ACC %.4f' %
                  (batch, num_batch, np.mean(loss_arr), np.mean(acc_arr)))


if __name__ == "__main__":
    ## Parser 생성하기
    parser = argparse.ArgumentParser(description="MNIST_Classifier",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--mode", default="train", choices=["train", "eval"], type=str, dest="mode")

    # Hyperparameters
    parser.add_argument("--lr", default=1e-3, type=float, dest="lr")
    parser.add_argument("--batch_size", default=128, type=int, dest="batch_size")
    parser.add_argument("--epoch", default=10, type=int, dest="num_epoch")

    parser.add_argument("--ckpt_dir", default='./checkpoint', type=str, dest="ckpt_dir")
    parser.add_argument("--log_dir", default='./log', type=str, dest="log_dir")

    parser.add_argument("--data_dir", default='./', type=str, dest="data_dir")

    args = parser.parse_args()

    if args.mode == "train":
        train(args)
    elif args.mode == "eval":
        eval(args)