## 라이브러리 추가하기
import os
import argparse

import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from torchvision import transforms, datasets
from util import load, save
from model import Network

from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd


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
    img_dir = args.img_dir
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


    # check conv
    # print(f"net conv1 weight : {net.conv1.weight}, shape : {net.conv1.weight.shape}")
    # weight = net.conv1.weight.detach().cpu().numpy()
    # print(f"weight : {weight.shape}")
    # plt.imshow(weight[0, 0, :, :], 'jet')
    # plt.colorbar()
    # plt.show()

    y_pred = []
    y_true = []

    # eval
    with torch.no_grad():
        net.eval()

        loss_arr = []
        acc_arr = []

        for batch, (input, label) in enumerate(loader, 1):
            input = input.to(device)
            label = label.to(device)

            output = net(input)
            pred = fn_pred(output)
            pred_num = pred.max(dim=1)

            y_pred = np.concatenate((y_pred, pred_num.indices.detach().cpu().numpy()), axis=0)
            y_true = np.concatenate((y_true, label.detach().cpu().numpy()), axis=0)

            loss = fn_loss(output, label)
            acc = fn_acc(pred, label)

            loss_arr += [loss.item()]
            acc_arr += [acc.item()]

            print('TEST: BATCH %04d/%04d | LOSS %.4f | ACC %.4f' %
                  (batch, num_batch, np.mean(loss_arr), np.mean(acc_arr)))

            # Plot Image
            fig = plt.figure(figsize=(12, 6))
            cols, rows = 4, 2
            for i in range(1, cols * rows + 1):
                sample_idx = torch.randint(high=min(len(label), batch_size), size=(1,))
                sample_idx = sample_idx.item()
                _pred = pred_num.indices[sample_idx]
                img, _label = input[sample_idx], label[sample_idx]

                fig.add_subplot(rows, cols, i)
                plt.title(f"label : {_label}, pred : {_pred}")
                plt.axis('off')
                plt.imshow(img.detach().cpu().numpy().squeeze(), cmap='gray')

            if not os.path.exists(img_dir):
                os.makedirs(img_dir)
            plt.savefig(f"{img_dir}/output{batch}.jpg")

    # Draw confusion matrix
    classes= ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix) * 10, index=[i for i in classes])
    plt.figure(figsize=(12, 7))
    plt.yticks(rotation=90)
    sn.heatmap(df_cm, annot=True)
    plt.savefig(f"{img_dir}/confusion_matrix.png")


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
    parser.add_argument("--img_dir", default='./output', type=str, dest="img_dir")
    args = parser.parse_args()

    if args.mode == "train":
        train(args)
    elif args.mode == "eval":
        eval(args)