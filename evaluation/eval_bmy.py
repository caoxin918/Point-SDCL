from __future__ import print_function
import os
import random
import argparse
from pathlib import Path
import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.transforms as transforms
from util import cal_loss, IOStream
from torch.utils.data import DataLoader
from data.bmy import BMY
from model.dgcnn import DGCNN
from model.pointnet import PointNet
import sklearn.metrics as metrics


def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/' + args.exp_name):
        os.makedirs('checkpoints/' + args.exp_name)
    if not os.path.exists('checkpoints/' + args.exp_name + '/' + 'models'):
        os.makedirs('checkpoints/' + args.exp_name + '/' + 'models')


def train(args, io):

    device = torch.device("cuda" if args.cuda else "cpu")

    # Try to load models
    if args.model == 'dgcnn':
        point_model = DGCNN().to(device)
    else:
        point_model = PointNet().to(device)

    # 加载训练的模型参数
    # state_dict = torch.load(args.pretrained, map_location='cpu')
    # missing_keys, unexpected_keys = point_model.load_state_dict(state_dict, strict=False)  # 缺失的关键字,多余的关键字
    # print(missing_keys)
    # print(unexpected_keys)

    # point_model.load_state_dict(torch.load(args.model_path))
    # print("Model Loaded !!")

    # Testing
    criterion = cal_loss

    train_val_loader = DataLoader(BMY(1024), batch_size=8, shuffle=True, drop_last=True)
    test_loader = DataLoader(BMY(1024,partition='test'), batch_size=8, shuffle=True, drop_last=True)
    opt = optim.SGD(point_model.parameters(),lr=args.lr*100,weight_decay=1e-4)
    sceduler = CosineAnnealingLR(opt, T_max=args.epochs)
    best_test_acc = 0
    for epoch in range(args.epochs):

        train_loss = 0
        count = 0.0
        point_model.train()
        train_pred = []
        train_true = []
        for i, (data, label) in enumerate(train_val_loader):
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            opt.zero_grad()
            logits = point_model(data)
            loss = criterion(logits, label)
            loss.backward()
            opt.step()
            preds = logits.max(dim=1)[1]
            count += batch_size
            train_loss += loss.item() * batch_size
            train_true.append(label.cpu().numpy())
            train_pred.append(preds.detach().cpu().numpy())
        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        outstr = 'Train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f' % (epoch,
                                                                                 train_loss * 1.0 / count,
                                                                                 metrics.accuracy_score(
                                                                                     train_true, train_pred),
                                                                                 metrics.balanced_accuracy_score(
                                                                                     train_true, train_pred))
        io.cprint(outstr)
        sceduler.step()
        ####################
        # Test
        ####################
        test_loss = 0.0
        count = 0.0
        point_model.eval()
        test_pred = []
        test_true = []
        for data, label in test_loader:
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            logits = point_model(data)
            loss = criterion(logits, label)
            preds = logits.max(dim=1)[1]
            count += batch_size
            test_loss += loss.item() * batch_size
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())
        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        test_acc = metrics.accuracy_score(test_true, test_pred)
        avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
        outstr = 'Test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f' % (epoch,
                                                                              test_loss * 1.0 / count,
                                                                              test_acc,
                                                                              avg_per_class_acc)
        io.cprint(outstr)
        if test_acc >= best_test_acc:
            best_test_acc = test_acc
            torch.save(point_model.state_dict(), 'checkpoints/%s/models/model_epoch200_SGD.t7' % args.exp_name)
            print("当前最高的准确率是",best_test_acc)







if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='dgcnn', metavar='N',
                        choices=['dgcnn', 'dgcnn_seg'],
                        help='Model to use, [pointnet, dgcnn]')
    parser.add_argument('--batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=210, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--start_epoch', type=int, default=0, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', action="store_true", help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool, default=False,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=2048,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=40, metavar='N',
                        help='Num of nearest neighbors to use')

    parser.add_argument('--pretrained', default='../Ablation_analysis/one/dgcnn/two_order/backbone/model_015.pth', type=Path, metavar='FILE',
                        help='path to pretrained model')
    args = parser.parse_args()

    _init_()

    io = IOStream('checkpoints/' + args.exp_name + '/run.log')
    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')

    if not args.eval:
        train(args, io)
    else:
        test(args, io)


