import argparse
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from model import *
from utils import *
from layer import *
from datasets import *
import numpy as np
import random

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=2024, help='random seed')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.001, help='weight decay')
parser.add_argument('--nhid', type=int, default=128, help='hidden size')
parser.add_argument('--dropout_ratio', type=float, default=0.1, help='dropout ratio')
parser.add_argument('--device', type=str, default='cuda:2', help='specify cuda devices')
parser.add_argument('--target', type=str, default='DBLPv7', help='target domain data')
parser.add_argument('--epochs', type=int, default=2000, help='maximum number of epochs')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--num_layers', type=int, default=2, help='number of gnn layers')
parser.add_argument('--gnn', type=str, default='gcn', help='different types of gnns')
parser.add_argument('--use_bn', type=bool, default=False, help='do not use batchnorm')
parser.add_argument('--K', type=int, default=40, help='number of nearest neighbors')

args = parser.parse_args()

if args.target in {'DBLPv7', 'ACMv9', 'Citationv1'}:
    path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data/Citation', args.target)
    target_dataset = CitationDataset(path, args.target)
    names = ['DBLPv7', 'ACMv9', 'Citationv1']
elif args.target in {'DE', 'EN'}:
    path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data/Twitch', args.target)
    target_dataset = TwitchDataset(path, args.target)
    names = ['RU', 'PTBR', 'FR', 'ES']
elif args.target in {'CSBM-G4'}:
    path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data/CSBM', args.target)
    target_dataset = CSBMDataset(path, args.target)
    names = ['CSBM-G1', 'CSBM-G2', 'CSBM-G3']

data = target_dataset[0]

args.num_classes = len(np.unique(data.y.numpy()))

args.num_features = data.x.size(1)
args.num_nodes = data.x.size(0)

src = []
for name in names:
    if name == args.target:
        continue
    src.append(name)

args.src = src

print(args)

# Load pre-trained model from source domain
model_list = []
param_group = []
for i in range(len(src)):
    model = NodeClassificationModel(args).to(args.device)
    model.load_state_dict(torch.load('pretrain/model_' + src[i] + '.pth'))
    model_list.append(model)
    param_group += list(model.parameters())

weight_list = []
for model in model_list:
    w_list = []
    for name, param in model.gnn.named_parameters():
        if name[-10:] == 'lin.weight' and args.gnn == 'gcn':
            w_list.append(param)
        elif name[-12:] == 'lin_l.weight' and args.gnn == 'sage':
            w_list.append(param)
        elif name[-14:] == 'lin_src.weight' and args.gnn == 'gat':
            w_list.append(param)
        elif name[-11:] == 'nn.0.weight' and args.gnn == 'gin':
            w_list.append(param)
    weight_list.append(w_list)

weight_listv2 = list(zip(*weight_list))

model = GraphATANode(args, weight_listv2, model_list).to(args.device)
param_group += list(model.parameters())

data = data.to(args.device)

optimizer_t = torch.optim.Adam(param_group, lr=args.lr, weight_decay=args.weight_decay)

def train_target(target_data):
    t = time.time()

    mem_fea = torch.rand(target_data.x.size(0), args.nhid).to(args.device)
    mem_cls = torch.ones(target_data.x.size(0), args.num_classes).to(args.device) / args.num_classes

    for epoch in range(args.epochs):
        model.train()
        optimizer_t.zero_grad()
        feat_output = model.feat_bottleneck(target_data.x, target_data.edge_index)
        cls_output = model.feat_classifier(feat_output)
        softmax_out = F.softmax(cls_output, dim=1)
        entropy_loss = torch.mean(Entropy(softmax_out))
        mean_softmax = softmax_out.mean(dim=0)
        div_loss = torch.sum(mean_softmax * torch.log(mean_softmax + 1e-5))
        im_loss = entropy_loss + div_loss

        feat_norm = F.normalize(feat_output, dim=1)
        mem_fea_norm = F.normalize(mem_fea, dim=1)
        distance = feat_norm @ mem_fea_norm.T
        _, idx_near = torch.topk(distance, dim=-1, largest=True, k=args.K + 1)
        idx_near = idx_near[:, 1:]
        pred_near = torch.mean(mem_cls[idx_near], dim=1)
        _, preds = torch.max(pred_near, dim=1)
        cls_loss = F.cross_entropy(cls_output, preds) 
        loss = im_loss + cls_loss

        loss.backward()
        optimizer_t.step()

        model.eval()
        with torch.no_grad():
            feat_output = model.feat_bottleneck(target_data.x, target_data.edge_index)
            cls_output = model.feat_classifier(feat_output)
            softmax_out = F.softmax(cls_output, dim=1)
            outputs_target = softmax_out**2 / ((softmax_out**2).sum(dim=0))
        
        mem_cls = (1.0 - args.momentum) * mem_cls + args.momentum * outputs_target.clone()
        mem_fea = (1.0 - args.momentum) * mem_fea + args.momentum * feat_output.clone()

        test_acc, _, _, _ = evaluate(target_data, model)
        print('Epoch: {:04d}'.format(epoch + 1), 'loss: {:.6f}'.format(loss), 'accuracy: {:.6f}'.format(test_acc), 'time: {:.6f}s'.format(time.time() - t))
    print('Optimization Finished!')


    return test_acc


if __name__ == '__main__':
    test_acc = train_target(data)
    print('After adaptation, test set results accuracy = {:.6f}'.format(test_acc))

    with open('results.txt', 'a+') as f:
        f.write(args.target + ',' + str(test_acc) + ',' + str(test_acc) + ',' + str(test_acc) + '\n')

    print('*'*50)
    print('Before adaptation, the performance of each model is as follows: ')
    for i in range(len(src)):
        model = NodeClassificationModel(args).to(args.device)
        model.load_state_dict(torch.load('pretrain/model_' + src[i] + '.pth'))
        test_acc, macro_f1, micro_f1, pretrain_test_loss = evaluate(data, model)
        print('Source: ' + src[i] + ', ' + 'accuracy: ' + str(test_acc))

