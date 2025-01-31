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
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.transforms import Constant

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=2024, help='random seed')
parser.add_argument('--batch_size', type=int, default=2000, help='batch size')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.001, help='weight decay')
parser.add_argument('--nhid', type=int, default=64, help='hidden size')
parser.add_argument('--dropout_ratio', type=float, default=0.1, help='dropout ratio')
parser.add_argument('--device', type=str, default='cuda:2', help='specify cuda devices')
parser.add_argument('--target', type=str, default='Mutagenicity_M4', help='target domain data')
parser.add_argument('--epochs', type=int, default=200, help='maximum number of epochs')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--num_layers', type=int, default=3, help='number of gnn layers')
parser.add_argument('--gnn', type=str, default='gcn', help='different types of gnns')
parser.add_argument('--use_bn', type=bool, default=False, help='do not use batchnorm')
parser.add_argument('--K', type=int, default=40, help='number of nearest neighbors')

args = parser.parse_args()

if args.target in {'PROTEINS_P4'}:
    path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data/PROTEINS', args.target)
    target_dataset = GraphTUDataset(path, args.target)
    names = ['PROTEINS_P1', 'PROTEINS_P2', 'PROTEINS_P3']
elif args.target in {'Mutagenicity_M4'}:
    path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data/Mutagenicity', args.target)
    target_dataset = GraphTUDataset(path, args.target)
    names = ['Mutagenicity_M1', 'Mutagenicity_M2', 'Mutagenicity_M3']
elif args.target in {'FRANKENSTEIN_F4'}:
    path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data/FRANKENSTEIN', args.target)
    target_dataset = GraphTUDataset(path, args.target)
    names = ['FRANKENSTEIN_F1', 'FRANKENSTEIN_F2', 'FRANKENSTEIN_F3']

args.num_classes = target_dataset.num_classes
args.num_features = target_dataset.num_features
args.num_graphs = len(target_dataset)

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
    model = GraphClassificationModel(args).to(args.device)
    model.load_state_dict(torch.load('pretrain/model_' + src[i] + '.pth'))
    model_list.append(model)
    param_group += list(model.parameters())

weight_list = []
for model in model_list:
    w_list = []
    for name, param in model.gnn.named_parameters():
        if name[-10:] == 'lin.weight':
            w_list.append(param)
    weight_list.append(w_list)

weight_listv2 = list(zip(*weight_list))

model = GraphATAGraph(args, weight_listv2, model_list).to(args.device)
param_group += list(model.parameters())

optimizer_t = torch.optim.Adam(param_group, lr=args.lr, weight_decay=args.weight_decay)

loader = DataLoader(target_dataset, batch_size=args.batch_size, shuffle=False)


def train_target():
    t = time.time()

    mem_fea = torch.rand(len(target_dataset), args.nhid).to(args.device)
    mem_cls = torch.ones(len(target_dataset), args.num_classes).to(args.device) / args.num_classes

    for epoch in range(args.epochs):
        for i, data in enumerate(loader):
            optimizer_t.zero_grad()
            data = data.to(args.device)
            feat_output = model.feat_bottleneck(data.x, data.edge_index, data.batch)
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
                feat_output = model.feat_bottleneck(data.x, data.edge_index, data.batch)
                cls_output = model.feat_classifier(feat_output)
                softmax_out = F.softmax(cls_output, dim=1)
                outputs_target = softmax_out**2 / ((softmax_out**2).sum(dim=0))
        
            mem_cls = (1.0 - args.momentum) * mem_cls + args.momentum * outputs_target.clone()
            mem_fea = (1.0 - args.momentum) * mem_fea + args.momentum * feat_output.clone()

        test_acc, _ = compute_test_graph(model, loader, args)
        print('Epoch: {:04d}'.format(epoch + 1), 'loss: {:.6f}'.format(loss), 'accuracy: {:.6f}'.format(test_acc), 'time: {:.6f}s'.format(time.time() - t))
    print('Optimization Finished!')

    return test_acc


if __name__ == '__main__':
    test_acc = train_target()
    print('After adaptation, test set results accuracy = {:.6f}'.format(test_acc))

    with open('results-graph.txt', 'a+') as f:
        f.write(args.target + ',' + str(test_acc) + ',' + str(test_acc) + ',' + str(test_acc) + '\n')

    print('*'*50)
    print('Before adaptation, the performance of each model is as follows: ')
    for i in range(len(src)):
        model = GraphClassificationModel(args).to(args.device)
        model.load_state_dict(torch.load('pretrain/model_' + src[i] + '.pth'))
        test_acc, _ = compute_test_graph(model, loader, args)
        print('Source: ' + src[i] + ', ' + 'accuracy: ' + str(test_acc))

