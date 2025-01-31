import argparse
import glob
import os
import time

import torch
import torch.nn.functional as F
from model import *
from utils import *
from datasets import *
import numpy as np
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.transforms import Constant


parser = argparse.ArgumentParser()

parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.001, help='weight decay')
parser.add_argument('--batch_size', type=int, default=2000, help='batch size')
parser.add_argument('--nhid', type=int, default=64, help='hidden size')
parser.add_argument('--dropout_ratio', type=float, default=0.1, help='dropout ratio')
parser.add_argument('--device', type=str, default='cuda:3', help='specify cuda devices')
parser.add_argument('--target', type=str, default='Mutagenicity_M4', help='source domain data')
parser.add_argument('--epochs', type=int, default=2000, help='maximum number of epochs')
parser.add_argument('--patience', type=int, default=100, help='patience for early stopping')
parser.add_argument('--num_layers', type=int, default=3, help='number of gnn layers')
parser.add_argument('--gnn', type=str, default='gcn', help='different types of gnns')
parser.add_argument('--use_bn', type=bool, default=False, help='do not use batchnorm')

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

print(args)

def train_source(model, train_loader, val_loader, optimizer):
    min_loss = 1e10
    patience_cnt = 0
    val_loss_values = []
    best_epoch = 0

    t = time.time()
    model.train()
    for epoch in range(args.epochs):
        train_loss = 0.0
        correct = 0
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            data = data.to(args.device)
            output = model(data.x, data.edge_index, data.batch)
            loss = F.nll_loss(output, data.y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pred = output.max(dim=1)[1]
            correct += pred.eq(data.y).sum().item()
        train_acc = correct / len(train_loader.dataset)

        val_acc, val_loss = compute_test_graph(model, val_loader, args)

        print('Epoch: {:04d}'.format(epoch + 1), 'train_loss: {:.6f}'.format(train_loss),
              'train_acc: {:.6f}'.format(train_acc), 'loss_val: {:.6f}'.format(val_loss),
              'acc_val: {:.6f}'.format(val_acc), 'time: {:.6f}s'.format(time.time() - t))

        val_loss_values.append(val_loss)
        torch.save(model.state_dict(), '{}.pth'.format(epoch))
        
        if val_loss_values[-1] < min_loss:
            min_loss = val_loss_values[-1]
            best_epoch = epoch
            patience_cnt = 0
        else:
            patience_cnt += 1

        if patience_cnt == args.patience:
            break

        files = glob.glob('*.pth')
        for f in files:
            epoch_nb = int(f.split('.')[0])
            if epoch_nb < best_epoch:
                os.remove(f)

    files = glob.glob('*.pth')
    for f in files:
        epoch_nb = int(f.split('.')[0])
        if epoch_nb > best_epoch:
            os.remove(f)
    print('Optimization Finished! Total time elapsed: {:.6f}'.format(time.time() - t))

    return best_epoch


if __name__ == '__main__':
    for name in names:
        if name == args.target:
            continue
        
        if name in {'PROTEINS_P1', 'PROTEINS_P2', 'PROTEINS_P3', 'PROTEINS_P4'}:
            path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data/PROTEINS', name)
            source_dataset = GraphTUDataset(path, name)
        elif name in {'Mutagenicity_M1', 'Mutagenicity_M2', 'Mutagenicity_M3', 'Mutagenicity_M4'}:
            path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data/Mutagenicity', name)
            source_dataset = GraphTUDataset(path, name)
        elif name in {'FRANKENSTEIN_F1', 'FRANKENSTEIN_F2', 'FRANKENSTEIN_F3', 'FRANKENSTEIN_F4'}:
            path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data/FRANKENSTEIN', name)
            source_dataset = GraphTUDataset(path, name)
        
        num_training = int(len(source_dataset) * 0.8)
        num_val = int(len(source_dataset) * 0.1)
        num_test = len(source_dataset) - (num_training + num_val)
        training_set, validation_set, test_set = random_split(source_dataset, [num_training, num_val, num_test])

        train_loader = DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(validation_set, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

        model = GraphClassificationModel(args).to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        best_model = train_source(model, train_loader, val_loader, optimizer)
        model.load_state_dict(torch.load('{}.pth'.format(best_model)))
        test_acc, test_loss = compute_test_graph(model, test_loader, args)
        print('Source {} test set results, loss = {:.6f}, accuracy = {:.6f}'.format(name, test_loss, test_acc))

        loader = DataLoader(target_dataset, batch_size=args.batch_size, shuffle=False)
        pretrain_test_acc, pretrain_test_loss = compute_test_graph(model, loader, args)
        print('Source model {} for target data, loss = {:.6f}, accuracy = {:.6f}'.format(name, pretrain_test_loss, pretrain_test_acc))

        # Save source model
        torch.save(model.state_dict(), 'pretrain/model_' + name + '.pth')
