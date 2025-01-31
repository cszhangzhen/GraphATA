import os.path as osp
import torch
import numpy as np
import torch.nn.functional as F

from typing import Any
from torch import Tensor
from sklearn.metrics import f1_score, roc_auc_score

import matplotlib.pyplot as plt
from sklearn import manifold
from sklearn.preprocessing import Normalizer


def compute_test_graph(model, loader, args):
    model.eval()
    correct = 0.0
    loss_test = 0.0
    for data in loader:
        data = data.to(args.device)
        out = model(data.x, data.edge_index, data.batch)
        pred = out.max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
        loss_test += F.nll_loss(out, data.y).item()
    return correct / len(loader.dataset), loss_test


def compute_test(mask, model, data):
    model.eval()
    with torch.no_grad():
        output = model(data.x, data.edge_index)
        loss = F.nll_loss(output[mask], data.y[mask])
        pred = output[mask].max(dim=1)[1]
        correct = pred.eq(data.y[mask]).sum().item()
        acc = correct * 1.0 / (mask.sum().item())

    return acc, loss


def evaluate(data, model):
    with torch.no_grad():
        model.eval()
        output = model(data.x, data.edge_index)
        loss = F.nll_loss(output, data.y)
        pred = output.max(dim=1)[1]
        correct = pred.eq(data.y).sum().item()
        acc = correct * 1.0 / len(data.y)

        preds = pred.cpu().numpy()
        labels = data.y.cpu().numpy()
        macro_f1 = f1_score(labels, preds, average='macro')
        micro_f1 = f1_score(labels, preds, average='micro')

    return acc, macro_f1, micro_f1, loss


def constant(value: Any, fill_value: float):
    if isinstance(value, Tensor):
        value.data.fill_(fill_value)
    else:
        for v in value.parameters() if hasattr(value, 'parameters') else []:
            constant(v, fill_value)
        for v in value.buffers() if hasattr(value, 'buffers') else []:
            constant(v, fill_value)


def zeros(value: Any):
    constant(value, 0.)


def Entropy(input_):
    entropy = -input_ * torch.log(input_ + 1e-5)
    entropy = torch.sum(entropy, dim=1)
    return entropy


def generalized_cross_entropy(y_true, y_pred):
    """
    2018 - nips - Generalized Cross Entropy Loss for Training Deep Neural Networks with Noisy Labels.
    """
    q = 0.7
    t_loss = (1 - torch.pow(torch.sum(y_true * y_pred, dim=-1), q)) / q
    return torch.mean(t_loss)
