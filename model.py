import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GINConv
from layer import *
from torch.nn.parameter import Parameter
from torch.nn import Sequential, Linear
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp


class GCN(nn.Module):
    def __init__(self, args):
        super(GCN, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_classes = args.num_classes
        self.dropout_ratio = args.dropout_ratio
        self.num_layers = args.num_layers

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.convs.append(GCNConv(self.num_features, self.nhid))
        self.bns.append(nn.BatchNorm1d(self.nhid))

        for _ in range(self.num_layers - 1):
            self.convs.append(GCNConv(self.nhid, self.nhid))
            self.bns.append(nn.BatchNorm1d(self.nhid))

        self.cls = torch.nn.Linear(self.nhid, self.num_classes)

        self.activation = F.relu
        self.use_bn = args.use_bn

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None):
        x = self.feat_bottleneck(x, edge_index, edge_weight)
        x = self.feat_classifier(x)

        return x
    
    def feat_bottleneck(self, x, edge_index, edge_weight=None):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_weight)
            if self.use_bn:
                x = self.bns[i](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        return x
    
    def feat_classifier(self, x):
        x = self.cls(x)
        
        return x


class SAGE(nn.Module):
    def __init__(self, args):
        super(SAGE, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_classes = args.num_classes
        self.dropout_ratio = args.dropout_ratio
        self.num_layers = args.num_layers

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.convs.append(SAGEConv(self.num_features, self.nhid))
        self.bns.append(nn.BatchNorm1d(self.nhid))

        for _ in range(self.num_layers - 1):
            self.convs.append(SAGEConv(self.nhid, self.nhid))
            self.bns.append(nn.BatchNorm1d(self.nhid))

        self.cls = torch.nn.Linear(self.nhid, self.num_classes)

        self.activation = F.relu
        self.use_bn = args.use_bn

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None):
        x = self.feat_bottleneck(x, edge_index, edge_weight)
        x = self.feat_classifier(x)

        return x
    
    def feat_bottleneck(self, x, edge_index, edge_weight=None):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_weight)
            
            if self.use_bn:
                x = self.bns[i](x)
            
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        return x
    
    def feat_classifier(self, x):
        x = self.cls(x)
        
        return x


class GAT(nn.Module):
    def __init__(self, args):
        super(GAT, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_classes = args.num_classes
        self.dropout_ratio = args.dropout_ratio
        self.num_layers = args.num_layers

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.convs.append(GATConv(self.num_features, self.nhid, heads=1, concat=False))
        self.bns.append(nn.BatchNorm1d(self.nhid))

        for _ in range(self.num_layers - 1):
            self.convs.append(GATConv(self.nhid, self.nhid, heads=1, concat=False))
            self.bns.append(nn.BatchNorm1d(self.nhid))

        self.cls = torch.nn.Linear(self.nhid, self.num_classes)

        self.activation = F.relu
        self.use_bn = args.use_bn

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, edge_index, edge_weight):
        x = self.feat_bottleneck(x, edge_index)
        x = self.feat_classifier(x)

        return x
    
    def feat_bottleneck(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if self.use_bn:
                x = self.bns[i](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        return x
    
    def feat_classifier(self, x):
        x = self.cls(x)
        
        return x


class GIN(nn.Module):
    def __init__(self, args):
        super(GIN, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_classes = args.num_classes
        self.dropout_ratio = args.dropout_ratio
        self.num_layers = args.num_layers

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # self.lin = torch.nn.Linear(self.num_features, self.nhid)

        self.convs.append(GINConv(Sequential(Linear(self.num_features, self.nhid)), train_eps=True))
        self.bns.append(nn.BatchNorm1d(self.nhid))

        for _ in range(self.num_layers - 1):
            self.convs.append(GINConv(Sequential(Linear(self.nhid, self.nhid)), train_eps=True))
            self.bns.append(nn.BatchNorm1d(self.nhid))

        self.cls = torch.nn.Linear(self.nhid, self.num_classes)

        self.activation = F.relu
        self.use_bn = args.use_bn

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None):
        x = self.feat_bottleneck(x, edge_index, edge_weight)
        x = self.feat_classifier(x)

        return x
    
    def feat_bottleneck(self, x, edge_index, edge_weight=None):

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_weight)
            if self.use_bn:
                x = self.bns[i](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        return x
    
    def feat_classifier(self, x):
        x = self.cls(x)
        
        return x


class NodeClassificationModel(torch.nn.Module):
    def __init__(self, args):
        super(NodeClassificationModel, self).__init__()
        self.args = args
        
        if args.gnn == 'gcn':
            self.gnn = GCN(args)
        elif args.gnn == 'sage':
            self.gnn = SAGE(args)
        elif args.gnn == 'gat':
            self.gnn = GAT(args)
        elif args.gnn == 'gin':
            self.gnn = GIN(args)
        else:
            assert args.gnn in ('gcn', 'sage', 'gat', 'gin'), 'Invalid gnn'
                
        self.reset_parameters()
    
    def reset_parameters(self):
        self.gnn.reset_parameters()
    
    def forward(self, x, edge_index):
        x = self.feat_bottleneck(x, edge_index)
        x = self.feat_classifier(x)
        
        return F.log_softmax(x, dim=1)

    def feat_bottleneck(self, x, edge_index):
        x = self.gnn.feat_bottleneck(x, edge_index)
        return x
    
    def feat_classifier(self, x):
        x = self.gnn.feat_classifier(x)
        
        return x


class GraphClassificationModel(torch.nn.Module):
    def __init__(self, args):
        super(GraphClassificationModel, self).__init__()
        self.args = args
        
        if args.gnn == 'gcn':
            self.gnn = GCN(args)
        elif args.gnn == 'sage':
            self.gnn = SAGE(args)
        elif args.gnn == 'gat':
            self.gnn = GAT(args)
        elif args.gnn == 'gin':
            self.gnn = GIN(args)
        else:
            assert args.gnn in ('gcn', 'sage', 'gat', 'gin'), 'Invalid gnn'
                
        self.reset_parameters()
    
    def reset_parameters(self):
        self.gnn.reset_parameters()
    
    def forward(self, x, edge_index, batch):
        x = self.feat_bottleneck(x, edge_index, batch)
        x = self.feat_classifier(x)
        
        return F.log_softmax(x, dim=1)

    def feat_bottleneck(self, x, edge_index, batch):
        x = self.gnn.feat_bottleneck(x, edge_index)
        x = gap(x, batch)
        return x
    
    def feat_classifier(self, x):
        x = self.gnn.feat_classifier(x)
        
        return x


class GraphATANode(torch.nn.Module):
    def __init__(self, args, model_weights, model_list):
        super(GraphATANode, self).__init__()
        self.args = args
        self.src = args.src
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_classes = args.num_classes
        self.dropout_ratio = args.dropout_ratio

        self.model_weights = model_weights
        self.model_list = model_list

        self.num_layers = args.num_layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.convs.append(NodeCentricConv(self.num_features, self.nhid, model_weights[0]))
        self.bns.append(nn.BatchNorm1d(self.nhid))

        for i in range(self.num_layers - 1):
            self.convs.append(NodeCentricConv(self.nhid, self.nhid, model_weights[i+1]))
            self.bns.append(nn.BatchNorm1d(self.nhid))

        self.activation = F.relu
        self.use_bn = args.use_bn

        self.cls = MLPModule(args, model_list)
    
    def forward(self, x, edge_index):
        x = self.feat_bottleneck(x, edge_index)
        x = self.feat_classifier(x)
        
        return F.log_softmax(x, dim=1)

    def feat_bottleneck(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if self.use_bn:
                x = self.bns[i](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout_ratio, training=self.training)

        return x
    
    def feat_classifier(self, x):
        x = self.cls(x)
        
        return x


class GraphATAGraph(torch.nn.Module):
    def __init__(self, args, model_weights, model_list):
        super(GraphATAGraph, self).__init__()
        self.args = args
        self.src = args.src
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_classes = args.num_classes
        self.dropout_ratio = args.dropout_ratio

        self.model_weights = model_weights
        self.model_list = model_list

        self.num_layers = args.num_layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.convs.append(NodeCentricConv(self.num_features, self.nhid, model_weights[0]))
        self.bns.append(nn.BatchNorm1d(self.nhid))

        for i in range(self.num_layers - 1):
            self.convs.append(NodeCentricConv(self.nhid, self.nhid, model_weights[i+1]))
            self.bns.append(nn.BatchNorm1d(self.nhid))

        self.activation = F.relu
        self.use_bn = args.use_bn

        self.cls = MLPModule(args, model_list)
    
    def forward(self, x, edge_index, batch):
        x = self.feat_bottleneck(x, edge_index, batch)
        x = self.feat_classifier(x)
        
        return F.log_softmax(x, dim=1)

    def feat_bottleneck(self, x, edge_index, batch):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if self.use_bn:
                x = self.bns[i](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        
        x = gap(x, batch) + gmp(x, batch)

        return x
    
    def feat_classifier(self, x):
        x = self.cls(x)
        
        return x

