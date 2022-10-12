

import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.softmax import edge_softmax
import dgl
import math
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module






class GDN(nn.Module):
    def __init__(self, in_feats, out_feats, leaky_relu_negative_slope: float = 0.2):

        super(GDN, self).__init__()

        self.fc = nn.Linear(in_feats, out_feats, bias=True)
        self.linear_t = nn.Linear(in_feats, out_feats, bias=False)
        self.linear_s = nn.Linear(in_feats, out_feats, bias=False)
        self.attn = nn.Linear( out_feats, 1, bias=False)
        self.activation = nn.LeakyReLU(negative_slope=leaky_relu_negative_slope)

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)

    def edge_attention(self, edges):
        hidden_src = self.linear_s(edges.src['h'])
        hidden_dst = self.linear_t(edges.dst['h'])
        e = self.attn(self.activation(hidden_dst + hidden_src))
        return {'e': e}

    def message_func(self, edges):
        return {'h': edges.src['h'] * edges.data['alpha']}  # message divided by weight

    def reduce_func(self, nodes):
        return {'h': torch.sum(nodes.mailbox['h'], dim=1)}

    def forward(self, g, feature):
        with g.local_scope():
            g.ndata['h'] = feature
            g.apply_edges(self.edge_attention)
            g.edata['alpha'] = edge_softmax(g, g.edata['e'], norm_by= 'src')
            g.update_all(self.message_func, self.reduce_func)
            g.ndata['h'] = self.fc(g.ndata['h'])
            h = g.ndata['h']
            return h





class GCN(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GCN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'








class DCRS(nn.Module):
    def __init__(self, fuse, nfeat, nhid):
        super(DCRS, self).__init__()




        self.GDN1 = GDN(nfeat, nhid)
        self.GDN2 = GDN(nhid, nhid)
        self.GDN3 = GDN(nhid, nhid)
        self.GDN4 = GDN(nhid, 1)

        self.GCN1 = GCN(nfeat, nhid)
        self.GCN2 = GCN(nhid, nhid)
        self.GCN3 = GCN(nhid, nhid)
        self.GCN4 = GCN(nhid, 1)

        self.a = fuse



    def forward(self, g, role_adj, feature):


        DiffEmb = torch.relu(self.GDN1(g, feature))
        DiffEmb = torch.relu(self.GDN2(g, DiffEmb))
        DiffEmb = torch.relu(self.GDN3(g, DiffEmb))
        DiffScore = self.GDN4(g, DiffEmb)

        RoleEmb = torch.relu(self.GCN1(feature, role_adj))
        RoleEmb = torch.relu(self.GCN2(RoleEmb, role_adj))
        RoleEmb = torch.relu(self.GCN3(RoleEmb, role_adj))
        RoleScore = self.GCN4(RoleEmb, role_adj)

        DismanScore = torch.sigmoid(torch.add((1-self.a) * DiffScore, self.a * RoleScore))

        return DismanScore

    def loss(self, score, adj, gamma: float = 1, mean: bool = True):
        tmp = 1 + torch.mul(score.unsqueeze(1), adj.cuda())
        tmp = torch.prod(tmp.pow(-1), dim=0)
        loss1 = tmp.mean() if mean else tmp.sum()
        loss2 = score.mean() if mean else score.sum()

        return loss1 + gamma * loss2


