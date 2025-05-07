# This is the networks script
import torch.nn as nn
import math
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import numpy as np
# import h5py
import os
import _pickle as cPickle

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class GraphConvolution(Module):
    """
    simple GCN layer
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        torch.nn.init.xavier_uniform_(self.weight, gain=1.414)
        if bias:
            self.bias = Parameter(torch.zeros((1, 1, out_features), dtype=torch.float32))
        else:
            self.register_parameter('bias', None)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        output = torch.matmul(x, self.weight)-self.bias
        output = F.relu(torch.matmul(adj, output))
        return output


class PowerLayer(nn.Module):
    '''
    The power layer: calculates the log-transformed power of the data
    '''
    def __init__(self, dim, length, step):
        super(PowerLayer, self).__init__()
        self.dim = dim
        self.pooling = nn.AvgPool2d(kernel_size=(1, length), stride=(1, step))

    def forward(self, x):
        return torch.log(self.pooling(x.pow(2)))


class LGGNet(nn.Module):
    def temporal_learner(self, in_chan, out_chan, kernel, pool, pool_step_rate):
        return nn.Sequential(
            nn.Conv2d(in_chan, out_chan, kernel_size=kernel, stride=(1, 1)),
            PowerLayer(dim=-1, length=pool, step=int(pool_step_rate*pool))
        )

    def __init__(self, num_classes, input_size, sampling_rate, num_T,
                 out_graph, dropout_rate, pool, pool_step_rate, idx, idx_graph):
        # input_size: EEG frequency x channel x datapoint
        super(LGGNet, self).__init__()
        self.i = idx
        self.idx = idx_graph
        self.window = [0.5, 0.25, 0.125]
        self.pool = pool
        self.channel = input_size[1]
        self.brain_area = len(self.idx)

        # by setting the convolutional kernel being (1,lenght) and the strids being 1 we can use conv2d to
        # achieve the 1d convolution operation
        self.Tception1 = self.temporal_learner(input_size[0], num_T,
                                               (1, int(self.window[0] * sampling_rate)),
                                               self.pool, pool_step_rate)
        self.Tception2 = self.temporal_learner(input_size[0], num_T,
                                               (1, int(self.window[1] * sampling_rate)),
                                               self.pool, pool_step_rate)
        self.Tception3 = self.temporal_learner(input_size[0], num_T,
                                               (1, int(self.window[2] * sampling_rate)),
                                               self.pool, pool_step_rate)
        self.BN_t = nn.BatchNorm2d(num_T)
        self.BN_t_ = nn.BatchNorm2d(num_T)
        self.OneXOneConv = nn.Sequential(
            nn.Conv2d(num_T, num_T, kernel_size=(1, 1), stride=(1, 1)),
            nn.LeakyReLU(),
            nn.AvgPool2d((1, 2)))
        # diag(W) to assign a weight to each local areas
        size = self.get_size_temporal(input_size)
        self.local_filter_weight = nn.Parameter(torch.FloatTensor(self.channel, size[-1]),
                                                requires_grad=True)
        nn.init.xavier_uniform_(self.local_filter_weight)
        self.local_filter_bias = nn.Parameter(torch.zeros((1, self.channel, 1), dtype=torch.float32),
                                              requires_grad=True)

        # aggregate function
        self.aggregate = Aggregator(self.idx)

        # trainable adj weight for global network
        self.global_adj = nn.Parameter(torch.FloatTensor(self.brain_area, self.brain_area), requires_grad=True)
        nn.init.xavier_uniform_(self.global_adj)
        # to be used after local graph embedding
        self.bn = nn.BatchNorm1d(self.brain_area)
        self.bn_ = nn.BatchNorm1d(self.brain_area)
        # learn the global network of networks
        self.GCN = GraphConvolution(size[-1], out_graph)

        self.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(int(self.brain_area * out_graph), num_classes))


    def forward(self, x):
        x = x.unsqueeze(1)
        x = x[:, :, self.i, :]
        y = self.Tception1(x)
        out = y
        y = self.Tception2(x)
        out = torch.cat((out, y), dim=-1)
        y = self.Tception3(x)
        out = torch.cat((out, y), dim=-1)
        out = self.BN_t(out)
        out = self.OneXOneConv(out)
        out = self.BN_t_(out)
        out = out.permute(0, 2, 1, 3)
        out = torch.reshape(out, (out.size(0), out.size(1), -1))
        out = self.local_filter_fun(out, self.local_filter_weight)
        out = self.aggregate.forward(out)
        adj = self.get_adj(out)
        out = self.bn(out)
        out = self.GCN(out, adj)
        out = self.bn_(out)
        out = out.view(out.size()[0], -1)
        out = self.fc(out)
        return out

    def get_size_temporal(self, input_size):
        # input_size: frequency x channel x data point
        data = torch.ones((1, input_size[0], input_size[1], int(input_size[2])))
        z = self.Tception1(data)
        out = z
        z = self.Tception2(data)
        out = torch.cat((out, z), dim=-1)
        z = self.Tception3(data)
        out = torch.cat((out, z), dim=-1)
        out = self.BN_t(out)
        out = self.OneXOneConv(out)
        out = self.BN_t_(out)
        out = out.permute(0, 2, 1, 3)
        out = torch.reshape(out, (out.size(0), out.size(1), -1))
        size = out.size()
        return size

    def local_filter_fun(self, x, w):
        w = w.unsqueeze(0).repeat(x.size()[0], 1, 1)
        x = F.relu(torch.mul(x, w) - self.local_filter_bias)
        return x

    def get_adj(self, x, self_loop=True):
        # x: b, node, feature
        adj = self.self_similarity(x)   # b, n, n
        num_nodes = adj.shape[-1]
        adj = F.relu(adj * (self.global_adj + self.global_adj.transpose(1, 0)))
        if self_loop:
            adj = adj + torch.eye(num_nodes).to("cuda:1")
        rowsum = torch.sum(adj, dim=-1)
        mask = torch.zeros_like(rowsum)
        mask[rowsum == 0] = 1
        rowsum += mask
        d_inv_sqrt = torch.pow(rowsum, -0.5)
        d_mat_inv_sqrt = torch.diag_embed(d_inv_sqrt)
        adj = torch.bmm(torch.bmm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
        return adj

    def self_similarity(self, x):
        # x: b, node, feature
        x_ = x.permute(0, 2, 1)
        s = torch.bmm(x, x_)
        return s


class Aggregator():

    def __init__(self, idx_area):
        # chan_in_area: a list of the number of channels within each area
        self.chan_in_area = idx_area
        self.idx = self.get_idx(idx_area)
        self.area = len(idx_area)

    def forward(self, x):
        # x: batch x channel x data
        data = []
        for i, area in enumerate(range(self.area)):
            if i < self.area - 1:
                data.append(self.aggr_fun(x[:, self.idx[i]:self.idx[i + 1], :], dim=1))
            else:
                data.append(self.aggr_fun(x[:, self.idx[i]:, :], dim=1))
        return torch.stack(data, dim=1)

    def get_idx(self, chan_in_area):
        idx = [0] + chan_in_area
        idx_ = [0]
        for i in idx:
            idx_.append(idx_[-1] + i)
        return idx_[1:]

    def aggr_fun(self, x, dim):
        # return torch.max(x, dim=dim).values
        return torch.mean(x, dim=dim)




if __name__ == '__main__':
    from torchinfo import summary

    DEAP_original_order = ['Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'P3', 'P7', 'PO3', 'O1',
                           'Oz', 'Pz', 'Fp2', 'AF4', 'Fz', 'F4', 'F8', 'FC6', 'FC2', 'Cz', 'C4', 'T8', 'CP6', 'CP2',
                           'P4', 'P8', 'PO4', 'O2']
    DREAMER_original_order = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
    # Define proper channel orders for the local-global graphs in LGGNet. Please refer to three graph definitions (general, frontal, hemesphere).
    graph_general_DEAP = [['Fp1', 'Fp2'], ['AF3', 'AF4'], ['F3', 'F7', 'Fz', 'F4', 'F8'],
                          ['FC5', 'FC1', 'FC6', 'FC2'], ['C3', 'Cz', 'C4'], ['CP5', 'CP1', 'CP2', 'CP6'],
                          ['P7', 'P3', 'Pz', 'P4', 'P8'], ['PO3', 'PO4'], ['O1', 'Oz', 'O2'],
                          ['T7'], ['T8']]

    graph_gen_DREAMER = [['AF3', 'AF4'], ['F7', 'F3', 'F4', 'F8'], ['FC5', 'FC6'], ['T7'], ['P7', 'P8'], ['O1', 'O2'],
                         ['T8']]

    graph_idx = graph_gen_DREAMER  # The general graph definition for DEAP is used as an example.
    idx = []
    num_chan_local_graph = []
    for i in range(len(graph_idx)):
        num_chan_local_graph.append(len(graph_idx[i]))
        for chan in graph_idx[i]:
            idx.append(DREAMER_original_order.index(chan))


    data = torch.randn(20, 14, 384).cuda()
    net = LGGNet(2, (1, 14, 384), 128, 64, 32, 0.5, 16, 0.25, idx, num_chan_local_graph).cuda()
    # print(net.parameters())
    result = net(data)
    summary(net, (20, 14, 384), depth=4)
    # print(result.shape)