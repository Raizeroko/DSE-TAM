import numpy as np
import torch
import torch.nn as nn


class dense(nn.Module):
    def __init__(self, input_dim1, input_dim2, hidden_dim,
                 activation=lambda x: x):
        super().__init__()
        self.W1 = nn.Parameter(torch.Tensor(np.random.normal(size=(input_dim1, hidden_dim))))
        self.W2 = nn.Parameter(torch.Tensor(np.random.normal(size=(input_dim2, hidden_dim))))
        self.b = nn.Parameter(torch.Tensor(np.zeros(hidden_dim)))
        self.activation = activation
        self.vector = nn.Linear(input_dim2, input_dim2)

    def forward(self, x):
        y = self.vector(x)
        return self.activation(torch.matmul(x, self.W1) + torch.matmul(y, self.W2) + self.b)


class self_attention(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(self_attention, self).__init__()
        self.q = input_dim
        self.k = input_dim
        self.hidden = hidden_dim
        self.dense = dense(self.q, self.k, self.k)
        self.self_attention = nn.Sequential(
            nn.ELU(),
            nn.Linear(self.k, self.k)
        )
        self.softmax = nn.Softmax(dim=2)
        self.dropout = nn.Dropout()

    def forward(self, x):
        # print(x.shape)
        y = self.dense(x)
        # print(y.shape)
        z = self.self_attention(y)
        # print(z.shape)
        p = z * x
        p = self.softmax(p)
        # print(p.shape)
        A = p * x
        # print(A.shape)
        A = A.reshape(-1, self.k)
        A = self.dropout(A)
        return A

class LSTM(nn.Module):
    def __init__(self,hidden_dim):
        super(LSTM,self).__init__()
        self.hidden_dim = hidden_dim
        # lstm
        self.lstm = nn.LSTM(
            input_size=1080,
            hidden_size=self.hidden_dim,
            num_layers=2,
            batch_first=True
            )

    def forward(self,x,hidden0=None):
        # x = x.reshape(-1,1,1080)
        x = x.reshape(x.shape[0], 1, -1)
        q ,(hidden,cell) = self.lstm(x)
        h = hidden[1].reshape(-1,1,64)
        c = cell[1].reshape(-1,1,64)
        return h,c

class CNN(nn.Module):
    def __init__(self,ic,ih,iw,kh,kw,ks,ph,pw,ps,oc):
        super(CNN,self).__init__()
        # input
        self.input_channel = ic
        self.input_height =  ih
        self.input_width = iw
        self.output_channel = oc
        self.kernel_height = kh
        self.kernel_width = kw
        self.kernel_stride = ks
        self.pooling_height = ph
        self.pooling_width = pw
        self.pooling_stride = ps
        # CNN
        self.conv = nn.Sequential(
            nn.Conv2d(self.input_channel,self.output_channel,(self.kernel_height,self.kernel_width),self.kernel_stride),
            nn.ELU(),
            nn.MaxPool2d((self.pooling_height,self.pooling_width),self.pooling_stride)
        )
        # dropout
        self.dropout = nn.Dropout2d(p=0.5)
    def __call__(self,x):
        x = x.permute(0, 1, 3, 2)
        c = self.conv(x)
        # c1 = c.reshape(800,-1)
        cd = self.dropout(c)
        return cd


class channel_wise_attention(nn.Module):
    def __init__(self, H, W, C, reduce):
        super(channel_wise_attention, self).__init__()
        self.H = H
        self.W = W
        self.C = C
        self.r = reduce
        # fc layer
        self.fc = nn.Sequential(
            nn.Linear(self.C, self.r),
            nn.Tanh(),
            nn.Linear(self.r, self.C)
        )
        # softmax
        self.softmax = nn.Softmax(dim=3)

    def forward(self, x):
        # mean pooling
        x1 = x.permute(0, 3, 1, 2)
        mean = nn.AvgPool2d((1, 384))
        feature_map = mean(x1).permute(0, 2, 3, 1)
        # FC Layer
        # feature_map : [800,1,1,C]
        feature_map_fc = self.fc(feature_map)

        # softmax
        v = self.softmax(feature_map_fc)
        # channel_wise_attention
        v = v.reshape(-1, self.C)
        vr = torch.reshape(torch.cat([v] * (self.H * self.W), axis=1), [-1, self.H, self.W, self.C])
        channel_wise_attention_fm = x * vr
        return v, channel_wise_attention_fm


class ACRNN(nn.Module):
    def __init__(self,input_height):
        super(ACRNN,self).__init__()
        self.H = 1
        self.W = 384
        self.C = input_height
        self.reduce = 15
        self.channel_wise_attention = channel_wise_attention(self.H,self.W,self.C,self.reduce)
        self.output_channel = 40
        self.kernel_height = 32
        self.kernel_width = 45
        self.kernel_stride = 1
        self.pooling_height = 1
        self.pooling_width = 75
        self.pooling_stride = 10
        self.cnn = CNN(self.H,self.C,self.W,self.kernel_height,self.kernel_width,self.kernel_stride,self.pooling_height,self.pooling_width,self.pooling_stride,self.output_channel)
        self.hidden_dim = 64
        self.lstm = LSTM(self.hidden_dim)
        self.hidden = 512
        self.self_attention = self_attention(self.hidden_dim,self.hidden)
        self.num_labels = 2
        self.softmax = nn.Sequential(
            nn.Linear(self.hidden_dim,self.num_labels),
            nn.Softmax(dim=1)
        )
    def forward(self,x):
        x = x.permute(0, 2, 1).unsqueeze(1)
        x_map, x_ca = self.channel_wise_attention(x)
        x_cn = self.cnn(x_ca)
        x_rn, x_c = self.lstm(x_cn)
        x_sa = self.self_attention(x_rn)
        x_sm = self.softmax(x_sa)
        return x_sm



if __name__ == '__main__':
    from torchinfo import summary

    net = ACRNN(32).cuda()
    print(net.parameters())
    summary(net, (20, 32, 384), depth=4)