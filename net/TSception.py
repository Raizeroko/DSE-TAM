# TSception: https://ieeexplore.ieee.org/document/9762054
import torch
import torch.nn as nn


class TSception(nn.Module):
    def conv_block(self, in_chan, out_chan, kernel, step, pool):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_chan, out_channels=out_chan,
                      kernel_size=kernel, stride=step),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=(1, pool), stride=(1, pool)))

    def __init__(self, num_classes, input_size, sampling_rate, num_T, num_S, hidden, dropout_rate):
        # input_size: 1 x EEG channel x datapoint
        super(TSception, self).__init__()
        self.inception_window = [0.5, 0.25, 0.125]
        self.pool = 8
        # by setting the convolutional kernel being (1,lenght) and the strids being 1 we can use conv2d to
        # achieve the 1d convolution operation
        self.Tception1 = self.conv_block(1, num_T, (1, int(self.inception_window[0] * sampling_rate)), 1, self.pool)
        self.Tception2 = self.conv_block(1, num_T, (1, int(self.inception_window[1] * sampling_rate)), 1, self.pool)
        self.Tception3 = self.conv_block(1, num_T, (1, int(self.inception_window[2] * sampling_rate)), 1, self.pool)

        self.Sception1 = self.conv_block(num_T, num_S, (int(input_size[1]), 1), 1, int(self.pool*0.25))
        self.Sception2 = self.conv_block(num_T, num_S, (int(input_size[1] * 0.5), 1), (int(input_size[1] * 0.5), 1),
                                         int(self.pool*0.25))
        self.fusion_layer = self.conv_block(num_S, num_S, (3, 1), 1, 4)
        self.BN_t = nn.BatchNorm2d(num_T)
        self.BN_s = nn.BatchNorm2d(num_S)
        self.BN_fusion = nn.BatchNorm2d(num_S)

        self.fc = nn.Sequential(
            nn.Linear(num_S, hidden),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden, num_classes)
        )

    def forward(self, x):
        x = torch.unsqueeze(x, dim=1)  # (batch, 1, chan, time)
        y = self.Tception1(x)
        out = y
        y = self.Tception2(x)
        out = torch.cat((out, y), dim=-1)
        y = self.Tception3(x)
        out = torch.cat((out, y), dim=-1)
        out = self.BN_t(out)
        z = self.Sception1(out)
        out_ = z
        z = self.Sception2(out)
        out_ = torch.cat((out_, z), dim=2)
        out = self.BN_s(out_)
        out = self.fusion_layer(out)
        out = self.BN_fusion(out)
        out = torch.squeeze(torch.mean(out, dim=-1), dim=-1)
        out = self.fc(out)
        return out



if __name__ == '__main__':
    from torchinfo import summary

    params = {
        # -----------网络参数-------------------------------------------------------
        'emb_dim': 48,  # embedding dimension of Embedding, Self-Attention, Mamba
        'emb_kernel': 16,  # 2D-conv embedding length of Embedding
        'd_state': 16,  # d_state of Mamba2
        'd_conv': 4,  # d_conv of Mamba2
        'expand': 4,  # expand of Mamba2
        'headdim': 8,  # headdim of Mamba2
        'num_layers': 1,  # d_conv of MambaFormer
        'num_heads': 8,  # num head of Self-Attention
        'dropout': 0.5,  # dropout of Embedding, Self-Attention, Mamba
        'lr': 1e-3,  # learning rate
        'weight_decay': 1e-4,  # L2-norm weight decay
        'time': 384,  # window size of EEG
        'num_classes': 2,
        'num_electrodes': 32,
        # -----------训练参数-------------------------------------------------------
        'seed': 20,  # set random seed
        'epoch': 200,  # training epoch
        'batch_size': 256,  # training batch size
        'session': 2,  # dataset session: 1/2/3 (SEED:session1/2/3,SEEDIV:session1/2/3, DEAP:Arousal/Valence/Dominance)
        'val': "LOSO",  # experiment validation：WS/WSSS/LOSO/KFold
        'shuffle': 'Trial',  # validation shuffle way: Sample/Trial
        'KFold': 6,  # if 'val'=='KFold': K numbers
        'net': "Conformer",  # Choose net：ACRNN/Mamba
        'dataset': 'DREAMER',  # choose dataset: DEAP/DREAMER/SEED/SEEDIV
        'feature': "Time",  # input feature: Time/DE
        'device': torch.device("cuda:1")  # training device
    }

    net = TSception(
            num_classes=params['num_classes'], input_size=(1, params['num_electrodes'], params['time']),
            sampling_rate=128, num_T=15, num_S=15,
            hidden=32, dropout_rate=params['dropout']).cuda()
    print(net.parameters())
    summary(net, (20, 32, 384), depth=4)