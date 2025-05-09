from einops.layers.torch import Rearrange
from mamba_ssm import Mamba
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvolution(nn.Module):
    def __init__(self, emb_dim, pool, num_elec, emb_kernel):
        super(GraphConvolution, self).__init__()
        self.token_emb = nn.Conv2d(1, emb_dim, kernel_size=(num_elec, emb_kernel), stride=emb_kernel, padding=0)
        self.activation = nn.Tanh()
        self.pooling = nn.LPPool1d(norm_type=2, kernel_size=pool)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        out = torch.matmul(adj, x)
        token_emb = out.unsqueeze(1)
        token_emb = self.token_emb(token_emb)
        token_emb = token_emb.squeeze(2)
        token_emb = self.activation(token_emb)
        token_emb = self.pooling(token_emb)
        token_emb = token_emb.permute(0, 2, 1)  # (batch, new_length, emb_dim)
        return token_emb



def normalize_A(A: torch.Tensor, symmetry: bool=False):
    A = F.relu(A)
    if symmetry:
        A = A + torch.transpose(A, 0, 1)
        d = torch.sum(A, 1)
        d = 1 / torch.sqrt(d + 1e-10)
        D = torch.diag_embed(d)
        L = torch.matmul(torch.matmul(D, A), D)
    else:
        d = torch.sum(A, 1)
        d = 1 / torch.sqrt(d + 1e-10)
        D = torch.diag_embed(d)
        L = torch.matmul(torch.matmul(D, A), D)
    return L


def generate_cheby_adj(A: torch.Tensor, num_layers: int):
    support = []
    for i in range(num_layers):
        if i == 0:
            support.append(torch.eye(A.shape[1]).to(A.device))
        elif i == 1:
            support.append(A)
        else:
            temp = torch.matmul(support[-1], A)
            support.append(temp)
    return support


class ChebyNet(nn.Module):
    def __init__(self, emb_dim, pool, num_elec, emb_kernel, g_layers):
        super(ChebyNet, self).__init__()
        self.num_layers = g_layers
        self.gc = nn.ModuleList()
        for i in range(g_layers):
            self.gc.append(GraphConvolution(emb_dim, pool, num_elec, emb_kernel))

    def forward(self, x, L):
        adj = generate_cheby_adj(L, self.num_layers)
        for i in range(len(self.gc)):
            if i == 0:
                result = self.gc[i](x, adj[i])
            else:
                result += self.gc[i](x, adj[i])
        result = F.relu(result)
        return result

class DGEmbedding(nn.Module):
    def __init__(self, num_elec, emb_dim, pool, emb_kernel, g_layers, time):
        super(DGEmbedding, self).__init__()
        self.layer = ChebyNet(emb_dim, pool, num_elec, emb_kernel, g_layers)
        self.BN = nn.BatchNorm1d(time)
        self.A = nn.Parameter(torch.FloatTensor(num_elec, num_elec))
        nn.init.xavier_normal_(self.A)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.BN(x.transpose(1, 2)).transpose(1, 2)
        L = normalize_A(self.A, True)
        result = self.layer(x, L)
        return result

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, maxlen):
        super(PositionalEmbedding, self).__init__()
        self.encoding = torch.zeros(maxlen, d_model)
        self.encoding.requires_grad_(False)

        pos = torch.arange(0, maxlen)
        pos = pos.float().unsqueeze(1)
        _2i = torch.arange(0, d_model, 2)
        self.encoding[:, 0::2] = torch.sin(pos / 10000 ** (_2i / d_model))
        self.encoding[:, 1::2] = torch.cos(pos / 10000 ** (_2i / d_model))

    def forward(self, x):
        seq_len = x.shape[1]
        return self.encoding[:seq_len, :].cuda()

class DSE(nn.Module):
    def __init__(self, num_elec, emb_dim, pool, emb_kernel, time, g_layers, max_len=1024):
        super(DSE, self).__init__()
        self.dg_emb = DGEmbedding(num_elec, emb_dim, pool, emb_kernel, g_layers, time)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_emb2 = PositionalEmbedding(emb_dim, max_len)
        self.pos_emb = nn.Embedding(max_len, emb_dim)
        self.register_buffer('position_ids', torch.arange(max_len).expand((1, -1)))

    def forward(self, x):
        # 普通嵌入
        # token_emb = x.unsqueeze(1)  # (batch, 1, feature, length)
        # token_emb = self.token_emb(token_emb)  # (batch, emb_dim, 1, new_length)
        # token_emb = token_emb.squeeze(2).permute(0, 2, 1)  # (batch, new_length, emb_dim)

        # 空间嵌入
        token_emb = self.dg_emb(x)

        # cls_emb
        batch_size = token_emb.size(0)
        # 生成 [CLS] token 并添加到输入数据的开头
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch, 1, emb_dim)
        token_emb = torch.cat((cls_tokens, token_emb), dim=1)  # (batch, new_length + 1, emb_dim)

        # learnable pos_emb
        # seq_len = token_emb.size(1)
        # pos_ids = self.position_ids[:, :seq_len]  # (1, new_length + 1)
        # pos_emb = self.pos_emb(pos_ids).expand(batch_size, -1, -1)  # (batch, new_length + 1, emb_dim)

        pos_emb = self.pos_emb2(token_emb).expand(batch_size, -1, -1)
        return pos_emb + token_emb


class STM(nn.Module):
    def __init__(self, emb_dim, d_state, d_conv, expand, dropout=0.5):
        super(STM, self).__init__()
        self.mamba = Mamba(d_model=emb_dim, d_state=d_state, d_conv=d_conv, expand=expand)
        # 层归一化
        self.norm1 = nn.LayerNorm(emb_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Mamba 机制
        mamba_output = self.mamba(x)
        # 残差连接和层归一化
        x = x + self.dropout(mamba_output)
        x = self.norm1(x)

        return x


class SelfAttention(nn.Module):
    def __init__(self, emb_dim, num_heads):
        super(SelfAttention, self).__init__()
        self.num_heads = num_heads
        self.dim = emb_dim
        self.d_k = emb_dim // num_heads

        # Query, Key, Value 线性层
        self.query = nn.Linear(emb_dim, emb_dim)
        self.key = nn.Linear(emb_dim, emb_dim)
        self.value = nn.Linear(emb_dim, emb_dim)
        self.out = nn.Linear(emb_dim, emb_dim)

    def forward(self, x):
        batch_size, time, dim = x.size()

        # 线性变换
        Q = self.query(x)  # (batch, time, dim)
        K = self.key(x)  # (batch, time, dim)
        V = self.value(x)  # (batch, time, dim)

        # 改变形状以适应多头注意力机制
        Q = Q.view(batch_size, time, self.num_heads, self.d_k).transpose(1, 2)  # (batch, num_heads, time, d_k)
        K = K.view(batch_size, time, self.num_heads, self.d_k).transpose(1, 2)  # (batch, num_heads, time, d_k)
        V = V.view(batch_size, time, self.num_heads, self.d_k).transpose(1, 2)  # (batch, num_heads, time, d_k)

        # 缩放点积注意力机制
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(self.d_k, dtype=torch.float32))  # (batch, num_heads, time, time)
        attention_weights = F.softmax(scores, dim=-1)  # (batch, num_heads, time, time)
        attention_output = torch.matmul(attention_weights, V)  # (batch, num_heads, time, d_k)

        # 合并多头
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, time,
                                                                              dim)  # (batch, time, dim)

        # 输出线性层
        output = self.out(attention_output)  # (batch, time, dim)

        return output


class STA(nn.Module):
    def __init__(self, emb_dim, num_heads, dropout=0.5):
        super(STA, self).__init__()
        self.attention = SelfAttention(emb_dim=emb_dim, num_heads=num_heads)
        # 层归一化
        self.norm1 = nn.LayerNorm(emb_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Mamba 机制
        mamba_output = self.attention(x)
        # 残差连接和层归一化
        x = x + self.dropout(mamba_output)
        x = self.norm1(x)
        return x


class TAM(nn.Module):
    def __init__(self, emb_dim, d_state, d_conv, expand, num_layers, num_heads=8, dropout=0.5):
        super(TAM, self).__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.ModuleList([
                STA(emb_dim, num_heads, dropout),
                STM(emb_dim, d_state, d_conv, expand, dropout)
            ]))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        for attn_layer, mamba_layer in self.layers:
            x = attn_layer(x)
            x = mamba_layer(x)
        return self.dropout(x)

class DSE_TAM(nn.Module):
    def __init__(self, params):
        super(DSE_TAM, self).__init__()
        self.num_elec = params['num_electrodes']
        self.emb_dim = params['emb_dim']
        self.pool = params['pool']
        self.emb_kernel = params['emb_kernel']
        self.d_state = params['d_state']
        self.d_conv = params['d_conv']
        self.expand = params['expand']
        self.headdim = params['headdim']
        self.n_layers = params['num_layers']
        self.n_cls = params['num_classes']
        self.dropout = params['dropout']
        self.n_heads = params['num_heads']
        self.time = params['time']
        self.g_layers = params['g_layers']

        self.dse = DSE(self.num_elec, self.emb_dim, self.pool, self.emb_kernel, self.time, self.g_layers)
        self.pre_stm = STM(self.emb_dim, self.d_state, self.d_conv, self.expand, self.dropout)
        self.tam = TAM(self.emb_dim, self.d_state, self.d_conv, self.expand, self.headdim,
                                       self.n_layers, num_heads=self.n_heads, dropout=self.dropout)

        self.classifier = nn.Sequential(
            nn.Linear(self.emb_dim, self.n_cls),
            nn.Softmax(dim=1)
        )
    def forward(self, x):
        x = self.dse(x)
        x = self.pre_stm(x)
        x = self.tam(x)
        cls_embedding = x[:, 0, :].reshape(x.shape[0], -1)
        output = self.classifier(cls_embedding)
        return output
