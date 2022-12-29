from torch import Tensor
from torch.nn import functional as F
from torch import nn
import torch
import math
from typing import Optional, Tuple

import copy


def clones(module, N):
    "产生N个完全相同的网络层，并放于nn.ModuleList"
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def subsequent_mask(size: int) -> Tensor:
    """
    用于获取给定长度的mask矩阵。
    mask矩阵是一个上三角为True，下三角为False的方阵
    """
    attn_shape = (1, size, size)
    subsequent_mask = torch.tril(torch.ones(attn_shape))

    return subsequent_mask == 0


def attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Tensor = None,
    dropout: float = None,
) -> Tuple[Tensor, Optional[Tensor]]:
    """
    query:(batch_size, seq_len, embeding_dim)
    key:(batch_size, seq_len, embeding_dim)
    values:(batch_size, seq_len, embeding_dim)
    mask:(seq_len, seq_len) 下三角矩阵，上三角为0

    example:
        batch_size = 32
        seq_len = 20
        d_k = 8

        q = torch.randn(batch_size, seq_len, d_k)
        k = torch.randn(batch_size, seq_len, d_k)
        v = torch.randn(batch_size, seq_len, d_k)
        mask = subsequent_mask(seq_len)

        out, attn = attention(q, k, v, mask)
    """

    d_k = query.size(-1)
    # 计算scores
    scores = (query @ key.transpose(-1, -2)) / math.sqrt(d_k)
    # 是否使用mask
    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))  # mask步骤，用负无穷 替换
    p_attn = F.softmax(scores, dim=-1)
    # 是否dropout
    if dropout is not None:
        p_attn = F.dropout(p_attn, dropout)
    return p_attn @ value, p_attn


class MultiHeadAttention(nn.Module):
    r"""
    parameters:
        - n_head 头数
        - d_model embedding的维度，必须是能被n_head整除
        - dropout
    example
        batch_size = 32
        seq_len = 20
        d_model = 32
        n_head = 4

        q = torch.randn(batch_size, seq_len, d_model)
        k = torch.randn(batch_size, seq_len, d_model)
        v = torch.randn(batch_size, seq_len, d_model)
        mask = subsequent_mask(seq_len)

        mn = MultiHeadAttention(n_head, d_model)
        mn(q, k, v, mask)
    """

    def __init__(self, n_head: int, d_model: int, dropout: float = 0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_head == 0  # 保证总维度被均匀的分给每一个head
        self.d_k = d_model // n_head
        self.h = n_head
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = dropout

    def forward(
        self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor = None
    ) -> Tensor:
        if mask is not None:
            mask = mask.unsqueeze(1)  #  添加一个维度成为(1, 1, seq_len, seq_len)
        batch_size = query.size(0)
        # 1)对qkv进行一个线性变换
        query, key, value = [
            l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears, (query, key, value))
        ]
        # 2）计算注意力，返回(batch_size, n_head, seq_len, d_model)和(batch_size, n_head, seq_len, seq_len)
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        # 3)'concat'用view和最后一个线性层进行变化
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model=512, d_ff=2048, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class SublayerConnection(nn.Module):
    """
    残差连接的实现，内置layer norm 和dropout
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

        # return self.norm(x + self.dropout(sublayer(x)))


class EncoderLayer(nn.Module):
    r"""
    parameters:
        - size layerNorm所进行size，传入embedding大小即可，也即d_model
        - attn MultiHeadAttention类，传入一个定义好的MultiHeadAttention即可，使用copy.deepcopy产生一个复制作为传入
        - feed_forward PositionwiseFeedForward类，同attn
        - dropout float，指定dropout
    example:
        import transformer as trans
        import copy

        batch_size = 32
        seq_len = 20
        d_model = 32
        n_head = 4
        d_ff = 64
        dropout = 0.1

        x = torch.cat([torch.randint(1, 100, size=(batch_size, 10)), torch.zeros((batch_size, 10))],dim = -1)
        x_embed = torch.randn(batch_size, seq_len, d_model)
        mask = (x != 0).unsqueeze(-2)

        c = copy.deepcopy

        attn = trans.MultiHeadAttention(n_head, d_model)
        ff = trans.PositionwiseFeedForward(d_model, d_ff,dropout)

        encoderlayer = trans.EncoderLayer(d_model, c(attn), c(ff), dropout)
        encoderlayer(x_embed, mask) # unsqueeze(-2)，x（batch_size, seq_len） -> (batch_size, 1, seq_len) 添加一个维度表示对所有的输入都进行相应的操作，也就是会进行广播机制扩展成(bn, seq_len ,seq_len)

    """

    def __init__(
        self,
        size,
        self_attn: MultiHeadAttention,
        feed_forward: PositionwiseFeedForward,
        dropout: float,
    ):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        r"""
        Encoder采用的mask是对padding进行mask的
        """
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))  # 使用lambda传入
        return self.sublayer[1](x, self.feed_forward)


class Encoder(nn.Module):
    r"""完整的Encoder包含N次重复的EncoderLayer
    - example
        batch_size = 32
        seq_len = 20
        d_model = 32
        n_head = 4
        d_ff = 64
        dropout = 0.1
        N = 6

        x = torch.cat([torch.randint(1, 100, size=(batch_size, 10)), torch.zeros((batch_size, 10))],dim = -1)
        x_embed = torch.randn(batch_size, seq_len, d_model)
        mask = (x == 0).unsqueeze(-2)

        c = copy.deepcopy

        attn = MultiHeadAttention(n_head, d_model)
        ff = PositionwiseFeedForward(d_model, d_ff,dropout)

        encoderlayer = EncoderLayer(d_model, c(attn), c(ff), dropout)
        encoder = Encoder(c(encoderlayer), N)

        out = encoder(x_embed, mask)
        print(out.shape, x_embed.shape)
    """

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.size)

    def forward(self, x, mask):
        "每一层的输入是x和mask"
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
