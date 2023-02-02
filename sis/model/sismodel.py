"model"
import sis.model.transformer as trans
from torch import nn

import copy
import torch

from sis.utils import torch_mean

c = copy.deepcopy
__author__ = "Tingfeng Xu"


class ConvModel(nn.Module):
    def __init__(self, EmbeddingLayer, seq_length, d_conv=32, kernel_size=4, stride=1):
        super(ConvModel, self).__init__()
        self.EmbeddingLayer = EmbeddingLayer
        self.seq_length = seq_length
        self.d_model = EmbeddingLayer.d_model
        self.d_conv = d_conv
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv_block = nn.Conv1d(
            self.d_model, self.d_conv, self.kernel_size, self.stride, padding="same"
        )

        self.fc1 = nn.Linear(self.d_conv, 1)
        self.fc2 = nn.Linear(self.seq_length, 1)

    def forward(self, x, return_scores=False):
        x_SLF = self.EmbeddingLayer(x["SLF_Seq_token"])

        x_SRnase = self.EmbeddingLayer(x["SRnase_Seq_token"])

        x_total = torch.concat([x_SLF, x_SRnase], dim=1).permute(0, 2, 1)
        # (batch, d_model, length)
        assert len(x_SLF.shape) == 3
        assert len(x_SRnase.shape) == 3

        x_total = torch.relu(self.conv_block(x_total)).permute(0, 2, 1)
        o = torch.relu(self.fc1(x_total)).flatten(1)
        last_mask = torch.concat([x["SLF_Seq_mask"], x["SRnase_Seq_mask"]], dim=1)

        scores = o.masked_fill(last_mask, 1e-9)
        if return_scores:
            self.scores = scores

        o = torch.sigmoid(self.fc2(scores))

        return o


class LinearModel(nn.Module):
    def __init__(self, EmbeddingLayer, seq_length):
        super(LinearModel, self).__init__()
        self.EmbeddingLayer = EmbeddingLayer
        self.seq_length = seq_length
        self.d_model = EmbeddingLayer.d_model
        self.fc1 = nn.Linear(self.d_model, 1)
        self.fc2 = nn.Linear(self.seq_length, 1)

    def forward(self, x, return_scores=False):
        x_SLF = self.EmbeddingLayer(x["SLF_Seq_token"])
        x_SRnase = self.EmbeddingLayer(x["SRnase_Seq_token"])

        x_SLF = torch.relu(self.fc1(x_SLF))
        x_SRnase = torch.relu(self.fc1(x_SRnase))

        o = torch.concat([x_SLF, x_SRnase], dim=1).squeeze(-1)

        last_mask = torch.concat([x["SLF_Seq_mask"], x["SRnase_Seq_mask"]], dim=1)

        scores = o.masked_fill(last_mask, 1e-9)
        if return_scores:
            self.scores = scores

        o = torch.sigmoid(self.fc2(scores))
        return o


class DoubleTransformerModel(nn.Module):
    def __init__(self, N, d_ff, dropout, seq_length, EmbeddingLayer):
        super(DoubleTransformerModel, self).__init__()

        self.N = N
        self.d_model = EmbeddingLayer.d_model
        self.d_ff = d_ff
        self.dropout = dropout
        self.seq_length = seq_length
        # embedding layer
        self.EmbeddingLayer = EmbeddingLayer
        # basic layer
        attn = trans.MultiHeadAttention(1, self.d_model)  # head as 1
        ff = trans.PositionwiseFeedForward(self.d_model, d_ff, dropout)
        encoderlayer = trans.EncoderLayer(self.d_model, c(attn), c(ff), dropout)
        # transformer encoder build
        self.encoder_SLF = trans.Encoder(c(encoderlayer), N)
        self.encoder_SRnase = trans.Encoder(c(encoderlayer), N)
        # fc layer
        self.fc1 = nn.Linear(self.d_model, 1)
        self.fc2 = nn.Linear(seq_length, 1)

    def forward(self, x, return_scores=False):
        x_SLF = self.EmbeddingLayer(x["SLF_Seq_token"])
        x_SRnase = self.EmbeddingLayer(x["SRnase_Seq_token"])

        x_SLF = torch.relu(self.fc1(self.encoder_SLF(x_SLF, None)))
        x_SRnase = torch.relu(self.fc1(self.encoder_SRnase(x_SRnase, None)))

        o = torch.concat([x_SLF, x_SRnase], dim=1).squeeze(-1)

        last_mask = torch.concat([x["SLF_Seq_mask"], x["SRnase_Seq_mask"]], dim=1)

        scores = o.masked_fill(last_mask, 1e-9)
        if return_scores:
            self.scores = scores

        o = torch.sigmoid(self.fc2(scores))
        return o


class Regression(nn.Module):
    def __init__(self, EmbeddingLayer) -> None:
        super().__init__()
        self.EmbeddingLayer = EmbeddingLayer
        self.d_model = EmbeddingLayer.d_model

        self.fc = nn.Linear(self.d_model * 2, 1)  # SLF + SRnase

    def forward(self, x, return_scores=False):
        x_SLF = self.EmbeddingLayer(x["SLF_Seq_token"])
        x_SLF_select = x["SLF_Seq_mask"]
        x_SLF = torch_mean(x_SLF, x_SLF_select)

        x_SRnase = self.EmbeddingLayer(x["SRnase_Seq_token"])
        x_SRnase_select = x["SRnase_Seq_mask"]
        x_SRnase = torch_mean(x_SRnase, x_SRnase_select)

        assert x_SLF.shape[-1] == self.d_model
        assert x_SRnase.shape[-1] == self.d_model

        o = torch.concat([x_SLF, x_SRnase], dim=-1)
        if return_scores:
            self.scores = o

        o = torch.sigmoid(self.fc(o))
        return o


# class regression(nn.Module):
#     def __init__(self, in_features, bias=True, device=None, dtype=None):
#         factory_kwargs = {"device": device, "dtype": dtype}
#         super(regression, self).__init__()

#         self.in_features = in_features

#         self.weight = Parameter(torch.empty((in_features, 1), **factory_kwargs))
#         if bias:
#             self.bias = Parameter(torch.empty(1, **factory_kwargs))
#         else:
#             self.register_parameter("bias", None)
#         self.reset_parameters()

#     def forward(self, x):
#         return x @ self.weight + self.bias

#     def reset_parameters(self):
#         init.kaiming_uniform_(self.weight, a=math.sqrt(5))
#         if self.bias is not None:
#             fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
#             bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
#             init.uniform_(self.bias, -bound, bound)
