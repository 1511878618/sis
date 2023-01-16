"model"
import sis.model.transformer as trans
from torch import nn
from sis.model.embedding import OnehotLayer
import torch.nn.functional as F
import copy
import torch

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
