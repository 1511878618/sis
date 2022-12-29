"model"
import sis.model.transformer as trans
from torch import nn
from sis.model.embedding import OnehotLayer
import torch.nn.functional as F
import copy
import torch

c = copy.deepcopy
__author__ = "Tingfeng Xu"


class DoubleTransformerModel(nn.Module):
    def __init__(self, N, d_model, d_ff, dropout, seq_length):
        super(DoubleTransformerModel, self).__init__()

        self.N = N
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = dropout
        self.seq_length = seq_length
        # embedding layer
        self.onehot = OnehotLayer(d_model)
        # basic layer
        attn = trans.MultiHeadAttention(1, d_model)  # head as 1
        ff = trans.PositionwiseFeedForward(d_model, d_ff, dropout)
        encoderlayer = trans.EncoderLayer(d_model, c(attn), c(ff), dropout)
        # transformer encoder build
        self.encoder_SLF = trans.Encoder(c(encoderlayer), N)
        self.encoder_SRnase = trans.Encoder(c(encoderlayer), N)
        # fc layer
        self.fc1 = nn.Linear(d_model, 1)
        self.fc2 = nn.Linear(seq_length, 1)

    def forward(self, x):
        x_SLF = self.onehot(x["SLF_Seq_token"])
        x_SRnase = self.onehot(x["SRnase_Seq_token"])

        x_SLF = self.fc1(self.encoder_SLF(x_SLF, None))
        x_SRnase = self.fc1(self.encoder_SRnase(x_SRnase, None))

        o = F.relu(torch.concat([x_SLF, x_SRnase], dim=1).squeeze(-1))

        last_mask = torch.concat([x["SLF_Seq_mask"], x["SRnase_Seq_mask"]], dim=1)
        o = torch.sigmoid(self.fc2(o.masked_fill(last_mask, 1e-9)))

        return o
