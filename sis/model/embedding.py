""" Define the Embedding """
import torch.nn as nn
import torch
import torch.nn.functional as F


__author__ = "Tingfeng Xu"


class OnehotLayer(nn.Module):
    def __init__(self, num_classes):
        super(OnehotLayer, self).__init__()
        self.num_classes = num_classes

    def forward(self, x):
        return F.one_hot(x, self.num_classes).float()
