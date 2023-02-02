import os

import torch


def modelParametersNum(model):
    totalNum = sum([i.numel() for i in model.parameters()])
    print(f"模型总参数个数：{sum([i.numel() for i in model.parameters()])}")
    return totalNum


def mkdirs(path):
    try:
        os.makedirs(path)
    except:
        pass


def try_gpu():
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
    return device


class Accumulator:
    """在n个变量上累加"""

    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def changeStrByPos(string: str, idx, new):
    """
    更改一个字符串的某个位置的元素
    """
    tmp = list(string)
    tmp[idx] = str(new)
    return "".join(tmp)


def torch_mean(x, select):
    # True 会被考虑，False会被丢掉
    # select 和x维度应该数量一直，保证可以广播机制
    # x = (batch, N1, N2..) select = (batch, N1, ...) bool matrix

    select = select.unsqueeze(-1)
    select = ~select
    x = x.masked_fill(select, 1e-9).sum(dim=1)

    count = select.int().sum(1)

    return x / count
