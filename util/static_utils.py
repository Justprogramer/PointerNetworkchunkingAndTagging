# -*-coding:utf-8-*-


def to_var(x):
    from torch.autograd import Variable
    return Variable(x)


def init_embedding(input_embedding, seed=1337):
    import torch
    import numpy as np
    import torch.nn as nn
    """初始化embedding层权重
    """
    torch.manual_seed(seed)
    scope = np.sqrt(3.0 / input_embedding.size(1))
    nn.init.uniform_(input_embedding, -scope, scope)
