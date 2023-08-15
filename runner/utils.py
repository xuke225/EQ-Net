import copy
import math
import os

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from module.qat_layer import QATQuantModule
# from module.qat_model import QATQuantModel
from module.qat_model import QATQuantSuperModel
from core.config import cfg


class CosineDecay(object):
    def __init__(self,
                 max_value,
                 min_value,
                 num_loops):
        self._max_value = max_value
        self._min_value = min_value
        self._num_loops = num_loops

    def get_value(self, i):
        if i < 0:
            i = 0
        if i >= self._num_loops:
            i = self._num_loops
        value = (math.cos(i * math.pi / self._num_loops) + 1.0) * 0.5
        value = value * (self._max_value - self._min_value) + self._min_value
        return value


class LinearDecay(object):
    def __init__(self,
                 max_value,
                 min_value,
                 num_loops):
        self._max_value = max_value
        self._min_value = min_value
        self._num_loops = num_loops

    def get_value(self, i):
        if i < 0:
            i = 0
        if i >= self._num_loops:
            i = self._num_loops - 1

        value = (self._max_value - self._min_value) / self._num_loops
        value = i * (-value)

        return value


def Adaptive_BN(model, train_loader, num_samples=256):
    model.train()
    with torch.no_grad():
        for batch, (inputs, targets) in enumerate(train_loader):
            if batch <= num_samples:
                inputs, targets = inputs.cuda(), targets.cuda()
                _ = model(inputs)
            else:
                break
    return model


def get_train_samples(train_loader, num_samples):
    train_data = []
    for batch in train_loader:
        train_data.append(batch[0])
        if len(train_data) * batch[0].size(0) >= num_samples:
            break
    return torch.cat(train_data, dim=0)[:num_samples]


def convert_to_QuantSuperModel(model, wq_params=None, aq_params=None, quantizer='lsq', search_space=None):
    model = QATQuantSuperModel(model=copy.deepcopy(model), weight_quant_params=wq_params,
                               act_quant_params=aq_params, quantizer=quantizer, search_space=search_space)

    return model

