import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union
from .base_uaq import UniformAffineQuantizerForWeight, UniformAffineQuantizerForACT
from .base_lsq import LSQQuantizerForACT, LSQQuantizerForWeight


class StraightThrough(nn.Module):
    def __init__(self, channel_num: int = 1):
        super().__init__()

    def forward(self, input):
        return input


class QATQuantModule(nn.Module):
    """
    Quantized Module that can perform quantized convolution or normal convolution.
    To activate quantization, please use set_quant_state function.
    """

    def __init__(self, org_module: Union[nn.Conv2d, nn.Linear], weight_quant_params,
                 act_quant_params, disable_act_quant: bool = False, quantizer: str = 'lsq'):
        super(QATQuantModule, self).__init__()
        if isinstance(org_module, nn.Conv2d):
            self.fwd_kwargs = dict(stride=org_module.stride, padding=org_module.padding,
                                   dilation=org_module.dilation, groups=org_module.groups)
            self.fwd_func = F.conv2d
        else:
            self.fwd_kwargs = dict()
            self.fwd_func = F.linear
        self.weight = org_module.weight
        self.org_weight = org_module.weight.data.clone()
        if org_module.bias is not None:
            self.bias = org_module.bias
            self.org_bias = org_module.bias.data.clone()
        else:
            self.bias = None
            self.org_bias = None
        # de-activate the quantized forward default
        self.use_weight_quant = False
        self.use_act_quant = False
        self.disable_act_quant = disable_act_quant
        # initialize quantizer
        if quantizer == 'lsq':
            self.weight_quantizer = LSQQuantizerForWeight(**weight_quant_params)
            self.act_quantizer = LSQQuantizerForACT(**act_quant_params)
        elif quantizer == 'uaq':
            self.weight_quantizer = UniformAffineQuantizerForWeight(**weight_quant_params)
            self.act_quantizer = UniformAffineQuantizerForACT(**act_quant_params)
        else:
            raise NotImplementedError

        self.activation_function = StraightThrough()
        self.ignore_reconstruction = False

    def forward(self, input: torch.Tensor):
        if self.use_weight_quant:
            weight = self.weight_quantizer(self.weight)
            bias = self.bias
        else:
            weight = self.org_weight
            bias = self.org_bias
        if self.use_act_quant:
            input = self.act_quantizer(input)
        out = self.fwd_func(input, weight, bias, **self.fwd_kwargs)
        # disable act quantization is designed for convolution before elemental-wise operation,
        # in that case, we apply activation function and quantization after ele-wise op.
        if self.disable_act_quant:
            return out
        # out = self.activation_function(out)
        return out

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant

    def set_quantization_params(self, weight_quantization_params, act_quantization_params):
        if weight_quantization_params is not None:
            self.weight_quantizer.set_quantization_params(**weight_quantization_params)
        if act_quantization_params is not None:
            self.act_quantizer.set_quantization_params(**act_quantization_params)

    def set_quantization_bit(self, weight_bit, act_bit):
        self.weight_quantizer.set_quantization_bit(weight_bit)
        self.act_quantizer.set_quantization_bit(act_bit)
