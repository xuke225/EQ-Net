import warnings
import torch
import torch.nn as nn


def grad_scale(x, scale):
    y = x.abs()
    y_grad = x.abs() * scale
    return (y - y_grad).detach() + y_grad


def round_ste(x: torch.Tensor):
    """
    Implement Straight-Through Estimator for rounding operation.
    """
    return (x.round() - x).detach() + x


def lp_loss(pred, tgt, p=2.0, reduction='none'):
    """
    loss function measured in L_p Norm
    """
    if reduction == 'none':
        return (pred - tgt).abs().pow(p).sum(1).mean()
    else:
        return (pred - tgt).abs().pow(p).mean()


class UniformAffineQuantizer(nn.Module):
    def __init__(self, scale_method: str = 'mse', leaf_param: bool = False):
        super(UniformAffineQuantizer, self).__init__()
        self.scale_method = scale_method
        self.leaf_param = leaf_param

        self.n_bits = 8
        self.n_levels = 2 ** self.n_bits

        self.sym = True
        self.channel_wise = False

        self.delta_set_sym_channel = None
        self.delta_set_asym_channel = None
        self.zero_point_set_sym_channel = None
        self.zero_point_set_asym_channel = None

        self.delta_set_asym_layer = None
        self.delta_set_sym_layer = None
        self.zero_point_set_asym_layer = None
        self.zero_point_set_sym_layer = None

        self.inited = False

    def forward(self, x: torch.Tensor):
        return

    @staticmethod
    def asym_quantize(x, max, min, bit):
        delta = (max - min) / (2 ** bit -1)
        zero_point = ((-min) / delta).round()
        x_int = torch.round(x / delta)
        x_quant = torch.clamp(x_int + zero_point, 0, 2 ** bit -1)
        # x_quant = torch.clamp(x_int + zero_point, 0, self.n_levels -1)
        x_float_q = (x_quant - zero_point) * delta
        return x_float_q

    @staticmethod
    def sym_quantize(x, max, bit):
        delta = (2 * max) / (2 ** bit - 1)
        # we assume weight quantization is always signed
        # zero_point = (max / delta).round()
        zero_point = 2 ** (bit - 1) - 1
        x_int = torch.round(x / delta)
        x_quant = torch.clamp(x_int + zero_point, 0, 2 ** bit - 1)
        # x_quant = torch.clamp(x_int + zero_point, 0, self.n_levels - 1)
        x_float_q = (x_quant - zero_point) * delta
        return x_float_q

    @staticmethod
    def get_min_max_method(x, x_min, x_max, i):
        new_min, new_max = None, None
        t = 3
        if t == 0:
            new_max = x_max * (1.0 - (i * 0.01))
            new_min = x_min * (1.0 -(i * 0.01))
        elif t == 1:
            new_max = x_max
            new_min = x_min * (1.0 -(i * 0.01))
        elif t == 2:
            new_max = x_max * (1.0 - (i * 0.01))
            new_min = x_min
        elif t == 3:
            x_mean = x.mean()
            step_left = (x_mean - x_min) / 100
            step_right = (x_max - x_mean) / 100
            new_min = x_mean - step_left * i
            new_max = x_mean + step_right * i
        elif t == 4:
            raise 'error'
        return new_min, new_max

    def set_quantization_bit(self, bit: int):
        assert 2 <= bit <= 8, 'bitwidth not supported'
        self.n_bits = bit
        self.n_levels = 2 ** self.n_bits

    def set_quantization_params(self, bit, symmetric, channel_wise=None):
        self.n_bits = bit
        self.n_levels = 2 ** self.n_bits

        self.sym = symmetric
        self.channel_wise = channel_wise

    def update_zero_point(self):
        return


class UniformAffineQuantizerForWeight(UniformAffineQuantizer):
    def __init__(self, n_bits: int = 8, symmetric: bool = False, channel_wise: bool = False, scale_method: str = 'mse', leaf_param: bool = False):
        super(UniformAffineQuantizerForWeight, self).__init__(scale_method, leaf_param)

    def forward(self, x: torch.Tensor):
        if not self.inited:
            self.init_quantization_scale(x)
            self.inited = True

        if self.channel_wise:
            delta_grad_scale = 1.0 / ((self.n_levels * x.numel() / x.size(1)) ** 0.5)
            if self.sym:
                delta_set = grad_scale(self.delta_set_sym_channel[:, self.n_bits - 2], delta_grad_scale)
                zero_point_set = self.zero_point_set_sym_channel[:, self.n_bits - 2]
            else:
                delta_set = grad_scale(self.delta_set_asym_channel[:, self.n_bits - 2], delta_grad_scale)
                zero_point_set = self.zero_point_set_asym_channel[:, self.n_bits - 2]
        else:
            delta_grad_scale = 1.0 / ((self.n_levels * x.numel()) ** 0.5)
            if self.sym:
                delta_set = grad_scale(self.delta_set_sym_layer[:, self.n_bits - 2], delta_grad_scale)
                zero_point_set = self.zero_point_set_sym_layer[:, self.n_bits - 2]
            else:
                delta_set = grad_scale(self.delta_set_asym_layer[:, self.n_bits - 2], delta_grad_scale)
                zero_point_set = self.zero_point_set_asym_layer[:, self.n_bits - 2]

        delta = delta_set

        if not self.leaf_param:
            delta = float(delta)

        if len(x.shape) == 4: # for CNN weight
            delta = delta.view(-1, 1, 1, 1)
            zero_point = zero_point_set.view(-1, 1, 1, 1)
        elif len(x.shape) == 2: # for Linear weight
            delta = delta.view(-1, 1)
            zero_point = zero_point_set.view(-1, 1)
        else:
            raise 'shape error'

        x_int = round_ste(x / delta) + zero_point
        # x_int =  x / delta + self.zero_point_set[self.n_bits-2]
        x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
        x_dequant = (x_quant - zero_point) * delta
        return x_dequant

    def init_quantization_scale(self, x: torch.Tensor):
        # symmetric:
        #   per-channel:
        #       scale[channel, bit] zero_point[channel, bit]
        #   per-layer:
        #       scale[1, bit] zero_point[1, bit]
        # asymmetric:
        #   per-channel:
        #       scale[channel, bit] zero_point[channel, bit]
        #   per-layer:
        #       scale[1, bit] zero_point[1, bit]
        device = torch.device("{}".format(x.device))
        x = x.clone().detach()
        # (1) init for symmetric per-channel and asymmetric per-channel
        # symmetric parameter:
        delta_set_sym_channel = torch.zeros(size=(x.shape[0], 7)).to(device)
        zero_point_set_sym_channel = torch.zeros(size=(x.shape[0], 7)).to(device)
        # asymmetric parameter:
        delta_set_asym_channel = torch.zeros(size=(x.shape[0], 7)).to(device)
        zero_point_set_asym_channel = torch.zeros(size=(x.shape[0], 7)).to(device)

        for channel in range(x.shape[0]):
            for bit in range(2, 9):
                sym_delta, sym_zero_point = self.symmetric_init_one_channel(x[channel], bit)
                delta_set_sym_channel[channel][bit-2] = sym_delta
                zero_point_set_sym_channel[channel][bit-2] = sym_zero_point

                asym_delta, asym_zero_point = self.asymmetric_init_one_channel(x[channel], bit)
                delta_set_asym_channel[channel][bit-2] = asym_delta
                zero_point_set_asym_channel[channel][bit-2] = asym_zero_point

        # delta_set_sym_channel = delta_set_sym_channel.to(device)
        # zero_point_set_sym_channel = zero_point_set_sym_channel.to(device)
        self.delta_set_sym_channel = torch.nn.Parameter(delta_set_sym_channel)
        self.zero_point_set_sym_channel = torch.nn.Parameter(zero_point_set_sym_channel, requires_grad=False)

        self.delta_set_asym_channel = torch.nn.Parameter(delta_set_asym_channel)
        self.zero_point_set_asym_channel = torch.nn.Parameter(zero_point_set_asym_channel, requires_grad=False)

        # (2) init for symmetric per-layer and asymmetric per-layer
        # symmetric parameter:
        delta_set_sym_layer = torch.zeros(size=(1, 7)).to(device)
        zero_point_set_sym_layer = torch.zeros(size=(1, 7)).to(device)
        # asymmetric parameter:
        delta_set_asym_layer = torch.zeros(size=(1, 7)).to(device)
        zero_point_set_asym_layer = torch.zeros(size=(1, 7)).to(device)
        for bit in range(2, 9):
            sym_delta, sym_zero_point = self.symmetric_init_one_channel(x, bit)
            delta_set_sym_layer[0][bit-2] = sym_delta
            zero_point_set_sym_layer[0][bit-2] = sym_zero_point

            asym_delta, asym_zero_point = self.asymmetric_init_one_channel(x, bit)
            delta_set_asym_layer[0][bit-2] = asym_delta
            zero_point_set_asym_layer[0][bit-2] = asym_zero_point

        self.delta_set_sym_layer  = torch.nn.Parameter(delta_set_sym_layer)
        self.zero_point_set_sym_layer = torch.nn.Parameter(zero_point_set_sym_layer, requires_grad=False)

        self.delta_set_asym_layer = torch.nn.Parameter(delta_set_asym_layer)
        self.zero_point_set_asym_layer = torch.nn.Parameter(zero_point_set_asym_layer, requires_grad=False)

    def asymmetric_init_one_channel(self, x, bit):
        delta, zero_point = 1.0, 0.0
        if self.scale_method == 'max':
            x_min = min(x.min().item(), 0)
            x_max = max(x.max().item(), 0)
            if 'scale' in self.scale_method:
                x_min = x_min * (bit + 2) / 8
                x_max = x_max * (bit + 2) / 8
            x_absmax = max(abs(x_min), x_max)
            delta = float(x_max - x_min) / (2 ** bit - 1)
            zero_point = round(-x_min / delta)

        elif self.scale_method == 'mse':
            x_max = x.max()
            x_min = x.min()
            best_score = 100000
            for i in range(80):
                new_min, new_max = self.get_min_max_method(x, x_min, x_max, i)
                # new_max = x_max * (1.0 - (i * 0.01))
                # new_min = x_min * (1.0 -(i * 0.01))
                x_q = self.asym_quantize(x, new_max, new_min, bit)
                score = lp_loss(x, x_q, p=2.4, reduction='all')
                if score < best_score:
                    best_score = score
                    delta = (new_max - new_min) / (2 ** bit -1)
                    zero_point = float(((-new_min) / delta).round()) if new_min < 0 else 0
            # print("bit:{}, zero point:{}".format(bit, zero_point))
        return delta, zero_point

    def symmetric_init_one_channel(self, x, bit):
        delta, zero_point = 1.0, 0.0
        if self.scale_method == 'max':
            x_min = min(x.min().item(), 0)
            x_max = max(x.max().item(), 0)
            if 'scale' in self.scale_method:
                x_min = x_min * (bit + 2) / 8
                x_max = x_max * (bit + 2) / 8
            x_absmax = max(abs(x_min), x_max)
            delta = float(x_max - x_min) / (2 ** bit -1)
            zero_point = round(-x_min / delta)

        elif self.scale_method == 'mse':
            # we always use symmetric quantization in mse mode
            x_absmax = x.abs().max()
            x_min = x.min().item()
            best_score = 100000
            for i in range(80):
                new_max = x_absmax * (1.0 - (i * 0.01))
                x_q = self.sym_quantize(x, new_max, bit)
                # L_p norm minimization as described in LAPQ
                # https://arxiv.org/abs/1911.07190
                score = lp_loss(x, x_q, p=2.4, reduction='all')
                if score < best_score:
                    best_score = score
                    delta = (2 * new_max) / (2 ** bit - 1)
                    # zero_point = float((new_max / delta).round()) if x_min < 0 else 0.0
                    zero_point = 2 ** (bit - 1) - 1 if x_min < 0 else 0.0
                    # re-calculate the scale delta if zero-point is not 0,

        return delta, zero_point


class UniformAffineQuantizerForACT(UniformAffineQuantizer):
    def __init__(self, n_bits: int = 8, symmetric: bool = False, channel_wise: bool = False, scale_method: str = 'mse', leaf_param: bool = False):
        super(UniformAffineQuantizerForACT, self).__init__(scale_method, leaf_param)

    def forward(self, x: torch.Tensor):
        if self.inited is False:
            self.init_quantization_scale(x)
            self.inited = True

        delta_grad_scale = 1.0 / ((self.n_levels * x.numel()) ** 0.5)
        if self.sym:
            delta_set = grad_scale(self.delta_set_sym_layer[:, self.n_bits - 2], delta_grad_scale)
            zero_point_set = self.zero_point_set_sym_layer[:, self.n_bits - 2]
        else:
            delta_set = grad_scale(self.delta_set_asym_layer[:, self.n_bits - 2], delta_grad_scale)
            zero_point_set = self.zero_point_set_asym_layer[:, self.n_bits - 2]
        delta = delta_set

        if not self.leaf_param:
            delta = float(delta)

        if len(x.shape) == 4:  # CNN input
            delta = delta.view(-1, 1, 1, 1)
            zero_point = zero_point_set.view(-1, 1, 1, 1)
        elif len(x.shape) == 2:  # Linear input
            delta = delta.view(-1, 1)
            zero_point = zero_point_set.view(-1, 1)
        else:
            raise 'shape error'

        x_int = round_ste(x / delta) + zero_point
        # x_int =  x / delta + self.zero_point_set[self.n_bits-2]
        x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
        x_dequant = (x_quant - zero_point) * delta
        return x_dequant

    def init_quantization_scale(self, x: torch.Tensor):
        device = torch.device("{}".format(x.device))
        x = x.clone().detach()

        delta_set_sym = torch.ones(size=(1, 7)).to(device)
        zero_point_set_sym = torch.ones(size=(1, 7)).to(device)

        delta_set_asym = torch.ones(size=(1, 7)).to(device)
        zero_point_set_asym = torch.ones(size=(1, 7)).to(device)

        for bit in range(2, 9):
            delta_sym, zero_point_sym = self.symmetric_init_one_channel(x, bit)
            delta_set_sym[0][bit - 2] = delta_sym
            zero_point_set_sym[0][bit - 2] = zero_point_sym

            delta_for_asym, zero_point_for_asym = self.asymmetric_init_one_channel(x ,bit)
            delta_set_asym[0][bit - 2] = delta_for_asym
            zero_point_set_asym[0][bit - 2] = zero_point_for_asym

        self.delta_set_sym_layer = torch.nn.Parameter(delta_set_sym)
        self.zero_point_set_sym_layer = torch.nn.Parameter(zero_point_set_sym, requires_grad=False)

        self.delta_set_asym_layer = torch.nn.Parameter(delta_set_asym)
        self.zero_point_set_asym_layer = torch.nn.Parameter(zero_point_set_asym, requires_grad=False)

    def asymmetric_init_one_channel(self, x, bit):
        delta, zero_point = 1.0, 0.0
        if self.scale_method == 'max':
            x_min = min(x.min().item(), 0)
            x_max = max(x.max().item(), 0)
            if 'scale' in self.scale_method:
                x_min = x_min * (bit + 2) / 8
                x_max = x_max * (bit + 2) / 8
            x_absmax = max(abs(x_min), x_max)
            delta = float(x_max - x_min) / (2 ** bit - 1)
            zero_point = round(-x_min / delta)

        elif self.scale_method == 'mse':
            x_max = x.max()
            x_min = x.min()
            best_score = 100000
            for i in range(80):
                new_min, new_max = self.get_min_max_method(x, x_min, x_max, i)
                # new_max = x_max * (1.0 - (i * 0.01))
                # new_min = x_min * (1.0 -(i * 0.01))
                x_q = self.asym_quantize(x, new_max, new_min, bit)
                score = lp_loss(x, x_q, p=2.4, reduction='all')
                if score < best_score:
                    best_score = score
                    delta = (new_max - new_min) / (2 ** bit - 1)
                    zero_point = float(((-new_min) / delta).round()) if new_min < 0 else 0
            # print("bit:{}, zero point:{}".format(bit, zero_point))
        return delta, zero_point

    def symmetric_init_one_channel(self, x, bit):
        delta, zero_point = 1.0, 0.0
        if self.scale_method == 'max':
            x_min = min(x.min().item(), 0)
            x_max = max(x.max().item(), 0)
            if 'scale' in self.scale_method:
                x_min = x_min * (bit + 2) / 8
                x_max = x_max * (bit + 2) / 8
            x_absmax = max(abs(x_min), x_max)
            delta = float(x_max - x_min) / (2 ** bit - 1)
            zero_point = round(-x_min / delta)

        elif self.scale_method == 'mse':
            # we always use symmetric quantization in mse mode
            x_absmax = x.abs().max()
            x_min = x.min().item()
            best_score = 100000
            for i in range(80):
                new_max = x_absmax * (1.0 - (i * 0.01))
                x_q = self.sym_quantize(x, new_max, bit)
                # L_p norm minimization as described in LAPQ
                # https://arxiv.org/abs/1911.07190
                score = lp_loss(x, x_q, p=2.4, reduction='all')
                if score < best_score:
                    best_score = score
                    delta = (2 * new_max) / (2 ** bit - 1)
                    # zero_point = float((new_max / delta).round()) if x_min < 0 else 0.0
                    zero_point = 2 ** (bit - 1) - 1 if x_min < 0 else 0.0
                    # re-calculate the scale delta if zero-point is not 0,

        return delta, zero_point

# quantizer = UniformAffineQuantizerForWeight(scale_method='mse', leaf_param=True, channel_wise=True)
#
# cali_input = torch.randn(size=(2, 3, 3, 3))
# cali_output = quantizer(cali_input)
#
# input_x = torch.randn(size=(2, 3, 3, 3))
# quantizer.set_quantization_params(refactored_bit=8, symmetric=False, channel_wise=False)
# output_1 = quantizer(input_x)
#
# quantizer.set_quantization_params(refactored_bit=8, symmetric=True, channel_wise=True)
# output_2 = quantizer(input_x)
#
# quantizer.set_quantization_params(refactored_bit=8, symmetric=True, channel_wise=False)
# output_3 = quantizer(input_x)
#
# print(input_x)
