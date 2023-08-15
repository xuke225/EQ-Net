import warnings
import torch
import torch.nn as nn
from core.config import cfg


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
    elif reduction == 'all':
        return (pred - tgt).abs().pow(p).mean()
    else:
        return torch.mean((pred - tgt).abs().pow(p), dim=1, keepdim=True)


class LSQQuantizer(nn.Module):
    """
    :param n_bits: number of bit for quantization
    :param symmetric: if True, the zero_point should always be 0
    :param channel_wise: if True, compute scale and zero_point in each channel
    :param scale_method: determines the quantization scale and zero point
    """

    def __init__(self, n_bits: int = 8, symmetric: bool = False, channel_wise: bool = False, scale_method: str = 'max',
                 leaf_param: bool = False):
        super(LSQQuantizer, self).__init__()
        self.sym = symmetric
        self.channel_wise = channel_wise
        self.scale_method = scale_method

        self.n_bits = n_bits
        self.pos_thd = None
        self.neg_thd = None
        self.leaf_param = leaf_param

        self.init_number = 1
        self.init_state = 0
        self.all_positive = False

    def forward(self, x: torch.Tensor):
        return

    def sym_quantize(self, x, max, min):
        scale = (max - min) / (self.pos_thd - self.neg_thd)

        beta = 0.0
        x = (x - beta) / scale
        x = torch.clamp(x, self.neg_thd, self.pos_thd)
        x_quant = torch.round(x)
        x_float_q = x_quant * scale + beta
        return x_float_q

    def symmetric_init(self, x, bit):
        self.set_quantization_bit(bit=bit)
        scale, beta = 1.0, 0.0
        x_absmax = x.abs().max()
        best_score = 100000
        if self.all_positive:
            for i in range(80):
                new_max = x_absmax * (1.0 - (i * 0.01))
                x_q = self.sym_quantize(x, new_max, 0)
                score = lp_loss(x, x_q, p=2.4, reduction='all')
                if score < best_score:
                    best_score = score
                    scale = new_max / (self.pos_thd - self.neg_thd)
                    beta = 0.0
        else:
            for i in range(80):
                new_max = x_absmax * (1.0 - (i * 0.01))
                x_q = self.sym_quantize(x, new_max, -new_max)
                score = lp_loss(x, x_q, p=2.4, reduction='all')
                if score < best_score:
                    best_score = score
                    scale = (2 * new_max) / (self.pos_thd - self.neg_thd)
                    beta = 0.0
        return scale, beta

    def asymmetric_init(self, x, bit):
        self.set_quantization_bit(bit=bit)
        scale, beta = 1.0, 0.0
        x_max = x.max()
        x_min = x.min()
        best_score = 100000
        for i in range(80):
            new_max = x_max * (1.0 - (i * 0.01))
            new_min = x_min * (1.0 - (i * 0.01))
            x_q = self.asym_quantize(x, new_max, new_min)
            score = lp_loss(x, x_q, p=2.4, reduction='all')
            if score < best_score:
                best_score = score
                scale = (new_max - new_min) / (self.pos_thd - self.neg_thd)
                beta = new_min - scale * self.neg_thd
                if self.all_positive:
                    beta = -1e-9
        return scale, beta

    def asym_quantize(self, x, max, min):
        scale = (max - min) / (self.pos_thd - self.neg_thd)
        beta = min - scale * self.neg_thd
        x = (x - beta) / scale
        x = torch.clamp(x, self.neg_thd, self.pos_thd)
        x_quant = torch.round(x)
        x_float_q = x_quant * scale + beta
        return x_float_q

    def set_quantization_params(self, bit=8, symmetric=False, channel_wise=False):
        self.n_bits = bit
        self.sym = symmetric
        self.channel_wise = channel_wise
        if self.all_positive:
            self.pos_thd = 2 ** self.n_bits - 1
            self.neg_thd = 0
        else:
            self.pos_thd = 2 ** (self.n_bits - 1) - 1
            self.neg_thd = - 2 ** (self.n_bits - 1)

    def set_quantization_bit(self, bit: int):
        self.n_bits = bit
        if self.all_positive: # ReLU
            self.pos_thd = 2 ** self.n_bits - 1
            self.neg_thd = 0
        else:
            self.pos_thd = 2 ** (self.n_bits - 1) - 1
            self.neg_thd = - 2 ** (self.n_bits - 1)


class LSQQuantizerForACT(LSQQuantizer):
    def __init__(self, n_bits=8, symmetric=True, channel_wise=False, scale_method='mse', leaf_param=False,
                 layer_number=100):
        super(LSQQuantizerForACT, self).__init__(n_bits=n_bits, symmetric=symmetric, channel_wise=channel_wise,
                                                 scale_method=scale_method, leaf_param=leaf_param)
        self.beta_set_sym = None
        self.beta_set_asym = None
        self.scale_set_asym = None
        self.scale_set_sym = None
        self.layer_number = layer_number

    def forward(self, x: torch.Tensor):

        if self.init_state < self.init_number:
            original_channel_wise = self.channel_wise
            original_sym = self.sym
            original_bit = self.n_bits

            self.init_quantization_param(x)
            self.init_state += 1

            self.set_quantization_params(bit=original_bit, symmetric=original_sym, channel_wise=original_channel_wise)

        s_grad_scale = 1.0 / ((self.pos_thd * x.numel()) ** 0.5)

        if self.sym:
            s_scale = grad_scale(self.scale_set_sym[:, self.n_bits - 2], s_grad_scale)
            beta = self.beta_set_sym[:, self.n_bits - 2]
        else:
            s_scale = grad_scale(self.scale_set_asym[:, self.n_bits - 2], s_grad_scale)
            beta = grad_scale(self.beta_set_asym[:, self.n_bits - 2], s_grad_scale)

        x = (x - beta) / (s_scale + 1e-6)

        x = torch.clamp(x, self.neg_thd, self.pos_thd)
        x_quant = round_ste(x)
        x_dequant = x_quant * (s_scale + 1e-6) + beta
        return x_dequant

    def init_quantization_param(self, x: torch.Tensor):
        device = torch.device("{}".format(x.device))
        x = x.clone().detach()
        if self.init_state == 0:

            self.scale_set_sym = torch.nn.Parameter(torch.ones(size=(1, 7)).to(device), requires_grad=True)
            self.beta_set_sym = torch.nn.Parameter(torch.zeros(size=(1, 7)).to(device), requires_grad=False)  # 0

            self.scale_set_asym = torch.nn.Parameter(torch.ones(size=(1, 7)).to(device), requires_grad=True)
            self.beta_set_asym = torch.nn.Parameter(torch.zeros(size=(1, 7)).to(device), requires_grad=True)

            if torch.min(x) >= -1e-9:
                self.all_positive = True

            if cfg.Resume is not None:
                return

            for bit in range(2, 9):
                scale_sym, _ = self.symmetric_init(x, bit)
                self.scale_set_sym.data[0][bit - 2] = scale_sym

                scale_asym, beta_asym = self.asymmetric_init(x, bit)
                self.scale_set_asym.data[0][bit - 2] = scale_asym
                self.beta_set_asym[0][bit - 2] = beta_asym

        elif self.init_state < self.init_number:
            return


class LSQQuantizerForWeight(LSQQuantizer):
    def __init__(self, n_bits=8, symmetric=False, channel_wise=True, scale_method='mse', leaf_param=False,
                 layer_number=100):
        super(LSQQuantizerForWeight, self).__init__(n_bits=n_bits, symmetric=symmetric, channel_wise=channel_wise,
                                                    scale_method=scale_method, leaf_param=leaf_param)
        self.layer_number = layer_number
        self.beta_set_sym_layer = None
        self.beta_set_asym_layer = None
        self.scale_set_asym_layer = None
        self.scale_set_sym_layer = None

        self.beta_set_sym_channel = None
        self.beta_set_asym_channel = None
        self.scale_set_asym_channel = None
        self.scale_set_sym_channel = None
        self.all_positive = False

    def forward(self, x: torch.Tensor):

        if self.init_state < self.init_number:
            original_channel_wise = self.channel_wise
            original_sym = self.sym
            original_bit = self.n_bits
            self.init_quantization_param(x)
            self.init_state += 1
            self.set_quantization_params(bit=original_bit, symmetric=original_sym, channel_wise=original_channel_wise)

        # start quantization
        if self.channel_wise:
            s_grad_scale = 2.0 / ((self.pos_thd * x.numel() / x.size(1)) ** 0.5)
            if self.sym:
                s_scale = grad_scale(self.scale_set_sym_channel[:, self.n_bits - 2], s_grad_scale)
                beta = self.beta_set_sym_channel[:, self.n_bits - 2]
            else:
                s_scale = grad_scale(self.scale_set_asym_channel[:, self.n_bits - 2], s_grad_scale)
                beta = grad_scale(self.beta_set_asym_channel[:, self.n_bits - 2], s_grad_scale)

            # s_scale = torch.max(s_scale)
            # print(s_scale.size())
        else:
            s_grad_scale = 1.0 / ((self.pos_thd * x.numel()) ** 0.5)
            if self.sym:
                s_scale = grad_scale(self.scale_set_sym_layer[:, self.n_bits - 2], s_grad_scale)
                beta = self.beta_set_sym_layer[:, self.n_bits - 2]
            else:
                s_scale = grad_scale(self.scale_set_asym_layer[:, self.n_bits - 2], s_grad_scale)
                beta = grad_scale(self.beta_set_asym_layer[:, self.n_bits - 2], s_grad_scale)

        if len(x.shape) == 4:  # for CNN weight
            s_scale = s_scale.view(-1, 1, 1, 1)
            beta = beta.view(-1, 1, 1, 1)
        elif len(x.shape) == 2:  # for Linear weight
            s_scale = s_scale.view(-1, 1)
            beta = beta.view(-1, 1)
        else:
            raise 'shape error'

        x = (x - beta) / (s_scale + 1e-6)

        x = torch.clamp(x, self.neg_thd, self.pos_thd)
        x_quant = round_ste(x)
        x_dequant = x_quant * (s_scale + 1e-6) + beta
        return x_dequant

    def init_quantization_param(self, x: torch.Tensor):
        device = torch.device("{}".format(x.device))
        x = x.clone().detach()

        if self.init_state == 0:

            self.scale_set_sym_channel = torch.nn.Parameter(torch.ones(size=(x.shape[0], 7)).to(device),
                                                            requires_grad=True)
            self.beta_set_sym_channel = torch.nn.Parameter(torch.zeros(size=(x.shape[0], 7)).to(device),
                                                           requires_grad=False)  # 0
            self.scale_set_asym_channel = torch.nn.Parameter(torch.ones(size=(x.shape[0], 7)).to(device),
                                                             requires_grad=True)
            self.beta_set_asym_channel = torch.nn.Parameter(torch.zeros(size=(x.shape[0], 7)).to(device),
                                                            requires_grad=True)

            self.scale_set_sym_layer = torch.nn.Parameter(torch.ones(size=(1, 7)).to(device), requires_grad=True)
            self.beta_set_sym_layer = torch.nn.Parameter(torch.zeros(size=(1, 7)).to(device), requires_grad=False)  # 0
            self.scale_set_asym_layer = torch.nn.Parameter(torch.ones(size=(1, 7)).to(device), requires_grad=True)
            self.beta_set_asym_layer = torch.nn.Parameter(torch.zeros(size=(1, 7)).to(device), requires_grad=True)

            if cfg.Resume is not None:
                return

            for bit in range(2, 9):
                # (1) init for symmetric per-channel and asymmetric per-channel
                scale_sym, beta_sym = self.symmetric_init_for_per_channel(x, bit)
                self.scale_set_sym_channel.data[:, bit - 2] = scale_sym
                self.beta_set_sym_channel.data[:, bit - 2] = beta_sym

                scale_asym, beta_asym = self.asymmetric_init_for_per_channel(x, bit)
                self.scale_set_asym_channel.data[:, bit - 2] = scale_asym
                self.beta_set_asym_channel.data[:, bit - 2] = beta_asym

                # (2) init for symmetric per-layer and asymmetric per-layer
                scale_sym, beta_sym = self.symmetric_init(x, bit)
                self.scale_set_sym_layer.data[0][bit - 2] = scale_sym
                self.beta_set_sym_layer.data[0][bit - 2] = beta_sym

                scale_asym, beta_asym = self.asymmetric_init(x, bit)
                self.scale_set_asym_layer.data[0][bit - 2] = scale_asym
                self.beta_set_asym_layer.data[0][bit - 2] = beta_asym

        elif self.init_state < self.init_number:
            return

    def symmetric_init_for_per_channel(self, w, bit):
        self.set_quantization_bit(bit=bit)

        w = w.view(w.size(0), -1)
        w_max, _ = torch.max(w.abs(), dim=1, keepdim=True)
        best_score = torch.full_like(w_max, 100.0)
        scale = torch.full_like(w_max, 1.0)
        beta = torch.full_like(w_max, 0.0)
        for i in range(5):
            new_w_max = w_max * (1.0 - (i * 0.01))
            new_w_min = -new_w_max
            w_quant = self.sym_quantize(w, new_w_max, new_w_min)
            score = lp_loss(w_quant, w, p=2.4, reduction='other')

            update_list = score < best_score
            best_score = torch.where(update_list, score, best_score)
            cur_scale = (new_w_max - new_w_min) / (self.pos_thd - self.neg_thd)
            scale = torch.where(update_list, cur_scale, scale)
        return scale.squeeze(), beta.squeeze()

    def asymmetric_init_for_per_channel(self, w, bit):
        self.set_quantization_bit(bit=bit)
        w = w.view(w.size(0), -1)
        w_max, _ = torch.max(w, dim=1, keepdim=True)
        w_min, _ = torch.min(w, dim=1, keepdim=True)
        best_score = torch.full_like(w_max, 100.0)
        scale = torch.full_like(w_max, 1.0)
        beta = torch.full_like(w_max, 0.0)
        for i in range(80):
            new_w_max = w_max * (1.0 - (i * 0.01))
            new_w_min = w_min * (1.0 - (i * 0.01))
            w_quant = self.asym_quantize(w, new_w_max, new_w_min)
            score = lp_loss(w_quant, w, p=2.4, reduction='other')

            update_list = score < best_score
            best_score = torch.where(update_list, score, best_score)

            cur_scale = (new_w_max - new_w_min) / (self.pos_thd - self.neg_thd)
            cur_beta = new_w_min - cur_scale * self.neg_thd

            scale = torch.where(update_list, cur_scale, scale)
            beta = torch.where(update_list, cur_beta, beta)
        return scale.squeeze(), beta.squeeze()

# # # input_x = torch.randn(size=(2, 3, 3, 3))
# input_x = torch.tensor([[[[1.4760, 1.4643, 0.7703],
#                           [-0.0752, -0.0676, 1.6004],
#                           [1.1702, 1.1717, 0.9130]],
#
#                          [[-1.4503, -0.3951, -0.7190],
#                           [0.7794, -1.2522, -0.5436],
#                           [-0.4576, 0.2754, 0.6507]],
#
#                          [[-0.3634, -2.1480, 0.0703],
#                           [-0.0784, 0.2409, -0.4890],
#                           [0.0821, 0.0196, -0.1180]]],
#
#                         [[[1.7081, -0.1882, -0.1476],
#                           [0.3916, -1.0820, 0.1741],
#                           [0.1220, -2.4713, 0.2306]],
#
#                          [[-0.9070, -0.2448, 1.7288],
#                           [-0.9307, -1.3349, 0.2605],
#                           [0.0640, -0.9338, -0.2198]],
#
#                          [[0.0920, 0.8888, 0.9612],
#                           [0.9745, 1.2338, 1.1588],
#                           [0.1100, 0.5644, -1.0955]]]])
#
# # print(input_x)
# # input_x = torch.randint(low=1, high=100, size=(2, 3, 3, 3))
# # quantizer = LSQQuantizerForACT(scale_method='mse', leaf_param=True, channel_wise=False)
# quantizer2 = LSQQuantizerForWeight(scale_method='mse', leaf_param=True, channel_wise=False)
# # quantizer(input_x)
# quantizer2(input_x)
#
# bit = 2
# # quantizer.set_quantization_params(bit=bit, symmetric=False, channel_wise=True)
# # output_1 = quantizer(input_x)
# # loss1 = lp_loss(output_1, input_x)
# #
# # quantizer.set_quantization_params(bit=bit, symmetric=True, channel_wise=True)
# # output_2 = quantizer(input_x)
# # loss2 = lp_loss(output_2, input_x)
#
# quantizer2.set_quantization_params(bit=bit, symmetric=False, channel_wise=False)
# output_3 = quantizer2(input_x)
# loss3 = lp_loss(output_3, input_x)
#
# quantizer2.set_quantization_params(bit=bit, symmetric=True, channel_wise=True)
# output_4 = quantizer2(input_x)
# loss4 = lp_loss(output_4, input_x)
#
# loss5 = lp_loss(input_x, input_x)
#
# print(input_x)
