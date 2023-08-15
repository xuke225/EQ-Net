
"""Loss functions."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
# from core.config import cfg
from functools import reduce
from collections import Counter
from torch.autograd import Function
from module.qat_layer import QATQuantModule


class CrossEntropyLossSmooth(torch.nn.modules.loss._Loss):
    def __init__(self, label_smoothing=0.1):
        super(CrossEntropyLossSmooth, self).__init__()
        self.eps = label_smoothing

    """ label smooth """

    def forward(self, output, target):
        n_class = output.size(1)
        one_hot = torch.zeros_like(output).scatter(1, target.view(-1, 1), 1)
        target = one_hot * (1 - self.eps) + self.eps / n_class
        output_log_prob = torch.nn.functional.log_softmax(output, dim=1)
        target = target.unsqueeze(1)
        output_log_prob = output_log_prob.unsqueeze(2)
        loss = -torch.bmm(target, output_log_prob)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class CELossSoft(torch.nn.modules.loss._Loss):
    """     output: output logits of the student network
            target: output logits of the teacher network
            T: temperature
            KL(p||q) = Ep \log p - \Ep log q
    """

    def __init__(self):
        super(CELossSoft, self).__init__()
        self.temperature = 1.0

    def forward(self, output, soft_logits, target=None, alpha=0.5):
        output, soft_logits = output / self.temperature, soft_logits / self.temperature
        soft_target_prob = torch.nn.functional.softmax(soft_logits, dim=1)
        output_log_prob = torch.nn.functional.log_softmax(output, dim=1)
        kd_loss = -torch.sum(soft_target_prob * output_log_prob, dim=1)
        if target is not None:
            celoss = F.cross_entropy(output, target)
            loss = alpha * (self.temperature ** 2) * kd_loss + (1. - alpha) * celoss
        else:
            loss = kd_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss



class KLLossSoft(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = 1.0

    def forward(self, output, soft_logits, target=None, alpha=0.5):
        kldivloss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(output / self.temperature, dim=1),
                                                        F.softmax(soft_logits / self.temperature, dim=1))
        if target is not None:
            celoss = F.cross_entropy(output, target)
            total_loss = alpha * (self.temperature ** 2) * kldivloss + (1. - alpha) * celoss
        else:
            total_loss = kldivloss
        return total_loss


# KurtosisLoss
class KurtosisLossCalc:
    def __init__(self, weight_tensor, kurtosis_target=1.8, k_mode='avg'):
        self.kurtosis_loss = 0
        self.kurtosis = 0
        self.weight_tensor = weight_tensor
        self.k_mode = k_mode
        self.kurtosis_target = kurtosis_target

    def fn_regularization(self):
        return self.kurtosis_calc()

    def kurtosis_calc(self):
        mean_output = torch.mean(self.weight_tensor)
        std_output = torch.std(self.weight_tensor)
        kurtosis_val = torch.mean((((self.weight_tensor - mean_output) / std_output) ** 4))
        self.kurtosis_loss = (kurtosis_val - self.kurtosis_target) ** 2
        self.kurtosis = kurtosis_val

        if self.k_mode == 'avg':
            self.kurtosis_loss = torch.mean((kurtosis_val - self.kurtosis_target) ** 2)
            self.kurtosis = torch.mean(kurtosis_val)
        elif self.k_mode == 'max':
            self.kurtosis_loss = torch.max((kurtosis_val - self.kurtosis_target) ** 2)
            self.kurtosis = torch.max(kurtosis_val)
        elif self.k_mode == 'sum':
            self.kurtosis_loss = torch.sum((kurtosis_val - self.kurtosis_target) ** 2)
            self.kurtosis = torch.sum(kurtosis_val)


def KurtosisLoss(model):
    KurtosisList = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) or isinstance(m, QATQuantModule):
            w_kurt_inst = KurtosisLossCalc(m.weight)
            w_kurt_inst.fn_regularization()
            KurtosisList.append(w_kurt_inst.kurtosis_loss)
    del KurtosisList[0]
    # del KurtosisList[-1]
    w_kurtosis_loss = reduce((lambda a, b: a + b), KurtosisList) / len(KurtosisList)
    w_kurtosis_regularization = w_kurtosis_loss
    return w_kurtosis_regularization


# SkewnessLoss:
class SkewnessLossCalc:
    def __init__(self, weight_tensor, skewness_target=0.0, k_mode='avg'):
        self.skewness_loss = 0
        self.skewness = 0
        self.weight_tensor = weight_tensor
        self.k_mode = k_mode
        self.skewness_target = skewness_target

    def fn_regularization(self):
        return self.skewness_calc()

    def skewness_calc(self):
        mean_output = torch.mean(self.weight_tensor)
        std_output = torch.std(self.weight_tensor)
        skewness_val = torch.mean((((self.weight_tensor - mean_output) / std_output) ** 3))
        self.skewness_loss = (skewness_val - self.skewness_target) ** 2
        self.skewness = skewness_val

        if self.k_mode == 'avg':
            self.skewness_loss = torch.mean((skewness_val - self.skewness_target) ** 2)
            self.skewness = torch.mean(skewness_val)
        elif self.k_mode == 'max':
            self.skewness_loss = torch.max((skewness_val - self.skewness_target) ** 2)
            self.skewness = torch.max(skewness_val)
        elif self.k_mode == 'sum':
            self.skewness_loss = torch.sum((skewness_val - self.skewness_target) ** 2)
            self.skewness = torch.sum(skewness_val)


def SkewnessLoss(model):
    SkewnessList = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) or isinstance(m, QATQuantModule):
            w_skew_inst = SkewnessLossCalc(m.weight)
            w_skew_inst.fn_regularization()
            SkewnessList.append(w_skew_inst.skewness_loss)
    del SkewnessList[0]
    # del SkewnessList[-1]
    w_skewness_loss = reduce((lambda a, b: a + b), SkewnessList) / len(SkewnessList)
    w_skewness_regularization = w_skewness_loss
    return w_skewness_regularization


def f_divergence(q_logits, p_logits, alpha, iw_clip=1e3):
    assert isinstance(alpha, float)
    q_prob = torch.nn.functional.softmax(q_logits, dim=1).detach()
    p_prob = torch.nn.functional.softmax(p_logits, dim=1).detach()
    q_log_prob = torch.nn.functional.log_softmax(q_logits, dim=1)  # gradient is only backpropagated here

    importance_ratio = p_prob / q_prob
    if abs(alpha) < 1e-3:
        importance_ratio = importance_ratio.clamp(0, iw_clip)
        f = -importance_ratio.log()
        f_base = 0
        rho_f = importance_ratio.log() - 1.0
    elif abs(alpha - 1.0) < 1e-3:
        f = importance_ratio * importance_ratio.log()
        f_base = 0
        rho_f = importance_ratio
    else:
        iw_alpha = torch.pow(importance_ratio, alpha)
        iw_alpha = iw_alpha.clamp(0, iw_clip)
        f = iw_alpha / alpha / (alpha - 1.0)
        f_base = 1.0 / alpha / (alpha - 1.0)
        rho_f = iw_alpha / alpha + f_base

    loss = torch.sum(q_prob * (f - f_base), dim=1)
    grad_loss = -torch.sum(q_prob * rho_f * q_log_prob, dim=1)
    return loss, grad_loss



class AdaptiveLossSoft(torch.nn.modules.loss._Loss):
    def __init__(self, alpha_min=-1.0, alpha_max=1.0, iw_clip=5.0):
        super(AdaptiveLossSoft, self).__init__()
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.iw_clip = iw_clip

    def forward(self, output, target, alpha_min=None, alpha_max=None):
        alpha_min = alpha_min or self.alpha_min
        alpha_max = alpha_max or self.alpha_max

        loss_left, grad_loss_left = f_divergence(output, target, alpha_min, iw_clip=self.iw_clip)
        loss_right, grad_loss_right = f_divergence(output, target, alpha_max, iw_clip=self.iw_clip)

        ind = torch.gt(loss_left, loss_right).float()
        loss = ind * grad_loss_left + (1.0 - ind) * grad_loss_right

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


