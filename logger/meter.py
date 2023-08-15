# Implementation adapted from XNAS: https://github.com/MAC-AutoML/XNAS

"""Meters."""

from collections import deque
import logger.logging as logging
import torch

from core.config import cfg
from logger.timer import Timer
import numpy as np
import torch
import torch.nn as nn
from logger.thop import profile

logger = logging.get_logger(__name__)


def time_string(seconds):
    """Converts time in seconds to a fixed-width string format."""
    days, rem = divmod(int(seconds), 24 * 3600)
    hrs, rem = divmod(rem, 3600)
    mins, secs = divmod(rem, 60)
    return "{0:02},{1:02}:{2:02}:{3:02}".format(days, hrs, mins, secs)


def topk_acc(preds, labels, ks):
    """Computes the top-k error for each k."""
    err_str = "Batch dim of predictions and labels must match"
    assert preds.size(0) == labels.size(0), err_str
    # Find the top max_k predictions for each sample
    _top_max_k_vals, top_max_k_inds = torch.topk(
        preds, max(ks), dim=1, largest=True, sorted=True
    )
    # (batch_size, max_k) -> (max_k, batch_size)
    top_max_k_inds = top_max_k_inds.t()
    # (batch_size, ) -> (max_k, batch_size)
    rep_max_k_labels = labels.view(1, -1).expand_as(top_max_k_inds)
    # (i, j) = 1 if top i-th prediction for the j-th sample is correct
    top_max_k_correct = top_max_k_inds.eq(rep_max_k_labels)
    # Compute the number of topk correct predictions for each k
    topks_correct = [top_max_k_correct[:k, :].contiguous().view(-1).float().sum() for k in ks]
    return [(x / preds.size(0)) * 100.0 for x in topks_correct]


def accuracy(output, target, topk=(1,)):
    """ Computes the precision@k for the specified values of k """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    # one-hot case
    if target.ndimension() > 1:
        target = target.max(1)[1]

    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(1.0 / batch_size))

    return res


def get_params_flops(model, input_size=224):
    # Model flops and params
    input = torch.randn(1, 3, input_size, input_size)
    _, _, model_all = profile(model, inputs=(input,))

    cnt = 0
    flops = []
    params = []
    conv_num = 0
    fc_num = 0
    total_params = 0
    total_flops = 0
    for m in model.modules():
        if len(list(m.children())) > 0:  # skip for non-leaf module
            continue
        if (isinstance(m, nn.Conv2d)):
            print("CONV-%s Params=%d Flops=%3f" % (str(cnt), m.total_params, m.total_ops))
            flops.append(m.total_ops)
            params.append(m.total_params)
            total_params += m.total_params
            total_flops += m.total_ops
            cnt = cnt + 1
            conv_num = conv_num + 1
        elif (isinstance(m, nn.Linear)):
            print("FC-%s Params=%d  Flops=%3f" % (str(cnt), m.total_params, m.total_ops))
            flops.append(m.total_ops)
            params.append(m.total_params)
            total_params += m.total_params
            total_flops += m.total_ops
            cnt = cnt + 1
            fc_num = fc_num + 1
    print("Total Params = %d  |  Total Flops = %d" % (total_params, total_flops))

    return flops, params, total_params, total_flops, conv_num, fc_num


def calc_model_flops(model, input_size, mul_add=False):
    hook_list = []
    module_flops = []

    def conv_hook(self, input, output):
        output_channels, output_height, output_width = output[0].size()
        bias_ops = 1 if self.bias is not None else 0
        kernel_ops = self.kernel_size[0] * self.kernel_size[1] * (self.in_channels / self.groups)
        flops = (kernel_ops * (2 if mul_add else 1) + bias_ops) * output_channels * output_height * output_width
        module_flops.append(flops)

    def linear_hook(self, input, output):
        weight_ops = self.weight.nelement() * (2 if mul_add else 1)
        bias_ops = self.bias.nelement()
        flops = weight_ops + bias_ops
        module_flops.append(flops)

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hook_list.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hook_list.append(m.register_forward_hook(linear_hook))

    dummy_input = torch.rand(2, 3, input_size, input_size).to(next(model.parameters()).device)
    model(dummy_input)

    for hook in hook_list:
        hook.remove()
    return module_flops


def calc_model_parameters_all(model):
    total_params = 0

    params = list(model.parameters())
    for param in params:
        cnt = 1
        for d in param.size():
            cnt *= d
        total_params += cnt

    return round(total_params / 1e6, 2)


def calc_model_parameters(model):
    params = []

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            weights_num, weights_channel, weights_height, weights_width = m.weight.data.size()
            param = weights_num * weights_channel * weights_height * weights_width
            params.append(param)
        elif isinstance(m, nn.Linear):
            weights_in, weights_out = m.weight.data.size()
            param = weights_in * weights_out
            # print(param)
            params.append(param)

    # return round(total_params / 1e6, 2)
    return params


def calc_model_featuremap(model, input_size):
    hook_list = []
    module_featuremap = []

    def conv_hook(self, input, output):
        # print(input[0].size())
        batch, input_channels, input_height, input_width = input[0].size()
        featuremap = input_channels * input_height * input_width
        module_featuremap.append(featuremap)

    def linear_hook(self, input, output):
        # print(input[0].size())
        batch, input_channels = input[0].size()
        featuremap = input_channels
        module_featuremap.append(featuremap)

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hook_list.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hook_list.append(m.register_forward_hook(linear_hook))

    dummy_input = torch.rand(2, 3, input_size, input_size).to(next(model.parameters()).device)
    model(dummy_input)

    for hook in hook_list:
        hook.remove()
    return module_featuremap


class AverageMeter(object):
    """
    Computes and stores the average and current value
    Copied from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class ScalarMeter(object):
    """Measures a scalar value (adapted from Detectron)."""

    def __init__(self, window_size):
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0

    def reset(self):
        self.deque.clear()
        self.total = 0.0
        self.count = 0

    def add_value(self, value):
        self.deque.append(value)
        self.count += 1
        self.total += value

    def get_win_median(self):
        return np.median(self.deque)

    def get_win_avg(self):
        return np.mean(self.deque)

    def get_global_avg(self):
        return self.total / self.count


class TrainMeter(object):
    """Measures training stats."""

    def __init__(self, epoch_iters):
        self.epoch_iters = epoch_iters
        # self.max_iter = (cfg.OPTIM.MAX_EPOCH+cfg.OPTIM.WARMUP_EPOCH) * epoch_iters
        self.max_iter = cfg.OPTIM.num_epochs * epoch_iters
        self.iter_timer = Timer()
        self.loss = ScalarMeter(cfg.LOG_PERIOD)
        self.loss_total = 0.0
        self.lr = None
        # Current minibatch errors (smoothed over a window)
        self.mb_top1_acc = ScalarMeter(cfg.LOG_PERIOD)
        self.mb_top5_acc = ScalarMeter(cfg.LOG_PERIOD)
        # Number of misclassified examples
        self.num_top1_cor = 0
        self.num_top5_cor = 0
        self.num_samples = 0

    def reset(self, timer=True):
        if timer:
            self.iter_timer.reset()
        self.loss.reset()
        self.loss_total = 0.0
        self.lr = None
        self.mb_top1_acc.reset()
        self.mb_top5_acc.reset()
        self.num_top1_cor = 0
        self.num_top5_cor = 0
        self.num_samples = 0

    def iter_tic(self):
        self.iter_timer.tic()

    def iter_toc(self):
        self.iter_timer.toc()

    def update_stats(self, top1_acc, top5_acc, loss, lr, mb_size):
        # Current minibatch stats
        self.mb_top1_acc.add_value(top1_acc)
        self.mb_top5_acc.add_value(top5_acc)
        self.loss.add_value(loss)
        self.lr = lr
        # Aggregate stats
        self.num_top1_cor += top1_acc * mb_size
        self.num_top5_cor += top5_acc * mb_size
        self.loss_total += loss * mb_size
        self.num_samples += mb_size

    def get_iter_stats(self, cur_epoch, cur_iter):
        stats = (
            '[train_iter] Epoch[{}] ({}/{}): \t'
            'lr: {:.4f}\t'
            'Loss: {:.4f}\t'
            'Top1_acc: {:.2f}%\t'
            'Top5_acc: {:.2f}%\t'
            'Time: {:.2f}s'.format(
                cur_epoch + 1, cur_iter + 1, self.epoch_iters, self.lr, self.loss.get_win_avg(),
                self.mb_top1_acc.get_win_avg(), self.mb_top5_acc.get_win_avg(), self.iter_timer.average_time * cfg.LOG_PERIOD
            )
        )

        return stats

    def log_iter_stats(self, cur_epoch, cur_iter):
        if (cur_iter + 1) % cfg.LOG_PERIOD != 0:
            return
        stats = self.get_iter_stats(cur_epoch, cur_iter)
        print(stats)
        logger.info(stats)

    def get_epoch_stats(self, cur_epoch):
        top1_acc = self.num_top1_cor / self.num_samples
        top5_acc = self.num_top5_cor / self.num_samples
        avg_loss = self.loss_total / self.num_samples
        stats = (
            '[train_epoch] Epoch[{}]:\t'
            'lr: {:.4f}\t'
            'Loss: {:.4f}\t'
            'Top1_acc: {:.2f}%\t'
            'Top5_acc: {:.2f}%\t'
            'Time: {:.2f}s'.format(
                cur_epoch + 1, self.lr, avg_loss,
                top1_acc, top5_acc, self.iter_timer.total_time
            )
        )
        return stats

    def log_epoch_stats(self, cur_epoch):
        stats = self.get_epoch_stats(cur_epoch)
        logger.info(stats)
        print(stats)


class TestMeter(object):
    """Measures testing stats."""

    def __init__(self, max_iter):
        self.max_iter = max_iter
        self.iter_timer = Timer()
        self.max_top1_acc = 0.0
        self.max_top5_acc = 0.0
        # Number of misclassified examples
        self.num_top1_cor = 0
        self.num_top5_cor = 0
        self.num_samples = 0
        self.loss_total = 0.0

    def reset(self, min_errs=False):
        if min_errs:
            self.max_top1_acc = 0.0
            self.max_top5_acc = 0.0
        self.loss_total = 0.0
        self.iter_timer.reset()
        self.num_top1_cor = 0
        self.num_top5_cor = 0
        self.num_samples = 0

    def iter_tic(self):
        self.iter_timer.tic()

    def iter_toc(self):
        self.iter_timer.toc()

    def update_stats(self, top1_acc, top5_acc, loss, mb_size):
        self.num_top1_cor += top1_acc * mb_size
        self.num_top5_cor += top5_acc * mb_size
        self.loss_total += loss * mb_size
        self.num_samples += mb_size

    def get_epoch_top1_acc(self):
        return self.num_top1_cor / self.num_samples

    def get_epoch_stats(self, cur_epoch):
        top1_acc = self.num_top1_cor / self.num_samples
        top5_acc = self.num_top5_cor / self.num_samples
        avg_loss = self.loss_total / self.num_samples
        self.max_top1_acc = max(self.max_top1_acc, top1_acc)
        self.max_top5_acc = max(self.max_top5_acc, top5_acc)

        stats = (
            '[test_epoch] Epoch[{}]:\t Test Loss: {:.4f}\tTop1_acc: {:.2f}%\tTop5_acc: {:.2f}%\tTime: {:.2f}s\tMax Top1_acc: {:.2f}%\tMax Top5_acc: {:.2f}%\n'
                .format(cur_epoch + 1, float(avg_loss), float(top1_acc), float(top5_acc), self.iter_timer.total_time,
                        self.max_top1_acc, self.max_top5_acc)
        )
        return stats

    def log_epoch_stats(self, cur_epoch):
        stats = self.get_epoch_stats(cur_epoch)
        print(stats)
        logger.info(stats)
