# Implementation adapted from XNAS: https://github.com/MAC-AutoML/XNAS

"""Learning rate schedulers."""

import math
from core.config import cfg


def _calc_learning_rate(
    init_lr, n_epochs, epoch, n_iter=None, iter=0,
):
    epoch -= cfg.OPTIM.WARMUP_EPOCH
    if cfg.OPTIM.LR_POLICY == "cos":
        t_total = n_epochs * n_iter
        t_cur = epoch * n_iter + iter
        lr = 0.5 * init_lr * (1 + math.cos(math.pi * t_cur / t_total))
    elif cfg.OPTIM.LR_POLICY == "step":
        # Rule of BigNAS: decay learning rate by 0.97 every 2.4 epochs
        # t_total = n_epochs * n_iter
        # t_cur = epoch * n_iter + iter
        t_cur_epoch = epoch + iter / n_iter
        lr = (0.97 ** (t_cur_epoch / 2.4)) * init_lr
    else:
        raise ValueError("do not support: {}".format(cfg.OPTIM.LR_POLICY))
    return lr


def _warmup_adjust_learning_rate(
        init_lr, n_epochs, epoch, n_iter, iter=0, warmup_lr=0
    ):
        """adjust lr during warming-up. Changes linearly from `warmup_lr` to `init_lr`."""
        T_cur = epoch * n_iter + iter + 1
        t_total = n_epochs * n_iter
        new_lr = T_cur / t_total * (init_lr - warmup_lr) + warmup_lr
        return new_lr


def adjust_learning_rate_per_batch(epoch, n_iter=None, iter=0, warmup=False):
    """adjust learning of a given optimizer and return the new learning rate"""
    
    # init_lr = cfg.OPTIM.BASE_LR * cfg.NUM_GPUS
    init_lr = cfg.OPTIM.BASE_LR
    n_epochs = cfg.OPTIM.MAX_EPOCH
    n_warmup_epochs = cfg.OPTIM.WARMUP_EPOCH
    warmup_lr = init_lr * cfg.OPTIM.WARMUP_FACTOR
    
    if warmup:
        new_lr = _warmup_adjust_learning_rate(
            init_lr, n_warmup_epochs, epoch, n_iter, iter, warmup_lr
        )
    else:
        new_lr = _calc_learning_rate(
            init_lr, n_epochs, epoch, n_iter, iter
        )
    return new_lr
