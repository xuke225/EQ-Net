import copy
import gc
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import logger.timer as timer
import logger.meter as meter
import logger.logging as logging
import logger.checkpoint as checkpoint
# from module.qat_model import QATQuantModel
from module.qat_model import QATQuantSuperModel
import torch.backends.cudnn as cudnn
from core.config import cfg
from .utils import Adaptive_BN

logger = logging.get_logger(__name__)


@torch.no_grad()
def evaluation_model(model, test_loader):
    top1_meter = meter.AverageMeter()
    top5_meter = meter.AverageMeter()
    model.eval()
    for cur_iter, (inputs, labels) in enumerate(test_loader):
        inputs, labels = inputs.cuda(), labels.cuda(non_blocking=True)
        preds = model(inputs)
        top1_acc, top5_acc = meter.topk_acc(preds, labels, [1, 5])
        top1_acc, top5_acc = top1_acc.item(), top5_acc.item()
        top1_meter.update(top1_acc, inputs.size(0))
        top5_meter.update(top5_acc, inputs.size(0))

    top1_acc, top5_acc = top1_meter.avg, top5_meter.avg
    top1_meter.reset()
    top5_meter.reset()
    return top1_acc, top5_acc


@torch.no_grad()
def evaluation_model_using_AdaptiveBN(model, train_loader, test_loader, num_samples):
    top1_meter = meter.AverageMeter()
    top5_meter = meter.AverageMeter()

    model = Adaptive_BN(model=model, train_loader=train_loader, num_samples=num_samples)

    model.eval()
    for cur_iter, (inputs, labels) in enumerate(test_loader):
        inputs, labels = inputs.cuda(), labels.cuda(non_blocking=True)
        preds = model(inputs)
        top1_acc, top5_acc = meter.topk_acc(preds, labels, [1, 5])
        top1_acc, top5_acc = top1_acc.item(), top5_acc.item()
        top1_meter.update(top1_acc, inputs.size(0))
        top5_meter.update(top5_acc, inputs.size(0))

    top1_acc, top5_acc = top1_meter.avg, top5_meter.avg
    top1_meter.reset()
    top5_meter.reset()
    return top1_acc, top5_acc


@torch.no_grad()
def evaluation_quant_model_2to8(model, train_loader, test_loader, search_space):
    if isinstance(model, torch.nn.DataParallel) or isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module
    if not isinstance(model, QATQuantSuperModel):
        raise 'model is not QATQuantModel!'
    logger.info('Eval :')
    model.eval()

    for channel_wise in search_space['channel_wise_list']:
        for w_sym in search_space['w_sym_list']:
            for a_sym in search_space['a_sym_list']:
                markdown_title = "|channel-wise:{},w-sym:{},a-sym:{}|||\n|:---:|:---:|:---:|\n|bit|top-1 acc|top-5 acc|".format(
                    channel_wise, w_sym, a_sym)
                logger.info(markdown_title)
                for bit in search_space['w_bit_list']:
                    model.set_quantization_params(channel_wise, w_sym, a_sym, bit, bit)
                    model = Adaptive_BN(model=model, train_loader=train_loader, num_samples=cfg.num_samples)
                    top1_acc, top5_acc = evaluation_model(model=model, test_loader=test_loader)

                    log_state = "bit:{}, channel_wise:{}, w_sym:{}, a_sym:{}, top1_acc:{:.2f}, top5_acc:{:.2f}". \
                        format(bit, channel_wise, w_sym, a_sym, top1_acc, top5_acc)
                    markdown_state = '|{}|{:.2f}|{:.2f}|'.format(bit, top1_acc, top5_acc)
                    print(log_state)
                    logger.info(markdown_state)
                logger.info('\n')
                print('\n')


class Evaluator():
    def __init__(self):
        super(Evaluator, self).__init__()
        self.top1_meter = meter.AverageMeter()
        self.top5_meter = meter.AverageMeter()
        self.loss_meter = meter.AverageMeter()
        self.full_timer = timer.Timer()
        self.test_loader = None
        self.loss_fn = None

    @torch.no_grad()
    def evaluation_model(self, model, test_loader=None, loss_fn=None):
        if test_loader:
            self.test_loader = test_loader
        if loss_fn:
            self.loss_fn = loss_fn

        self.full_timer.tic()

        model.eval()
        loss = 0.0
        for cur_iter, (inputs, labels) in enumerate(test_loader):
            inputs, labels = inputs.cuda(), labels.cuda(non_blocking=True)
            preds = model(inputs)
            top1_acc, top5_acc = meter.topk_acc(preds, labels, [1, 5])
            top1_acc, top5_acc = top1_acc.item(), top5_acc.item()
            self.top1_meter.update(top1_acc, inputs.size(0))
            self.top5_meter.update(top5_acc, inputs.size(0))
            if self.loss_fn:
                loss = loss_fn(preds, labels)
                loss = loss.item()
                self.loss_meter.update(loss, inputs.size(0))

        top1_acc, top5_acc = self.top1_meter.avg, self.top5_meter.avg
        if self.loss_fn:
            loss = self.loss_meter.avg

        self.full_timer.toc()

        logger.info(
            'Test Loss: {:.4f}\tTop1: {:.2f}%\tTop5: {:.2f}%\tTime: {:.2f}s\n'
            .format(float(loss), float(top1_acc), float(top5_acc), self.full_timer.total_time)
        )

        self.top1_meter.reset()
        self.top5_meter.reset()
        self.loss_meter.reset()
        self.full_timer.reset()

        return top1_acc, top5_acc, loss


@torch.no_grad()
def evaluation_quant_modeltest(model_1, model_2, train_loader, test_loader, search_space):
    model_1.eval()
    model_2.eval()
    for channel_wise in search_space['channel_wise_list']:
        for w_sym in search_space['w_sym_list']:
            for a_sym in search_space['a_sym_list']:
                markdown_title = "|channel-wise:{},w-sym:{},a-sym:{}|||\n|:---:|:---:|:---:|\n|bit|top-1 acc|top-5 acc|".format(
                    channel_wise, w_sym, a_sym)
                print(markdown_title)
                for bit in search_space['w_bit_list']:
                    model_1.set_quantization_params(channel_wise, w_sym, a_sym, bit, bit)
                    model_1 = Adaptive_BN(model=model_1, train_loader=train_loader, num_samples=cfg.num_samples)
                    top1_acc, top5_acc = evaluation_model(model=model_1, test_loader=test_loader)

                    log_state = "bit:{}, channel_wise:{}, w_sym:{}, a_sym:{}, top1_acc:{:.2f}, top5_acc:{:.2f}". \
                        format(bit, channel_wise, w_sym, a_sym, top1_acc, top5_acc)
                    logger.info(log_state)

                    markdown_state = '|{}|{:.2f}|{:.2f}|'.format(bit, top1_acc, top5_acc)
                    print(markdown_state)
                    # print(state)
                logger.info('\n')
                print('\n')
