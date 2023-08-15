# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.
import os
import sys
import torch.backends.cudnn as cudnn
import models.cifar10 as models
from core.builder import setup_env
from runner.criterion import AdaptiveLabelSmoothing
import core.config as config
import runner.evaluator
from runner.utils import Adaptive_BN, get_train_samples, convert_to_QuantSuperModel

from data import cifar10
from logger.meter import *
from search.accuracy_predictor.acc_dataset import AccuracyDataset
from search.accuracy_predictor.arch_encoder import OQAEncoder
from search.bitwidth_estimator import BW_Estimator
from search.bitwidth_estimator import BitwidthDataset
from runner.evaluator import evaluation_quant_model_2to8, evaluation_model
from runner.evaluator import evaluation_quant_model_2to8, evaluation_model, evaluation_model_using_AdaptiveBN

import numpy as np
import random
from core.config import cfg, load_configs

import os
import torch
import torch.nn as nn
import data

__all__ = ['AccuracyPredictor']


class AccuracyPredictor(nn.Module):

    def __init__(self, arch_encoder, hidden_size=400, n_layers=3,
                 checkpoint_path=None, device='cuda:0'):
        super(AccuracyPredictor, self).__init__()
        self.arch_encoder = arch_encoder
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.device = device

        # build layers
        layers = []
        for i in range(self.n_layers):
            layers.append(nn.Sequential(
                nn.Linear(self.arch_encoder.n_dim if i == 0 else self.hidden_size, self.hidden_size),
                nn.ReLU(inplace=True),
            ))
        layers.append(nn.Linear(self.hidden_size, 1, bias=False))
        self.layers = nn.Sequential(*layers)
        self.base_acc = nn.Parameter(torch.zeros(1, device=self.device), requires_grad=False)

        if checkpoint_path is not None and os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            if 'state_dict' in checkpoint:
                checkpoint = checkpoint['state_dict']
            self.load_state_dict(checkpoint)
            print('Loaded checkpoint from %s' % checkpoint_path)

        self.layers = self.layers.to(self.device)

        self.super_model = None
        self.train_loader = None
        self.test_loader = None
        self.init = False

    def init_super_model(self, ):
        setup_env('supernet')
        cfg.OUT_DIR = os.path.join(cfg.OUT_DIR, 'acc_dataset')
        model = models.__dict__[cfg.ARCH]()
        print(model)
        if torch.cuda.is_available():
            torch.cuda.set_device(cfg.GPUS[0])
            model = model.cuda()
            cudnn.benchmark = True
        # Data loaders
        loader = cifar10.Data(cfg.DATASET)
        [train_loader, test_loader] = [loader.trainLoader, loader.testLoader]

        wq_params = {'n_bits': cfg.w_bit, 'scale_method': 'mse', 'leaf_param': True}
        aq_params = {'n_bits': cfg.a_bit, 'scale_method': 'mse', 'leaf_param': cfg.act_quant}

        qnn = convert_to_QuantSuperModel(model, wq_params=wq_params, aq_params=aq_params)
        qnn.eval()
        if not cfg.disable_8bit_head_stem:
            print('Setting the first and the last layer to 8-bit')
            qnn.set_first_last_layer_to_8bit()

        cali_data = get_train_samples(train_loader=train_loader, num_samples=cfg.num_samples)

        # 初始化量化参数
        qnn.set_quant_state(True, True)
        with torch.no_grad():
            _ = qnn(cali_data.cuda())

        if cfg.super_model is not None:
            ckpt = torch.load(cfg.super_model, map_location='cuda')
            qnn.load_state_dict(ckpt['state_dict'])
            # evaluation_quant_model_2to8(qnn, train_loader=train_loader, test_loader=test_loader)

        self.super_model = torch.nn.DataParallel(qnn, device_ids=cfg.GPUS)
        self.train_loader = train_loader
        self.test_loader = test_loader

    def forward(self, x):
        y = self.layers(x).squeeze()
        return y + self.base_acc

    def predict_acc(self, arch_dict_list, channel_wise, w_sym, a_sym):
        # if not self.init:
        #     self.init_super_model()
        #     self.init = True
        X = [self.arch_encoder.arch2feature(arch_dict, channel_wise, w_sym, a_sym) for arch_dict in arch_dict_list]
        X = torch.tensor(np.array(X)).float().to(self.device)
        acc = self.forward(X)
        # top1_list = []
        # for arch in arch_dict_list:
        #     self.super_model.module.set_layer_according_to_mix_factor(True, False, arch['w_bit_list'], arch['a_bit_list'])
        #     top1, _ = evaluation_model_using_AdaptiveBN(self.super_model, train_loader=self.train_loader, test_loader=self.test_loader)
        #     top1_list.append(top1)
        # acc = torch.tensor(top1_list)

        return acc
