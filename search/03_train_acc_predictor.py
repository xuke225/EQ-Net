import torch.backends.cudnn as cudnn
import torch.optim as optim
import os
import time
from importlib import import_module
import copy
import sys
import math
import random
from scipy.stats import kendalltau
from matplotlib import pyplot as plt
import matplotlib

matplotlib.use('Agg')
import seaborn as sns

sys.path.append('../')
import torch
import core.config as config
import logger.logging as logging
# import models.cifar10 as models
import torchvision.models as models
from core.builder import setup_env
from core.config import cfg
from runner.trainer import Trainer, AccPredictorTrainer
from logger.meter import *
from search.accuracy_predictor.acc_dataset import AccuracyDataset
from search.accuracy_predictor.acc_predictor import AccuracyPredictor
from search.accuracy_predictor.arch_encoder import OQAEncoder
from search.bitwidth_estimator import BW_Estimator
from search.bitwidth_estimator import BitwidthDataset


config.load_configs()
logger = logging.get_logger(__name__)


def main():
    setup_env('search')
    loss_func = nn.SmoothL1Loss()
    test_func = nn.MSELoss()

    # if torch.cuda.is_available():
    torch.cuda.set_device(cfg.GPUS[0])
    cudnn.benchmark = True
    device = torch.device(f"cuda:{cfg.GPUS[0]}") if torch.cuda.is_available() else 'cpu'

    featuremaps = calc_model_featuremap(models.__dict__[cfg.ARCH]().cuda(), 224)
    flops, weights, total_params, total_flops, conv_num, fc_num = get_params_flops(model=models.__dict__[cfg.ARCH](), input_size=224)

    encoder = OQAEncoder(module_nums=conv_num + fc_num)
    predictor = AccuracyPredictor(encoder, hidden_size=200, n_layers=7, device=device)
    bw_estimator = BW_Estimator(weights, sum(weights[1:-1]), featuremaps, sum(featuremaps[1:-1]))
    bitwidthdataset = BitwidthDataset(Bitwidth_estimator=bw_estimator, module_nums=conv_num + fc_num,
                                      path=cfg.bw_dataset_path)

    accdataset = AccuracyDataset(cfg.acc_dataset_path + '/src/', bitwidthdataset)
    train_loader, valid_loader, base_acc = accdataset.build_acc_data_loader(encoder, batch_size=cfg.DATASET.train_batch_size)
    optimizer = optim.Adam(predictor.parameters(), lr=cfg.OPTIM.lr, weight_decay=cfg.OPTIM.weight_decay)
    trainer = AccPredictorTrainer(model=predictor, criterion=loss_func, test_criterion=test_func, optimizer=optimizer,
                                  lr_scheduler=None, train_loader=train_loader, test_loader=valid_loader)

    start_epoch = 0
    for epoch in range(start_epoch, cfg.OPTIM.num_epochs):
        trainer.train_epoch(cur_epoch=epoch, rank=cfg.GPUS[0])
        trainer.test_epoch(cur_epoch=epoch, rank=cfg.GPUS[0])


if __name__ == '__main__':
    main()
