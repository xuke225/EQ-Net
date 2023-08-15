import random
import torch.backends.cudnn as cudnn

import json

import os
import math
import time
from importlib import import_module
import sys
sys.path.append('../')

from core.builder import setup_env

import matplotlib.pyplot as plt
# from core.config import cfg
import core.config as config
from logger.meter import *
# import models.cifar10 as models
import torchvision.models as models

from search.bitwidth_estimator import BitwidthDataset
from search.bitwidth_estimator import BW_Estimator

from core.config import cfg, load_configs

# Load config and check
config.load_configs()
logger = logging.get_logger(__name__)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


setup_seed(20230101)


def main():
    setup_env('search')
    device = torch.device("cuda:1") if torch.cuda.is_available() else 'cpu'
    # Statistics the size of each layer feature map
    input_size = 224 if cfg.DATASET.data_set == 'imagenet' else 32

    featuremaps = calc_model_featuremap(models.__dict__[cfg.ARCH](pretrained=True).to(device), input_size)
    # Statistics the weight size and flops size of each layer of the network
    flops, weights, total_params, total_flops, conv_num, fc_num = get_params_flops(model=models.__dict__[cfg.ARCH](), input_size=input_size)

    bw_estimator = BW_Estimator(weights, sum(weights[1:-1]), featuremaps,
                                sum(featuremaps[1:-1]))  # exclude the first and the last

    cfg.bw_dataset_path = os.path.join(cfg.OUT_DIR, 'bw_dataset')

    bw_weights_list = cfg.SEARCH_SPACE.w_bit_list
    bw_fm_list = cfg.SEARCH_SPACE.a_bit_list

    bitwidthdataset = BitwidthDataset(Bitwidth_estimator=bw_estimator, module_nums=conv_num + fc_num,
                                      path=cfg.bw_dataset_path, bw_weights_list=bw_weights_list, bw_fm_list=bw_fm_list)

    bitwidthdataset.build_bw_dataset(n_arch=500000)

    prob_map = bitwidthdataset.build_trasition_prob_matrix(0.2)

    print(prob_map)
    print(prob_map['Avg_w'])
    print(prob_map['Avg_a'])

    w_0_7_sorted_result = sorted(prob_map['a_bit_list'][20][4].items(), key=lambda item: item[0], reverse=True)
    print(w_0_7_sorted_result)

    plt.figure()

    y = [i[1] for i in w_0_7_sorted_result]
    x = [i[0] for i in w_0_7_sorted_result]

    for xloc, yloc in w_0_7_sorted_result:
        plt.text(xloc + 0.01, yloc + 0.01, '%.2f' % yloc, ha='center', va='bottom')

    plt.bar(x, y, width=0.1)
    plt.xlabel('choice', size=12)
    plt.ylabel('Frequency', size=12)
    plt.savefig(cfg.bw_dataset_path + '/distribution_weight.png',
                format='png')


if __name__ == '__main__':
    main()
