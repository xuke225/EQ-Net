import os
import sys

sys.path.append('../')
import torch.backends.cudnn as cudnn
# import models.cifar10 as models
import torchvision.models as models
from core.builder import setup_env

import core.config as config
import runner.evaluator
from runner.utils import Adaptive_BN, get_train_samples, convert_to_QuantSuperModel

from data import imagenet_dali, imagenet_train_val_split
from logger.meter import *
from search.accuracy_predictor.acc_dataset import AccuracyDataset
from search.accuracy_predictor.arch_encoder import OQAEncoder
from search.bitwidth_estimator import BW_Estimator
from search.bitwidth_estimator import BitwidthDataset
from runner.evaluator import evaluation_quant_model_2to8, evaluation_model, evaluation_model_using_AdaptiveBN

import numpy as np
import random
from core.config import cfg, load_configs

# Load config and check
config.load_configs()
logger = logging.get_logger(__name__)



def main():
    setup_env('search')
    cfg.OUT_DIR = os.path.join(cfg.OUT_DIR, 'acc_dataset')
    model = models.__dict__[cfg.ARCH](pretrained=True)
    # print(model)
    if torch.cuda.is_available():
        torch.cuda.set_device(cfg.GPUS[0])
        model = model.cuda()
        cudnn.benchmark = True

    # input_size = 224 if cfg.DATASET.data_set is 'imagenet' else 32
    # if cfg.DATASET.data_set == 'imagenet':
        # train_loader, train_set = imagenet_dali.get_imagenet_iter_torch('train', cfg.DATASET.data_path,
        #                                                                 cfg.DATASET.train_batch_size,
        #                                                                 num_threads=2, crop=224, device_id=cfg.GPUS[0])
        #
        # test_loader, test_set = imagenet_dali.get_imagenet_iter_torch('val', cfg.DATASET.data_path,
        #                                                               cfg.DATASET.eval_batch_size,
        #                                                               num_threads=2, crop=224, device_id=cfg.GPUS[0])
    Loader = imagenet_train_val_split.Data(cfg)
    train_loader = Loader.trainLoader
    test_loader = Loader.validLoader
    input_size = 224
    # else:
    #     loader = cifar10.Data(cfg.DATASET)
    #     [train_loader, test_loader] = [loader.trainLoader, loader.testLoader]
    #     input_size = 32

    featuremaps = calc_model_featuremap(models.__dict__[cfg.ARCH]().cuda(), input_size)
    flops, weights, total_params, total_flops, conv_num, fc_num = get_params_flops(model=models.__dict__[cfg.ARCH](),
                                                                                   input_size=input_size)

    wq_params = {'n_bits': cfg.w_bit, 'scale_method': 'mse', 'leaf_param': True}
    aq_params = {'n_bits': cfg.a_bit, 'scale_method': 'mse', 'leaf_param': cfg.act_quant}
    search_space = {
        'w_bit_list': cfg.SEARCH_SPACE.w_bit_list,
        'a_bit_list': cfg.SEARCH_SPACE.a_bit_list,
        'w_sym_list': cfg.SEARCH_SPACE.w_sym_list,
        'a_sym_list': cfg.SEARCH_SPACE.a_sym_list,
        'channel_wise_list': cfg.SEARCH_SPACE.channel_wise_list,
    }
    qnn = convert_to_QuantSuperModel(model, wq_params=wq_params, aq_params=aq_params, quantizer=cfg.quantizer,
                                     search_space=search_space)
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

    # qnn = torch.nn.DataParallel(qnn, device_ids=cfg.GPUS)
    qnn = qnn.cuda()
    # w_bit_list = [4, 2, 2, 2, 2, 2, 4, 3, 6, 7, 3, 2, 3, 7, 4, 7, 3, 8, 3, 7, 6]
    # a_bit_list = [4, 7, 3, 8, 3, 7, 7, 2, 2, 2, 2, 2, 4, 3, 6, 7, 3, 2, 3, 7, 6]
    # qnn.set_quantization_params(True, True, True, w_bit_list, a_bit_list)
    # qnn.module.set_layer_according_to_mix_factor(True, True, w_bit_list, a_bit_list)
    # top1, top5 = evaluation_model(qnn, test_loader=test_loader)
    # print('acc:{}'.format(top1))
    # return

    bw_estimator = BW_Estimator(weights, sum(weights[1:-1]), featuremaps, sum(featuremaps[1:-1]))

    bw_weights_list = cfg.SEARCH_SPACE.w_bit_list
    bw_fm_list = cfg.SEARCH_SPACE.a_bit_list
    bw_dataset = BitwidthDataset(Bitwidth_estimator=bw_estimator, module_nums=conv_num + fc_num,
                                 path=cfg.bw_dataset_path, bw_weights_list=bw_weights_list, bw_fm_list=bw_fm_list)

    acc_dataset = AccuracyDataset(cfg.OUT_DIR, bw_dataset)

    if cfg.ARCH == 'mobilenet_v2' or cfg.ARCH == 'efficientnet_b0':
        min_bw = 3.4
    else:
        min_bw = 2.4
    print(min_bw)
    acc_dataset.build_acc_dataset(qnn, test_loader, train_loader, evaluation_model_using_AdaptiveBN,
                                  n_arch=cfg.acc_sample, min_bw=min_bw,
                                  max_bw=7.6, channel_wise=cfg.EVAL.channel_wise, w_sym=cfg.EVAL.w_sym,
                                  a_sym=cfg.EVAL.a_sym, num_samples=cfg.num_samples)


if __name__ == '__main__':
    main()
