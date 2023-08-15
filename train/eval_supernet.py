import sys

sys.path.append('../')
import torch.backends.cudnn as cudnn
import torchvision.models as models
from core.builder import setup_env

import core.config as config
from runner.utils import Adaptive_BN, get_train_samples, convert_to_QuantSuperModel

from data import imagenet_dali
from logger.meter import *
from runner.evaluator import evaluation_quant_model_2to8, evaluation_model, evaluation_model_using_AdaptiveBN
import numpy as np
import random
from core.config import cfg, load_configs

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
    setup_env()
    model = models.__dict__[cfg.ARCH](pretrained=True)
    print(model)
    if torch.cuda.is_available():
        torch.cuda.set_device(cfg.GPUS[0])
        model = model.cuda()
        cudnn.benchmark = True

    train_loader, _ = imagenet_dali.get_imagenet_iter_torch('train', cfg.DATASET.data_path,
                                                            cfg.DATASET.train_batch_size,
                                                            num_threads=2, crop=224, device_id=cfg.GPUS[0])

    test_loader, _ = imagenet_dali.get_imagenet_iter_torch('val', cfg.DATASET.data_path, cfg.DATASET.eval_batch_size,
                                                           num_threads=2, crop=224, device_id=cfg.GPUS[0])

    wq_params = {'n_bits': cfg.w_bit, 'scale_method': 'mse', 'leaf_param': True}
    aq_params = {'n_bits': cfg.a_bit, 'scale_method': 'mse', 'leaf_param': True}
    search_space = {
        'w_bit_list': cfg.SEARCH_SPACE.w_bit_list,
        'a_bit_list': cfg.SEARCH_SPACE.a_bit_list,
        'w_sym_list': cfg.SEARCH_SPACE.w_sym_list,
        'a_sym_list': cfg.SEARCH_SPACE.a_sym_list,
        'channel_wise_list': cfg.SEARCH_SPACE.channel_wise_list,
    }
    qnn = convert_to_QuantSuperModel(model, wq_params=wq_params, aq_params=aq_params, quantizer=cfg.quantizer,
                                     search_space=search_space)

    if not cfg.disable_8bit_head_stem:
        print('Setting the first and the last layer to 8-bit')
        qnn.set_first_last_layer_to_8bit()

    cali_data = get_train_samples(train_loader=train_loader, num_samples=cfg.num_samples)
    qnn.set_quant_state(True, True)
    with torch.no_grad():
        _ = qnn(cali_data.cuda())

    print('load model...')
    ckpt = torch.load(cfg.super_model, map_location='cuda')
    qnn.load_state_dict(ckpt['state_dict'])

    search_space = {
        'w_bit_list': [2, 3, 4, 5, 6, 7, 8],
        'a_bit_list': [2, 3, 4, 5, 6, 7, 8],
        'w_sym_list': [True, False],
        'a_sym_list': [True, False],
        'channel_wise_list': [True, False],
    }
    evaluation_quant_model_2to8(model=qnn, train_loader=train_loader, test_loader=test_loader,
                                search_space=search_space)


if __name__ == '__main__':
    main()
