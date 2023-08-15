import os
import copy
import random
import sys
sys.path.append('../')

from torch.utils.data.distributed import DistributedSampler
from logger.checkpoint import save_checkpoint
import torch.optim as optim
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from data import imagenet_dali

import core.config as config
import logger.meter as meter
import logger.logging as logging
from core.config import cfg
from core.builder import setup_env
# from runner.criterion import AdaptiveLabelSmoothing
# DDP
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from runner.trainer import Trainer, QuantSuperNetTrainer
from runner.evaluator import Evaluator, evaluation_model, evaluation_quant_model_2to8
from runner.utils import Adaptive_BN, get_train_samples, convert_to_QuantSuperModel, CosineDecay
from runner.scheduler import adjust_learning_rate_per_batch
from runner.criterion import *
import torchvision.models as models

# Load config and check
config.load_configs()
logger = logging.get_logger(__name__)
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, cfg.GPUS))


def main(local_rank, world_size):
    setup_env('supernet')
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', rank=local_rank, world_size=world_size)

    model = models.__dict__[cfg.ARCH](pretrained=True)
    model = model.cuda()
    # if cfg.USE_MLP:
    #     mlp = Global_T().cuda()
    cudnn.benchmark = True

    criterion = eval(cfg.CRITERION.criterion)
    soft_criterion = eval(cfg.CRITERION.soft_criterion)

    # Data loaders
    # train_loader = imagenet_dali.get_imagenet_iter_dali('train', cfg.DATASET.data_path, cfg.DATASET.train_batch_size,
    #                                                     num_threads=4, crop=224, device_id=cfg.GPUS[0])
    #
    # test_loader = imagenet_dali.get_imagenet_iter_dali('val', cfg.DATASET.data_path, cfg.DATASET.eval_batch_size,
    #                                                    num_threads=4, crop=224, device_id=cfg.GPUS[0])
    _, train_set = imagenet_dali.get_imagenet_iter_torch('train', cfg.DATASET.data_path,
                                                         cfg.DATASET.train_batch_size,
                                                         num_threads=2, crop=224, device_id=cfg.GPUS[0])

    _, test_set = imagenet_dali.get_imagenet_iter_torch('val', cfg.DATASET.data_path,
                                                        cfg.DATASET.eval_batch_size,
                                                        num_threads=2, crop=224, device_id=cfg.GPUS[0])

    train_sampler = DistributedSampler(train_set)
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=cfg.DATASET.train_batch_size,
                                               sampler=train_sampler,
                                               shuffle=False, pin_memory=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=cfg.DATASET.train_batch_size, shuffle=False,
                                              pin_memory=True, num_workers=2)

    # (1)convert to QuantModel
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

    # (2)Init Quant params
    cali_data = get_train_samples(train_loader=train_loader, num_samples=cfg.num_samples)
    qnn.set_quant_state(True, True)
    with torch.no_grad():
        _ = qnn(cali_data.cuda())

    optimizer = optim.SGD(qnn.parameters(), lr=cfg.OPTIM.lr, momentum=cfg.OPTIM.momentum,
                          weight_decay=cfg.OPTIM.weight_decay)
    if cfg.OPTIM.optimizer == 'Adam':
        optimizer = optim.Adam(qnn.parameters(), lr=cfg.OPTIM.lr, weight_decay=cfg.OPTIM.weight_decay)
        # if cfg.USE_MLP:
        #     optimizer = optim.Adam([
        #         {'params': qnn.parameters(), 'lr': cfg.OPTIM.lr, 'weight_decay': cfg.OPTIM.weight_decay},
        #         {'params': mlp.parameters()},
        #     ])
    elif cfg.OPTIM.optimizer == 'SGD':
        optimizer = optim.SGD(qnn.parameters(), lr=cfg.OPTIM.lr, momentum=cfg.OPTIM.momentum,
                              weight_decay=cfg.OPTIM.weight_decay)
        # if cfg.USE_MLP:
        #     optimizer = optim.SGD([
        #         {'params': qnn.parameters(), 'lr': cfg.OPTIM.lr, 'weight_decay': cfg.OPTIM.weight_decay},
        #         {'params': mlp.parameters()},
        #     ])

    # (5)load checkpoint
    if cfg.Resume is not None:
        ckpt = torch.load(cfg.Resume, map_location='cuda')
        qnn.load_state_dict(ckpt['state_dict'])
        start_epoch = ckpt['epoch']
        # optimizer.load_state_dict(ckpt['optimizer'])
    else:
        model_state_dict = qnn.state_dict()
        state = {
            'state_dict': model_state_dict,
            'epoch': 0,
            'optimizer': optimizer.state_dict()
        }
        save_checkpoint(state, False, 'model_super_' + cfg.quantizer, 0)
        start_epoch = 0

    # (6)Training
    qnn = DDP(qnn, device_ids=[local_rank], find_unused_parameters=True)
    trainer = QuantSuperNetTrainer(
        model=qnn,
        criterion=criterion,
        soft_criterion=soft_criterion,
        teacher_model=model,
        optimizer=optimizer,
        lr_scheduler=None,
        train_loader=train_loader,
        test_loader=test_loader,
        mlp=None
    )

    # Training
    logger.info("Start supernet training.")
    dist.barrier()
    trainer.start()
    for cur_epoch in range(start_epoch, cfg.OPTIM.num_epochs):
        trainer.train_loader.sampler.set_epoch(cur_epoch)

        decay_value = None
        # if cfg.USE_MLP:
        #     gradient_decay = CosineDecay(max_value=0, min_value=-1, num_loops=10)
        #     decay_value = gradient_decay.get_value(cur_epoch + 1)

        trainer.train_epoch(cur_epoch, rank=local_rank, decay_value=decay_value)
        trainer.test_epoch(cur_epoch, rank=local_rank)
    trainer.finish()
    dist.barrier()
    torch.cuda.empty_cache()
    if local_rank == 0:
        evaluation_quant_model_2to8(model=qnn, train_loader=train_loader, test_loader=test_loader, search_space=search_space)


if __name__ == '__main__':
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '23333'

    if torch.cuda.is_available():
        cfg.NUM_GPUS = torch.cuda.device_count()

    mp.spawn(main, nprocs=cfg.NUM_GPUS, args=(cfg.NUM_GPUS,), join=True)

