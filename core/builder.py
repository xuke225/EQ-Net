import os
import torch
import random
import numpy as np
import time
import core.config as config
import logger.logging as logging
from core.config import cfg

logger = logging.get_logger(__name__)


def setup_env():
    cfg.OUT_DIR = os.path.join(cfg.OUT_DIR, cfg.DATASET.data_set, cfg.ARCH)
    os.makedirs(cfg.OUT_DIR, exist_ok=True)

    local_time = time.strftime('%Y%m%d%H%M', time.localtime())
    cfg.OUT_DIR = os.path.join(cfg.OUT_DIR, local_time)
    cfg.fig_path = os.path.join(cfg.OUT_DIR, 'figure')
    os.makedirs(cfg.OUT_DIR, exist_ok=True)

    if cfg.quantizer == 'uaq':
        logging.setup_logging(logfile_name='train_supernet_uaq.log')
        cfg.checkpoint_dir = os.path.join(cfg.OUT_DIR, 'checkpoints', 'uaq')
        os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    elif cfg.quantizer == 'lsq':
        logging.setup_logging(logfile_name='train_supernet_lsq.log')
        cfg.checkpoint_dir = os.path.join(cfg.OUT_DIR, 'checkpoints', 'lsq')
        os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    config.dump_cfgfile()

    logger.info("Config:\n{}".format(cfg))
    logger.info(logging.dump_log_data(cfg, "cfg"))
    if cfg.DETERMINSTIC:
        # Fix RNG seeds
        np.random.seed(cfg.RNG_SEED)
        torch.manual_seed(cfg.RNG_SEED)
        torch.cuda.manual_seed(cfg.RNG_SEED)
        torch.cuda.manual_seed_all(cfg.RNG_SEED)
        random.seed(cfg.RNG_SEED)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True
    else:
        # Configure the CUDNN backend
        torch.backends.cudnn.benchmark = cfg.CUDNN_BENCH
    device = 'cuda:0'   # TODO: ddp support
    return device
