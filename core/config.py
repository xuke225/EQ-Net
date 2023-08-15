# Implementation adapted from XNAS: https://github.com/MAC-AutoML/XNAS

"""Configuration file (powered by YACS)."""

import os
import sys
import argparse
from yacs.config import CfgNode

# Global config object
_C = CfgNode(new_allowed=True)
cfg = _C

_C.CUDNN_BENCH = True

_C.LOG_PERIOD = 30

_C.EVAL_PERIOD = 1

_C.SAVE_PERIOD = 5

_C.OUT_DIR = "exp/"

_C.DETERMINSTIC = True


# -------------------------------------------------------- #

def dump_cfgfile(cfg_dest="config.yaml"):
    """Dumps the config to the output directory."""
    cfg_file = os.path.join(cfg.OUT_DIR, cfg_dest)
    with open(cfg_file, "w") as f:
        _C.dump(stream=f)


def load_cfgfile(out_dir, cfg_dest="config.yaml"):
    """Loads config from specified output directory."""
    cfg_file = os.path.join(out_dir, cfg_dest)
    _C.merge_from_file(cfg_file)


def load_configs():
    """Load config from command line arguments and set any specified options.
       How to use: python xx.py --cfg path_to_your_config.cfg test1 0 test2 True
       opts will return a list with ['test1', '0', 'test2', 'True'], yacs will compile to corresponding values
    """
    parser = argparse.ArgumentParser(description="Config file options.")

    parser.add_argument("--cfg", default='../configs/train-supernet.yaml', type=str)

    args = parser.parse_args()
    _C.merge_from_file(args.cfg)

