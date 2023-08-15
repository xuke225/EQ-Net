# Implementation adapted from XNAS: https://github.com/MAC-AutoML/XNAS

"""Logging."""

import os
import decimal
import logging
import simplejson
from core.config import cfg

# Show filename and line number in logs
_FORMAT = "[%(filename)s: %(lineno)3d]: %(message)s"

# Log file name
_LOG_FILE = "stdout.log"

# Data output with dump_log_data(data, data_type) will be tagged w/ this
_TAG = "json_stats: "

# Data output with dump_log_data(data, data_type) will have data[_TYPE]=data_type
_TYPE = "_type"


def setup_logging(logfile_name):
    """Sets up the logging."""
    # Clear the root logger to prevent any existing logging config
    # (e.g. set by another module) from messing with our setup
    # logging.root.handlers = []
    # # Construct logging configuration
    # logging_config = {"level": logging.INFO, "format": _FORMAT}
    # logging_config["filename"] = os.path.join(cfg.OUT_DIR, logfile_name)
    # # Configure logging
    # logging.basicConfig(**logging_config)

    logger = logging.getLogger('gal')
    log_format = '%(asctime)s | %(message)s'
    formatter = logging.Formatter(log_format, datefmt='%m/%d %I:%M:%S %p')
    file_path = os.path.join(cfg.OUT_DIR, logfile_name)
    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)


def get_logger(name):
    """Retrieves the logger."""
    name = os.path.join(cfg.OUT_DIR, name)

    return logging.getLogger(name)


def dump_log_data(data, data_type, prec=4):
    """Covert data (a dictionary) into tagged json string for logging."""
    data[_TYPE] = data_type
    data = float_to_decimal(data, prec)
    data_json = simplejson.dumps(data, sort_keys=False, use_decimal=True)
    return "{:s}".format(data_json)


def float_to_decimal(data, prec=4):
    """Convert floats to decimals which allows for fixed width json."""
    if isinstance(data, dict):
        return {k: float_to_decimal(v, prec) for k, v in data.items()}
    if isinstance(data, float):
        return decimal.Decimal(("{:." + str(prec) + "f}").format(data))
    else:
        return data
