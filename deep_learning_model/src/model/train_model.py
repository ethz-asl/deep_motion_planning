import _init_paths

import argparse
import logging
import os
import time
import json

import numpy as np
import tensorflow as tf

from model.training_wrapper import TrainingWrapper

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Train our model on the given dataset')

    def check_extension(extensions, filename):
        ext = os.path.splitext(filename)[1]
        if ext not in extensions:
            parser.error("Unsupported file extension: Use {}".format(extensions))
        return filename

    parser.add_argument('config_file', help='Path to the configuration file (.json)', type=lambda
            s:check_extension(['.json'],s))

    parser.add_argument('-m', '--mail', help='Send an email when training finishes',
            action='store_true')
    args = parser.parse_args()

    return args

def run_training(config, mail):
    """Train a tensorflow model"""
    with TrainingWrapper(config, mail) as wrapper:
        wrapper.run()

def main():
    logger = logging.getLogger(__name__)
    logger.info('Train our model on the given dataset:')

    args = parse_args()

    with open(args.config_file, 'r') as config_file:
        config = json.load(config_file)

    logger.info('Start training')
    run_training(config, args.mail)

if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
