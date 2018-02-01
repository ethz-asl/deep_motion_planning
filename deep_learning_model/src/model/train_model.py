import _init_paths

import argparse
import logging
import os
import time

import tensorflow as tf
import numpy as np

from model.training_wrapper import TrainingWrapper

def parse_args():
  """Parse input arguments."""
  parser = argparse.ArgumentParser(description='Train our model on the given dataset')

  def check_extension(extensions, filename):
    ext = os.path.splitext(filename)[1]
    if ext not in extensions:
      parser.error("Unsupported file extension: Use {}".format(extensions))
    return filename

  parser.add_argument('datafile_train', help='Filename of the training data', type=lambda
      s:check_extension(['.h5'],s))
  parser.add_argument('datafile_eval', help='Filename of the evaluation data', type=lambda
      s:check_extension(['.h5'],s))
  parser.add_argument('-s', '--max_steps', help='Number of batches to run', type=int, default=1000000)
  parser.add_argument('-e', '--eval_steps', help='Evaluate model every N steps', type=int, default=25000)
  parser.add_argument('-b', '--batch_size', help='Size of training batches', type=int,  default=16)
  parser.add_argument('-t', '--train_dir', help='Directory to save the model snapshots',
      default='./models/default')
  parser.add_argument('-l', '--learning_rate', help='Initial learning rate', type=float, default=0.01)
  parser.add_argument('--weight_initialize', help='Initialize network weights with this checkpoint\
            file', type=str)
  parser.add_argument('-m', '--mail', help='Send an email when training finishes',
      action='store_true')
  args = parser.parse_args()

  return args

def run_training(args):
  """Train a tensorflow model"""
  with TrainingWrapper(args) as wrapper:
    wrapper.run()

def main():
  logger = logging.getLogger(__name__)
  logger.info('Train our model on the given dataset:')

  args = parse_args()
  logger.info(args.datafile_train)

  logger.info('Start training')
  run_training(args)

if __name__ == "__main__":
  log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
  logging.basicConfig(level=logging.INFO, format=log_fmt)

  main()
