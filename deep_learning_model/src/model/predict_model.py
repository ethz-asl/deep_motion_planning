import _init_paths

import argparse
import logging
import os
import time
import math

import numpy as np
import pandas as pd

from data.data_handler import DataHandler
from deep_motion_planner.tensorflow_wrapper import TensorflowWrapper

class Config():
  """Configuration object to store various parameters used in the code"""
  perception_radius = 10.0 # Radius to truncate the sensor data
  number_of_scans = 1080 # Number of used laser scans
  model = None # Path to the evaluated model
  data = None # Path to the data used in the evaluation
  eval_n_elements = 60000 # Number of samples used to evaluate the model
  use_snapshots = False # Specify if the model should be loaded from snapshots
  write_result = False # Specify if we write the results into a .csv file
  results = None # Path to the .csv result file
  mean_filter_size = 5

def parse_args():
  """Parse input arguments."""
  parser = argparse.ArgumentParser(description='Evaluate the given model on the given data')

  def check_extension(extensions, filename):
    ext = os.path.splitext(filename)[1]
    if ext not in extensions:
      parser.error("Unsupported file extension: Use {}".format(extensions))
    return filename

  parser.add_argument('datafile_eval', help='Filename of the evaluation data', type=lambda
      s:check_extension(['.h5'],s))
  parser.add_argument('model', help='Path to the model')
  parser.add_argument('-s', '--snapshots', help='Load model from a snapshot directory',
      action='store_true')
  parser.add_argument('-c', '--capture', help='Filepath to write (append) results', default=None)
  args = parser.parse_args()

  return args


def run_evaluation(cfg):
  """Test a tensorflow model"""
  logger = logging.getLogger(run_evaluation.__name__)

  data, ground_truth = DataHandler(cfg.data, cfg.eval_n_elements, shuffle=False).next_batch()

  prediction = np.zeros(ground_truth.shape)
  samples = prediction.shape[0]
  tictoc = np.zeros(samples)
  print_every = 2500

  logger.info('Load Tensorflow model')
  with TensorflowWrapper(os.path.dirname(cfg.model), os.path.basename(cfg.model),
      cfg.use_snapshots) as model:

    logger.info('Start model evaluation')
    for i in range(samples):

      start_time = time.time()
      prediction[i,:] = model.inference(data[i,:])
      tictoc[i%print_every] = time.time() - start_time

      next_i = i+1
      if next_i > 0 and next_i % print_every == 0 or next_i == samples:
        logger.info('Processed: {:6.2f}% ({:5d}/{}) ({:.{width}f} ms/sample)'.format(next_i/samples*100.0, next_i, samples,
          np.mean(tictoc)*1e3, width=math.ceil(math.log10(samples))))

  # Compute the loss
  error = np.mean(np.abs(ground_truth - prediction), axis=0)
  comined_error = np.mean(error)

  logger.info('Evaluation error: {:.5f} ({:.5f}/{:.5f})'.format(comined_error, error[0], error[1]))

  # Write the results into a .csv file
  if cfg.write_result:
    add_header_line = not os.path.isfile(cfg.results)

    with open(cfg.results, 'a') as capture:
      if add_header_line:
        capture.write('date_time,model,error_combined,error_linear,error_angular,execution_time\n')

      capture.write('{},{},{},{},{},{}\n'.format(
        time.strftime('%Y-%m-%d %H:%M:%S'),
        cfg.model,
        comined_error,
        error[0],
        error[1],
        np.mean(tictoc)
        ))

def main():
  logger = logging.getLogger(__name__)
  logger.info('Test our model on the given dataset:')

  args = parse_args()
  logger.info(args.datafile_eval)

  cfg = Config()
  cfg.model = os.path.abspath(args.model)
  cfg.data = args.datafile_eval
  cfg.use_snapshots = args.snapshots
  if args.capture:
    cfg.write_result = True
    cfg.results = args.capture

  logger.info('Start evaluation')
  run_evaluation(cfg)

if __name__ == "__main__":
  log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
  logging.basicConfig(level=logging.INFO, format=log_fmt)

  main()
