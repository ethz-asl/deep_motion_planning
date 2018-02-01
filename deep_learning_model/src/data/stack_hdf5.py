# -*- coding: utf-8 -*-

from __future__ import print_function

import os
import logging
import argparse

import pandas as pd

def parse_args():
  """Parse input arguments."""
  parser = argparse.ArgumentParser(description='Stack the given HDF5 container into a new file')
  parser.add_argument('input_files', help='Files to stack', nargs=argparse.REMAINDER)

  def check_extension(extensions, filename):
    ext = os.path.splitext(filename)[1]
    if ext not in extensions:
      parser.error("Unsupported file extension: Use {}".format(extensions))
    return filename

  parser.add_argument('-o', '--output_file', help='Filename of the combined data', type=lambda
      s:check_extension([".h5"],s), default='stacked.h5')
  parser.add_argument('-s', '--chunksize', help='Size of chunks read from the input file', default=10000, type=int)
  args = parser.parse_args()

  return args, parser

def main(project_dir):
  logger = logging.getLogger(__name__)
  logger.info('Combine HDF5 container into a single one')

  (args, parser) = parse_args()

  # Make some input file checks
  input_files = args.input_files
  if len(input_files) < 2:
    logger.error('You have to specify at least two files')
    parser.print_help()
    exit(-1)

  for f in input_files:
    if not os.path.exists(f):
      logger.error('At least one input file does not exist: {}'.format(f))
      exit(-1)
    elif os.path.basename(f).split('.')[1] != 'h5':
      logger.error('File must be HDF5 (.h5): {}'.format(f))
      exit(-1)

  # Do not overwrite existing data containers
  if os.path.exists(args.output_file):
    logger.error('Output file already exists: {}'.format(args.output_file))
    exit(-1)

  with pd.HDFStore(args.output_file) as store:
    num_elements = 0

    # Iterate over all input files and add them to the HDF5 container
    for i,f in enumerate(input_files):

      with pd.HDFStore(f, mode='r') as input_file:
        nrows = input_file.get_storer('data').nrows

        nadded = 0
        for chunk_index in range(nrows // args.chunksize + 1):
          chunk = input_file.select('data',
              start=chunk_index*args.chunksize, stop=(chunk_index+1)*args.chunksize)

          progress = (chunk_index*args.chunksize*100.0) / nrows
          print('{:.2f}% ({}/{}) {}'.format(progress, chunk_index*args.chunksize, nrows, f), end='\r')

          # Make sure the final dataframe has a continous index
          chunk.index = pd.Series(chunk.index) + num_elements
          store.append('data', chunk)

          num_elements += chunk.shape[0]
          nadded += chunk.shape[0]

      print('{:.2f}% ({}/{}) {}'.format(100.0, nadded, nrows, f))

  logger.info('We combined {} lines'.format(num_elements))
  logger.info('File saved: {}'.format(args.output_file))


if __name__ == '__main__':
  log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
  logging.basicConfig(level=logging.INFO, format=log_fmt)

  # not used in this stub but often useful for finding various files
  project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)

  main(project_dir)

