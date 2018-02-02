# -*- coding: utf-8 -*-
from __future__ import print_function

import sys
sys.path.insert(0, '/usr/local/lib/python2.7/dist-packages')


import os
import logging
import argparse
import random

import pandas as pd


def parse_args():
  """Parse input arguments."""
  parser = argparse.ArgumentParser(description='Fuse csv files and prepare data')
  parser.add_argument('mixer_file', help='Path to the mixer definition')
  parser.add_argument('--random', help='Select the files randomly', action='store_true')
  parser.add_argument('--list', help='Use list of files as mixer', action='store_true')

  def check_extension(extensions, filename):
    ext = os.path.splitext(filename)[1]
    if ext not in extensions:
      parser.error("Unsupported file extension: Use {}".format(extensions))
    return filename

  parser.add_argument('output_file', help='Filename of the processed data (stored in \
      ./data/processed)', type=lambda
      s:check_extension([".h5"], s))

  args = parser.parse_args()

  return args


def parse_mixer_file(filepath):
  """
  Take a mixer file and extract the paths and number of files taken from those paths
  """
  logger = logging.getLogger(parse_mixer_file.__name__)
  mixer = list()
  with open(filepath, 'r') as mixer_file:
    for i, line in enumerate(mixer_file):

      line = line.strip()

      # A line looks like the following
      # <folder path> <number of trajectories>
      # If the second argument is not present, use all files in the folder indicated by -1

      # skip empty lines and lines starting with # are comments
      if len(line) == 0 or line[0] == '#':
        continue

      splits = line.split(' ')

      if len(splits) == 3:
        path = splits[0].strip()
        num_trajectories = int(splits[1])
        num_samples = int(splits[2])
      elif len(splits) == 2:
        path = splits[0].strip()
        num_trajectories = int(splits[1])
        num_samples = -1
      elif len(splits) == 1:
        path = splits[0].strip()
        num_trajectories = -1
        num_samples = -1
      else:
        raise ValueError('Invalid number of values in line {}: {}'.format(i + 1,
          line.strip()))

      if num_trajectories < 0:
        logger.info('Take all files from {}'.format(path))
      else:
        logger.info('Take {} files from {}'.format(num_trajectories, path))

      mixer.append((path, num_trajectories, num_samples))

    if len(mixer) == 0:
      raise ValueError('Mixer file did not contain any valid element')

  return mixer


def get_file_list(mixer_file, select_random, use_list_of_files):
  """
  Take a mixer file and return a list of .csv files
  """
  logger = logging.getLogger(get_file_list.__name__)
  files = list()

  if use_list_of_files:
    with open(mixer_file, 'r') as list_file:
      for line in list_file:
        files.append(os.path.join('data/raw', line.strip()))

      if select_random:
        random.shuffle(files)

  else:

    mixer = parse_mixer_file(mixer_file)

    for m in mixer:
      path = os.path.join(project_dir, m[0])
      all_mixer_files = [os.path.join(path, f) for f in os.listdir(path)
              if os.path.isfile(os.path.join(path, f)) and f.split('.')[-1] == 'csv']

      current_files = list()
      # Check if the number of samples is limited
      if m[2] >= 0:
        sample_count = 0
        for f in all_mixer_files:
          # Get number of lines without the header line
          num_lines = sum(1 for line in open(f)) - 1

          if (sample_count + num_lines) > m[2]:
            current_files.append((f, m[2] - sample_count))
            sample_count += (m[2] - sample_count)
            break
          else:
            current_files.append((f, -1))
            sample_count += num_lines

        if sample_count < m[2]:
          logger.warn('Not enough samples ({} < {}): {}'.format(sample_count, m[2], m[0]))
      else:
        # No limit, take all samples in the files
        current_files = zip(all_mixer_files, [-1] * len(all_mixer_files))

      if m[1] < 0:
        # -1 means all .csv files
        files += current_files
      elif m[1] > 0:
        if m[1] > len(current_files):
          logger.warn('Not enough files ({} < {}): {}'.format(len(current_files),
            m[1], m[0]))
        files += current_files[:m[1]]

    if select_random:
      random.shuffle(files)
    else:
      files = sorted(files, key=lambda x: int(os.path.basename(x[0]).split('_')[-1].split('.')[0]))

  return files


def main(project_dir):
  """
  Combine single .csv files of the trajectories into ONE .h5 file containing (a mixture) of datasets.
  """
  logger = logging.getLogger(__name__)
  logger.info('making final data set from raw data')

  args = parse_args()

  target_file = os.path.join(project_dir, 'data', 'processed', args.output_file)

  # Do not overwrite existing data containers
  if os.path.exists(target_file):
    logger.warn('Target file already exists: {}'.format(target_file))
    overwrite = raw_input('Overwrite (yN)?')
    if overwrite.lower() == 'y':
      logger.info('Overwrite file')
    else:
      logger.info('Abort! File was not generated')
      exit()

  # Generate a list with all the csv files in the input folders
  logger.info('Parse mixer file: {}'.format(args.mixer_file))
  all_files = get_file_list(os.path.join(project_dir, args.mixer_file), args.random, args.list)

  # Make shure, files in a sequence are not from the same source
  random.shuffle(all_files)

#   with pd.HDFStore(target_file, 'w') as store:
  store = pd.HDFStore(target_file, mode='w')
  num_elements = 0

  longest_filename = max([len(x) for (x, _) in all_files])
  # Iterate over all input files and add them to the HDF5 container
  for i, (f, samples) in enumerate(all_files):

    # Avoid trailing characters if the previous string was longer
    filler = ' ' * (longest_filename - len(f))

    current = pd.read_csv(f)
    print('({}/{}) \t{} \tlength: {}'.format(i, len(all_files), f, current.shape[0]) + filler, end='\r')

    # Make sure the final dataframe has a continous index
    current['original_index'] = current.index
    current.index = pd.Series(current.index) + num_elements
    current['target_id'] = i + 1

    # If samples is negative, append all rows to the file, otherwise up to 'samples'
    if samples < 0:
      store.append('data', current)
      num_elements += current.shape[0]
    else:
      store.append('data', current.iloc[:samples])
      num_elements += current.iloc[:samples].shape[0]

    print('')

  logger.info('We combined {} lines'.format(num_elements))
  logger.info('File saved: {}'.format(target_file))

  store.close()

  logger.info("Loading stored file for testing.")
  test_file_storer = pd.HDFStore(target_file, mode='r')
  logger.info("File successfully loaded with {} elements.".format(test_file_storer.get_storer('data').nrows))


if __name__ == '__main__':
  log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
  logging.basicConfig(level=logging.INFO, format=log_fmt)

  # not used in this stub but often useful for finding various files
  project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)

  main(project_dir)

