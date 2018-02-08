import sys
sys.path.insert(0, '/usr/local/lib/python2.7/dist-packages')
import logging
import numpy as np


def subsample_laser(laser_data, num_chunks):
  """
  Subsample laser measurements (take minimum of each chunk).
  """
  logger = logging.getLogger("subsample_laser")
  num_laser = laser_data.shape[1]

  values_per_chunk = int(num_laser / num_chunks)

  laser_min_chunks = np.zeros([laser_data.shape[0], num_chunks])

  for ii in range(num_chunks):
    laser_min_chunks[:, ii] = np.min(laser_data[:, ii*values_per_chunk:(ii+1)*values_per_chunk], axis=1)


  return laser_min_chunks