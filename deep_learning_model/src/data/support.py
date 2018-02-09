import sys
sys.path.insert(0, '/usr/local/lib/python2.7/dist-packages')
import logging
import numpy as np


def subsample_laser(laser_data, num_chunks):
  """
  Subsample laser measurements (take minimum of each chunk).
  Inputs:
    laser_data: raw laser measurements of size [batch_size, number measurements]
    num_chunks: number of chunks in which the laser data should be summarized
  """
  num_laser = laser_data.shape[1]

  values_per_chunk = int(num_laser / num_chunks)

  laser_min_chunks = np.zeros([laser_data.shape[0], num_chunks])

  for ii in range(num_chunks):
    laser_min_chunks[:, ii] = np.min(laser_data[:, ii*values_per_chunk:(ii+1)*values_per_chunk], axis=1)

  return laser_min_chunks


def crop_laser(laser_data, max_range=30.0):
  return np.minimum(laser_data, max_range)


def normalize_laser(laser_data, max_range=30):
  return laser_data / max_range

def invert_laser(laser_data):
  return 1 - laser_data

def transform_laser(laser_data, num_chunks=36, max_range=30.0):
  laser = subsample_laser(laser_data, num_chunks)
  laser = crop_laser(laser, max_range)
  laser = normalize_laser(laser, max_range)
  laser = invert_laser(laser)
  return 2*laser - 1


def transform_target_distance(target_distance, norm_range=30.0):
  tmp = np.minimum(target_distance, norm_range)
  tmp = 1 - tmp / norm_range
  return 2 * tmp - 1

def transform_target_angle(target_angle, norm_angle=np.pi):
  return target_angle / norm_angle






