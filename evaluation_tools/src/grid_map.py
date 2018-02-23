import numpy
import rospkg
import cv2
import pylab as pl
import numpy as np
import yaml
import os


class GridMap():

  def __init__(self, map_file, yaml_file):
    self.grid_array = cv2.imread(os.path.join(map_file), 0)
    with open(yaml_file) as stream:
      self.yaml_data = yaml.load(stream)
    self.origin = np.array([self.yaml_data['origin'][0], self.yaml_data['origin'][1]])
    self.resolution = self.yaml_data['resolution']
    self.grid_dimensions = self.grid_array.shape

  def get_coordinate_as_cv_idx(self, coord):
    coord_centered = coord - self.origin  # adjust to center
    coord_idx = np.floor(coord_centered / self.resolution)  # convert metric coordinate to index
    coord_idx_cv = coord_idx
    coord_idx_cv[1] = self.grid_dimensions[1] - coord_idx[1]  # invert y axis for open CV coordinate frame
    coord_idx_cv[0] = int(np.minimum(self.grid_dimensions[0]-1, coord_idx_cv[0]))
    coord_idx_cv[1] = int(np.minimum(self.grid_dimensions[1]-1, coord_idx_cv[1]))

    return coord_idx_cv

  def is_occupied(self, metric_coordinate):
    idx_coord = self.get_coordinate_as_cv_idx(metric_coordinate)
    return self.grid_array[idx_coord[1], idx_coord[0]] < 255

  def area_occupied(self, center_coordinate, min_dist=0.3):
    """
    Query if any point in an rectangular area around center_coordinate is occupied.
    (min_dist comprises safety distance and robot radius)
    """
    idx_dist = int(np.ceil(min_dist / self.resolution))
    center_idx = self.get_coordinate_as_cv_idx(center_coordinate)
    x_low_idx = int(np.maximum(0, center_idx[0] - idx_dist))
    x_high_idx = int(np.minimum(self.grid_dimensions[0]-1, center_idx[0] + idx_dist))
    y_low_idx = int(np.maximum(0, center_idx[1] - idx_dist))
    y_high_idx = int(np.minimum(self.grid_dimensions[1]-1, center_idx[1] + idx_dist))

    return np.min(self.grid_array[y_low_idx:y_high_idx, x_low_idx:x_high_idx]) < 255
