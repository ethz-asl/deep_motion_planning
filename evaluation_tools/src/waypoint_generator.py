# import sys
# sys.path.insert(0, '/usr/local/lib/python2.7/dist-packages')
from grid_map import GridMap
import csv
import rospkg
import os
import numpy as np
import pandas as pd
import pylab as pl
import logging
import time
import argparse
from datetime import datetime

logger = logging.getLogger('waypoint_generator')
ch = logging.StreamHandler()
# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s %(name)s.%(levelname)s: %(message)s')
ch.setFormatter(formatter)
# add the handler to the logger
logger.addHandler(ch)
logger.setLevel(logging.INFO)

plotting = True

def sample_initial_pose(x_range, y_range, grid_map):
  cell_occupied = True
  while cell_occupied:
    x = np.random.uniform(x_range[0], x_range[1])
    y = np.random.uniform(y_range[0], y_range[1])
    yaw = np.random.uniform(0, 2*np.pi)
    cell_occupied = grid_map.area_occupied(np.array([x, y]), min_dist=0.4)

  return np.array([x, y, yaw])


def sample_final_pose(x_range, y_range, initial_pose, min_distance, grid_map):
  cell_occupied = True
  distance = 0
  while cell_occupied or distance < min_distance:
    x = np.random.uniform(x_range[0], x_range[1])
    y = np.random.uniform(y_range[0], y_range[1])
    yaw = np.random.uniform(0, 2*np.pi)
    cell_occupied = grid_map.area_occupied(np.array([x, y]), min_dist=0.4)
    distance = np.linalg.norm(np.array([x, y]) - initial_pose[0:2])

  return np.array([x, y, yaw])


def parse_args():
  parser = argparse.ArgumentParser(description='CNN autoencoder model training')

  parser.add_argument('--map_name', help='Name of the map. (default="simple").',
                      type=str, default='simple')
  parser.add_argument('--storage_path', help='Path where the final waypoints will be stored.', type=str)
  parser.add_argument('--number_trajectories', help='Number of trajectories (start-goal pairs)', type=int, default=100)

  args = parser.parse_args()

  return args

args = parse_args()
print(args)
print(" ")


map_name = args.map_name
base_path = rospkg.RosPack().get_path('stage_worlds')
yaml_path = os.path.join(base_path, "cfg", map_name + ".yaml")
map_path = os.path.join(base_path, "worlds/bitmaps/", map_name + ".png")

grid_map = GridMap(map_path, yaml_path)

n_trajectories = args.number_trajectories
x_range = [-5., 5.]
y_range = [-5., 5.]

initial_poses = []
final_poses = []

column_line = ['sample_number', 'start_x', 'start_y', 'start_yaw', 'final_x', 'final_y', 'final_yaw']

date_str = datetime.strftime(datetime.now(), '%Y-%m-%d_%H-%M-%S')
storage_path = args.storage_path
if not os.path.exists(storage_path):
  os.mkdir(storage_path)

output_file = open(os.path.join(storage_path, map_name + '_' + str(n_trajectories) + '.csv'), 'w')
writer= csv.writer(output_file, delimiter=',')
writer.writerow(column_line)

time_start = time.time()
for ii in range(n_trajectories):
  initial_pose = sample_initial_pose(x_range, y_range, grid_map)
  final_pose = sample_final_pose(x_range, y_range, initial_pose, min_distance=1., grid_map=grid_map)
  initial_poses.append(initial_pose)
  final_poses.append(final_pose)
  row_to_write = [ii+1, initial_pose[0], initial_pose[1], initial_pose[2],
                  final_pose[0], final_pose[1], final_pose[2]]
  writer.writerow(row_to_write)

output_file.flush()
output_file.close()

if plotting:
  pl.close('all')

  pl.figure()
  ax = pl.gca()
  for p in initial_poses:
    ax.plot(p[0], p[1], marker='s', alpha=0.8, color='g', lw=0)
  for p in final_poses:
    ax.plot(p[0], p[1], marker='s', alpha=0.8, color='g', lw=0)

  ax.set_aspect('equal')

pl.show(block=False)