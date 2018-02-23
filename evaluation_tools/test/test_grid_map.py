import numpy as np
import rospkg
import os
import pylab as pl

import sys
sys.path.append('../src')
from grid_map import GridMap

pl.close('all')

base_path = rospkg.RosPack().get_path('stage_worlds')
yaml_path = os.path.join(base_path, "cfg/simple.yaml")
map_path = os.path.join(base_path, "worlds/bitmaps/simple.png")

grid_map = GridMap(map_path, yaml_path)

max_range = 5.
query_resolution = 0.1
x_range = [-max_range, max_range+query_resolution]
y_range = [-max_range, max_range+query_resolution]

x_values = np.arange(x_range[0], x_range[1], query_resolution)
y_values = np.arange(y_range[0], y_range[1], query_resolution)

occupancy_list = []

for xx in x_values.tolist():
  for yy in y_values.tolist():
    point_occupied = grid_map.is_occupied(np.array([xx, yy]))
    occupancy_list.append((xx, yy, point_occupied))

pl.figure()
ax = pl.gca()
for xx, yy, occ in occupancy_list:
  if occ:
    ax.plot(xx, yy, marker='s', alpha=0.8, color='r', lw=0)
  else:
    ax.plot(xx, yy, marker='s', alpha=0.8, color='g', lw=0)
ax.set_aspect('equal')
pl.show(block=False)