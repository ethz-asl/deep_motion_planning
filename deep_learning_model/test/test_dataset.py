import pandas as pd
import numpy as np
import logging
import time
import pylab as pl
import sys
sys.path.append('../src/data')
import support as sup

pl.close('all')

filename = '../data/processed/office_250.h5'
data_store = pd.HDFStore(filename, mode='r')
data = pd.read_hdf(filename, 'data', mode='r')

laser_columns = []
goal_columns = []
odom_vel_columns = []
robot_pose_global_frame_columns = []
target_global_frame_columns = []

for j,column in enumerate(data.columns):
  if column.split('_')[0] in ['laser']:
    laser_columns.append(j)
  if column.split('_')[0] in ['target'] and not column.split('_')[1] == 'id':
    goal_columns.append(j)
  if 'robot_pose_global_frame' in column:
    robot_pose_global_frame_columns.append(j)
  if 'target_global_frame' in column:
    target_global_frame_columns.append(j)

batch_size = 8
n_laser_chunks = 72
n_laser = len(laser_columns)
meas_per_chunk = int(n_laser / n_laser_chunks)

laser = data.iloc[0:batch_size,laser_columns].values
goal = data.iloc[0:batch_size,goal_columns].values

t_start = time.time()
laser_sub = sup.subsample_laser(laser, n_laser_chunks)
print("Subsampling took {} ms.".format((time.time()-t_start) * 1000))

trans_goal_dist = sup.transform_target_distance(np.expand_dims(goal[:,0], axis=1), norm_range=30.0)
trans_goal_angle = sup.transform_target_angle(np.expand_dims(goal[:,1], axis=1), norm_angle=np.pi)

pl.figure('laser')
ax = pl.gca()
ax.plot(np.arange(0, n_laser, 1), laser[0,:])

for ii in range(n_laser_chunks):
  ax.fill_between([ii*meas_per_chunk, (ii+1)*meas_per_chunk], [0, 0], [laser_sub[0,ii]]*2, color='r', alpha=0.3)



print("Original goal distance: {}".format(goal[0,0]))
print("Transformed goal distance: {}".format(trans_goal_dist[0,0]))
print(" ")
print("Original goal angle: {}".format(goal[0,1]))
print("Transformed goal angle: {}".format(trans_goal_angle[0,0]))

pl.show(block=False)




