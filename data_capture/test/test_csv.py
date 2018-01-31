import pandas as pd
import numpy as np
import pylab as pl

pl.close('all')

data = pd.read_csv('../data/2018-01-31_15-18-05/target_1.csv')


laser_data = data.loc[:, 'laser_1':'laser_1079']
fig_laser = pl.figure('laser measurements')
ax = pl.gca()
for ii in range(100):
  ax.plot(laser_data.loc[ii, :])
  
fig_velocity = pl.figure('velocities')
ax = pl.gca()
ax.plot(data.linear_x_command, label='trans_vel_command')
ax.plot(data.linear_x_odom, label='trans_vel_odom')
ax.plot(data.angular_z_command, label='rot_vel_command')
ax.plot(data.angular_z_odom, label='rot_vel_odom')
pl.legend(loc='upper left')


pl.show(block=False)