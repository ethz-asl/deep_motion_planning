import pandas as pd
import numpy as np
import pylab as pl

pl.close('all')

data = pd.read_csv('../data/2018-02-13_14-39-09/target_1.csv')


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

fig_target = pl.figure('target global')
# Position x
ax = pl.subplot(311)
ax.plot(data.robot_pose_global_frame_x)
ax.plot(data.target_global_frame_x)

# Position y
ax = pl.subplot(312)
ax.plot(data.robot_pose_global_frame_y)
ax.plot(data.target_global_frame_y)

# Heading
ax = pl.subplot(313)
ax.plot(data.robot_pose_global_frame_yaw)
ax.plot(data.target_global_frame_yaw)


fig_target = pl.figure('target robot frame')
# Position x
ax = pl.gca()
ax.plot(data.target_x, label='rel_x')
ax.plot(data.target_y, label='rel_y')
ax.plot(data.target_yaw, label='rel_yaw')
pl.legend(loc='best')


pl.show(block=False)