import rospy
import numpy 
import pylab as pl
import rospkg
import argparse
import os
from matplotlib import gridspec

from planner_comparison.plan_scoring import *
from planner_comparison.RosbagInterface import *
import planner_comparison.util as pc_util

def parse_args():
    """
    Parse input arguments.
    """
    parser = argparse.ArgumentParser(description='Compare different motion planners')
    parser.add_argument('--paths', nargs='+', help='List of bag files that should be analyzed', type=str, required=False)
    args = parser.parse_args()
    return args


save_figures = True
file_path = os.path.dirname(os.path.realpath(__file__))
figure_path = os.path.join(file_path, '..', 'data')
#figure_path = '/home/pfmark/jade_catkin_ws/src/deep_motion_planning/planner_comparison/data/figures/'

fig_width_pt = 245.71811                # Get this from LaTeX using \showthe\columnwidth
inches_per_pt = 1.0/72.27               # Convert pt to inch
# golden_mean = (np.sqrt(5)-1.0)/2.0    # Aesthetic ratio
fig_width = 5  # width in inches
fig_height = fig_width*1.0      # height in inches
fig_size =  [fig_width,fig_height]
fontsize = 9
params = {'backend': 'ps',
          'axes.labelsize': fontsize,
          'font.size': fontsize,
          'axes.titlesize': fontsize,
          'legend.fontsize': fontsize,
          'xtick.labelsize': fontsize,
          'ytick.labelsize': fontsize,
          'text.usetex': True,
          'font.family': 'serif',
          'font.serif': 'Computer Modern Roman',
          'figure.figsize': fig_size}
pl.rcParams.update(params)

args = parse_args()
pl.close('all')
pl.rc('text', usetex=True)
pl.rc('font', family='serif')
 
data_path = rospkg.RosPack().get_path('planner_comparison') + "/data/"
names = ['CNN\_bigFC', 'ROS']

paths_comparison = ['/data/rosbags/deep_motion_planning/turtle/deep_SO_4M_updated_targets.bag',
                    '/data/rosbags/deep_motion_planning/turtle/ros_planner.bag']

# paths_background = []
paths_background = ['/data/rosbags/deep_motion_planning/turtle/deep_SO_4M_2nd_run.bag',
                    '/data/rosbags/deep_motion_planning/turtle/deep_SO_4M_3rd_run.bag',
                    '/data/rosbags/deep_motion_planning/turtle/deep_SO_4M_4th_run.bag',
                    '/data/rosbags/deep_motion_planning/turtle/deep_SO_4M_5th_run.bag',
                    '/data/rosbags/deep_motion_planning/turtle/deep_SO_4M_6th_run.bag']

planner_missions = []
planner_missions_background = []

for path in paths_comparison:
  rosbag_if = RosbagInterface(path)
  missions = extract_missions(rosbag_if.msg_container)
  planner_missions.append(missions)
  
for idx, path in enumerate(paths_background):
  print('Extracting background path {0}'.format(idx+1))
  rosbag_if = RosbagInterface(path)
  missions = extract_missions(rosbag_if.msg_container)
  planner_missions_background.append(missions)


shift_numbers = ['rd', 'r', 'u', 'l', 'u', 'u', 'u', 'r', 'u', 'rd', 'u', 'u', 'r']
colors = ['g', 'k']
manual_offset_x = 0.25
manual_offset_y = 0.25
map = rosbag_if.msg_container['map'].msgs[0]
grid = np.reshape(map.data, [map.info.height, map.info.width])
pl.figure('Missions')
ax = pl.subplot(111)
map_size = [map.info.width*map.info.resolution/2, map.info.height*map.info.resolution/2]
map_offset = [map.info.origin.position.x * map.info.resolution + manual_offset_x, map.info.origin.position.y * map.info.resolution + manual_offset_y]
pl.imshow(grid, extent=[-map_size[0] + map_offset[0], map_size[0] + map_offset[0], -map_size[1] + map_offset[1], map_size[1] + map_offset[1]], origin='lower', cmap='Greys')
handles = []
joystick_handle = None
for missions in planner_missions_background:
  for m in missions:
    pc_util.plot_mission(ax, m, None, color='g', plot_numbers=False, alpha=0.3)
    pc_util.plot_joystick_interference(ax, m, color='m', alpha=0.3, linewidth=1.0)
for ii, missions in enumerate(planner_missions):
  for jj, m in enumerate(missions):
    plot_numbers = True if ii is 0 else False
    th = pc_util.plot_mission(ax, m, jj+1, color=colors[ii], plot_numbers=plot_numbers, shift_direction=shift_numbers[jj], shift_dist=0.7)
    if ii == 0:
      jh = pc_util.plot_joystick_interference(ax, m, color='m', alpha=1.0, linewidth=1.0)
      if len(jh) > 0:
        joystick_handle = jh
    if jj == 0:
      handles.append(th)
ax.set_xlabel('x [m]', fontsize=fontsize)
ax.set_ylabel('y [m]', fontsize=fontsize)
pl.axis('equal')
ax.set_xlim([-15, 6])
ax.set_ylim([-8, 2])
ax.tick_params(labelsize=fontsize)
pl.legend((handles[0][0], handles[1][0], joystick_handle[0][0]), (names[0], names[1], 'Joystick'), loc='best', fancybox=True, framealpha=0.5, numpoints=1)
pl.locator_params(nbins=8)
pl.tight_layout()
pl.subplots_adjust(left=0.17, right=0.95, top=0.95, bottom=0.15)


abs_dist = np.zeros([len(planner_missions[0])])
joy_dist = np.zeros([len(planner_missions[0])])
fused_missions = planner_missions_background
fused_missions.append(planner_missions[0])
for miss in fused_missions:
  for ii, m in enumerate(miss):
    abs_dist[ii] += m.distance()
    joy_dist[ii] += pc_util.compute_joystick_distance(m)

autonomous_ratio = 100 * (1 - joy_dist / abs_dist)
print('Autonomous ratio: {}'.format(autonomous_ratio))

if save_figures:
  print('Saving figure.')
  pl.savefig(os.path.join(figure_path, 'comparison_real_robot.pdf'))

pl.show(block=False)
