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
    parser.add_argument('--paths', nargs='+', help='List of bag files that should be analyzed', type=str, required=True)
    args = parser.parse_args()
    return args


save_figures = True
figure_path = '/home/pfmark/jade_catkin_ws/src/deep_motion_planning/planner_comparison/data/figures/'

fig_width_pt = 245.71811                # Get this from LaTeX using \showthe\columnwidth
inches_per_pt = 1.0/72.27               # Convert pt to inch
# golden_mean = (np.sqrt(5)-1.0)/2.0    # Aesthetic ratio
fig_width = fig_width_pt*inches_per_pt  # width in inches
fig_height = fig_width*1.0      # height in inches
fig_size =  [fig_width,fig_height]
fontsize = 9
params = {'backend': 'ps',
          'axes.labelsize': fontsize,
          'text.fontsize': fontsize,
          'title.fontsize': fontsize,
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
names = ['Deep', 'Ros']

planner_missions = []

for path in args.paths:
  planner_name = path.split('/')[-1][:-4] 
  rosbag_if = RosbagInterface(path)
  missions = extract_missions(rosbag_if.msg_container)
  planner_missions.append(missions)
  
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
for ii, missions in enumerate(planner_missions):
  for jj, m in enumerate(missions):
    plot_numbers = True if ii is 0 else False
    th = pc_util.plot_mission(ax, m, jj+1, color=colors[ii], plot_numbers=plot_numbers)
    if ii == 0:
      jh = pc_util.plot_joystick_interference(ax, m, color='m', alpha=1.0, linewidth=1.0)
      if len(jh) > 0:
        joystick_handle = jh
    if jj == 0:
      handles.append(th)
ax.set_xlabel('x [m]', fontsize=fontsize)
ax.set_ylabel('y [m]', fontsize=fontsize)
pl.axis('equal')
ax.set_xlim([-15.5, 0])
ax.set_ylim([-10.5, 6.5])
ax.tick_params(labelsize=fontsize)
pl.legend((handles[0][0], handles[1][0], joystick_handle[0][0]), (names[0], names[1], 'Joystick'), loc='best', fancybox=True, framealpha=0.5, numpoints=1)
pl.locator_params(nbins=8)
pl.tight_layout()
pl.subplots_adjust(left=0.17, right=0.95, top=0.95, bottom=0.15)

if save_figures:
  print('Saving figure.')
  pl.savefig(os.path.join(figure_path, 'comparison_real_robot.pdf'))

pl.show(block=False)