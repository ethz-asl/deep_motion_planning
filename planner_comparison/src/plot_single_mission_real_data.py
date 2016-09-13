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
figure_path = '/home/pfmark/jade_catkin_ws/src/deep_motion_planning/planner_comparison/data/figures/'

fig_width_pt = 245.71811                # Get this from LaTeX using \showthe\columnwidth
inches_per_pt = 1.0/72.27               # Convert pt to inch
# golden_mean = (np.sqrt(5)-1.0)/2.0    # Aesthetic ratio
fig_width = fig_width_pt*inches_per_pt  # width in inches
fig_height = fig_width*3      # height in inches
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
bagfile = '/data/rosbags/deep_motion_planning/turtle/deep_SO_4M_updated_targets.bag'

rosbag_if = RosbagInterface(bagfile)
missions = extract_missions(rosbag_if.msg_container)
  
color = 'g'
idx_mission = 1
manual_offset_x = 0.25
manual_offset_y = 0.25
map = rosbag_if.msg_container['map'].msgs[0]
grid = np.reshape(map.data, [map.info.height, map.info.width])
pl.figure('Missions')
ax = pl.subplot(111)
map_size = [map.info.width*map.info.resolution/2, map.info.height*map.info.resolution/2]
map_offset = [map.info.origin.position.x * map.info.resolution + manual_offset_x, map.info.origin.position.y * map.info.resolution + manual_offset_y]
pl.imshow(grid, extent=[-map_size[0] + map_offset[0], map_size[0] + map_offset[0], -map_size[1] + map_offset[1], map_size[1] + map_offset[1]], origin='lower', cmap='Greys')
pc_util.plot_mission(ax, missions[idx_mission], None, color='g', plot_numbers=False, alpha=1, linewidth=2)
traj = missions[idx_mission].get_trajectory()
ax.plot(traj[0,0], traj[1,0], marker='o', mfc='r', mec='r')
pl.axis('equal')
ax.set_xlim([-0.5, 2.5])
ax.set_ylim([-6, 0])
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])
pl.tight_layout()
pl.subplots_adjust(left=0.17, right=0.95, top=0.95, bottom=0.15)

if save_figures:
  print('Saving figure.')
  pl.savefig(os.path.join(figure_path, 'single_traj.pdf'))

pl.show(block=False)