import rospy
import numpy 
import pylab as pl
import rospkg
import argparse
from matplotlib import gridspec
import matplotlib.ticker as ticker

from planner_comparison.plan_scoring import *
from planner_comparison.RosbagInterface import *
import planner_comparison.util as pc_util

def parse_args():
    """
    Parse input arguments.
    """
    parser = argparse.ArgumentParser(description='Compare different motion planners')
    parser.add_argument('--paths1', nargs='+', help='List of bag files of first map that should be analyzed (deep + ros)', type=str, required=True)
    parser.add_argument('--paths2', nargs='+', help='List of bag files of first map that should be analyzed (deep + ros)', type=str, required=False, default=[])
    args = parser.parse_args()
    return args
  
def compute_relative_cost_deep(cost_deep, cost_ros):
  if cost_ros > 0:
    return (cost_deep - cost_ros) / cost_ros
  else:
    return cost_deep

def plot_trajectory_comparison(ax, deep_missions, ros_missions, grid, colors=['g', 'k'], show_labels=False):
  pl.imshow(grid, extent=[-5, 5, -5, 5], origin='lower', cmap='Greys')
  for idx, m in enumerate(deep_missions):
    dh = pc_util.plot_mission(ax, m, idx+1, color=colors[0], plot_numbers=True)
  for idx, m in enumerate(ros_missions):
    rh = pc_util.plot_mission(ax, m, idx+1, color=colors[1], plot_numbers=False)
  if show_labels:
    pl.legend((dh[0], rh[0]), ('Deep', 'ROS'), fancybox=True, framealpha=0.5)
    
def plot_relative_error(ax, deep_missions, ros_missions, colors=['g', 'k'], show_labels=False):
  cost_deep = pc_util.compute_detailed_cost_sum(deep_missions)
  cost_ros = pc_util.compute_detailed_cost_sum(ros_missions)
  cost_rel = cost_deep
  for key in cost_deep.keys():
    cost_rel[key] = compute_relative_cost_deep(cost_deep[key], cost_ros[key])
  pc_util.plot_relative_error_bars(ax, cost_rel, colors=colors, bar_width=0.3)
  h = []
  if show_labels:
    h.append(ax.plot([],color=colors[0], lw=7, label='Deep'))
    h.append(ax.plot([],color=colors[1], lw=7, label='ROS'))
  return h
  
args = parse_args()
pl.close('all')
fig_width_pt = 245.71811                # Get this from LaTeX using \showthe\columnwidth
inches_per_pt = 1.0/72.27               # Convert pt to inch
# golden_mean = (np.sqrt(5)-1.0)/2.0    # Aesthetic ratio
fig_width = fig_width_pt*inches_per_pt  # width in inches
fig_height = fig_width*0.9      # height in inches
fig_size =  [fig_width,fig_height]
fontsize = 9
params = {'backend': 'ps',
          'axes.labelsize': fontsize,
          'text.fontsize': fontsize,
          'title.fontsize': fontsize,
          'legend.fontsize': fontsize,
          'xtick.labelsize': fontsize,
          'ytick.labelsize': fontsize,
          'text.usetex': False,
          'font.family': 'serif',
          'font.serif': 'Computer Modern Roman',
          'figure.figsize': fig_size}
pl.rcParams.update(params)
figure_path = '/home/pfmark/jade_catkin_ws/src/deep_motion_planning/planner_comparison/data/figures/'
 
data_path = rospkg.RosPack().get_path('planner_comparison') + "/data/"
names = ['deep', 'ros']

planner_missions1 = []
planner_missions2 = []

for path in args.paths1:
  rosbag_if = RosbagInterface(path)
  missions = extract_missions(rosbag_if.msg_container)
  planner_missions1.append(missions)
  map = rosbag_if.msg_container['map'].msgs[0]
  grid1 = np.reshape(map.data, [map.info.height, map.info.width])
  
two_maps = len(args.paths2) > 0
if two_maps:
  for path in args.paths2:
    rosbag_if = RosbagInterface(path)
    missions = extract_missions(rosbag_if.msg_container)
    planner_missions2.append(missions)
    map = rosbag_if.msg_container['map'].msgs[0]
    grid2 = np.reshape(map.data, [map.info.height, map.info.width])
  
colors = ['g', 'k']  
map = rosbag_if.msg_container['map'].msgs[0]
grid = np.reshape(map.data, [map.info.height, map.info.width])
pl.figure('Missions')

if two_maps:
  ax = pl.subplot2grid((5,2), (0,0), rowspan=3)
else:
  ax = pl.subplot2grid((5,1), (0,0), rowspan=3)
plot_trajectory_comparison(ax, planner_missions1[0], planner_missions1[1], grid1, colors=['g', 'k'], show_labels=False)
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])

if two_maps:
  ax = pl.ax = pl.subplot2grid((5,2), (3,0), rowspan=2)
else:
  ax = pl.ax = pl.subplot2grid((5,1), (3,0), rowspan=2)
plot_relative_error(ax, planner_missions1[0], planner_missions1[1], colors=['g', 'k'], show_labels=True)
ax.legend(loc='best', fancybox = True, framealpha = 0.5)
ax.yaxis.set_major_locator(ticker.MultipleLocator(30))
ax.yaxis.grid()

if two_maps:
  ax = pl.ax = pl.subplot2grid((5,2), (0,1), rowspan=3)
  plot_trajectory_comparison(ax, planner_missions2[0], planner_missions2[1], grid2, colors=['g', 'k'], show_labels=False)
  ax.get_xaxis().set_ticks([])
  ax.get_yaxis().set_ticks([])

  ax = pl.ax = pl.subplot2grid((5,2), (3,1), rowspan=2)
  plot_relative_error(ax, planner_missions2[0], planner_missions2[1], colors=['g', 'k'], show_labels=False)
  ax.legend(loc='best', fancybox = True, framealpha = 0.5)
  ax.yaxis.set_major_locator(ticker.MultipleLocator(30))
  ax.yaxis.grid()



pl.subplots_adjust(left=0.10, right=0.97, top=0.97, bottom=0.17, wspace=None, hspace=None)
pl.tight_layout()
pl.show(block=False)

print('Saving figure.')
pl.savefig(os.path.join(figure_path, 'trajectory_comparison.pdf'))
