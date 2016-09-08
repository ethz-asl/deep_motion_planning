import rospy
import numpy 
import pylab as pl
import rospkg
import argparse
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

args = parse_args()
pl.close('all')
pl.rc('text', usetex=False)
pl.rc('font', family='serif')
fontsize = 14
 
data_path = rospkg.RosPack().get_path('planner_comparison') + "/data/"
names = ['deep', 'ros']

planner_missions = {}

for path in args.paths:
  planner_name = path.split('/')[-1][:-4] 
  rosbag_if = RosbagInterface(path)
  missions = extract_missions(rosbag_if.msg_container)
  planner_missions[planner_name] = missions
  
colors = ['k', 'r']  
map = rosbag_if.msg_container['map'].msgs[0]
grid = np.reshape(map.data, [map.info.height, map.info.width])
pl.figure('Missions')
gs = gridspec.GridSpec(2, 1, width_ratios=[1, 1], height_ratios=[3, 1])
ax = pl.subplot(gs[0])
pl.imshow(grid, extent=[-5, 5, -5, 5], origin='lower', cmap='Greys')
for ii, name in enumerate(planner_missions.keys()):
  for jj, m in enumerate(planner_missions[name]):
    plot_numbers = True if ii is 0 else False
    pc_util.plot_mission(ax, m, jj, color=colors[ii], plot_numbers=plot_numbers)
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')

ax = pl.subplot(gs[1])
bar_width = 0.2
bar_handles = []
for ii, name in enumerate(planner_missions.keys()):
  cost = pc_util.compute_detailed_cost_sum(planner_missions[name])
  shift = -bar_width/2 if ii is 0 else bar_width/2
  bars = pc_util.plot_error_bars(ax, cost, color=colors[ii], shift=shift, bar_width=bar_width)
  bar_handles.append(bars[0])
ax.set_ylabel('Error [-]')
x_tick_pos = np.arange(len(cost)) + 1
pl.xticks(x_tick_pos, cost.keys(), rotation=45, fontsize=9)
pl.tight_layout()
pl.legend((bar_handles[0], bar_handles[1]), (names[0], names[1]), loc='best', fancybox = True, framealpha = 0.5, fontsize=fontsize)



pl.show(block=False)