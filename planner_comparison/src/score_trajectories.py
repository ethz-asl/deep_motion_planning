import rospy
import numpy 
import pylab as pl
import rospkg
import argparse

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
filenames = []

planner_missions = {}

for path in args.paths:
  planner_name = path.split('/')[-1][:-4] 
  rosbag_if = RosbagInterface(path)
  missions = extract_missions(rosbag_if.msg_container)
  planner_missions[planner_name] = missions
  
  
# Plotting
pl.figure('Error Plot')
ax = pl.subplot(111)
width = 0.2
colors = ['r', 'b', 'g', 'k', 'c']
for idx, planner_name in enumerate(planner_missions.keys()):
  cost =  np.sum([m.compute_cost()[0] for m in planner_missions[planner_name]])
  cost_bar = ax.bar(idx+1, cost, width=width, color=colors[idx], align='center')
ax.set_ylabel('Cost [-]', fontsize=fontsize)
ax.grid('on')
x_tick_pos = np.arange(len(planner_missions)) + 1
pl.xticks(x_tick_pos, planner_missions.keys(), rotation=45, fontsize=fontsize)
pl.tight_layout()

pl.show(block=False)