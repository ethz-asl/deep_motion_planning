import pickle
import rospy
import rospkg
import os
import pylab as pl
import numpy as np

from planner_comparison.time_msg_container import *

# Messages
from geometry_msgs.msg import Twist, PoseStamped, Point, Quaternion
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Path
from move_base_msgs.msg import MoveBaseActionFeedback

class ErrorValues():
  
   def __init__(self, name):
     self.rot_mean_training = 0.0
     self.trans_mean_training = 0.0
     self.rot_mean_eval = 0.0
     self.trans_mean_eval = 0.0
     self.rot_std_training = 0.0
     self.trans_std_training = 0.0
     self.rot_std_eval = 0.0
     self.trans_std_eval = 0.0
     self.name = name

pl.close('all')
figure_path = '/home/pfmark/jade_catkin_ws/src/deep_motion_planning/planner_comparison/data/figures/'
fig_width_pt = 245.71811                # Get this from LaTeX using \showthe\columnwidth
inches_per_pt = 1.0/72.27               # Convert pt to inch
# golden_mean = (np.sqrt(5)-1.0)/2.0    # Aesthetic ratio
fig_width = fig_width_pt*inches_per_pt  # width in inches
fig_height = fig_width*0.7      # height in inches
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

data_path = rospkg.RosPack().get_path('planner_comparison') + "/data/"
filenames = ['simple.pkl', 'shapes.pkl', 'office.pkl']
names = ['training', 'eval_{simple}', 'eval_{complex}']

data_storage = {}
for idx, name in enumerate(filenames): 
  data_storage[names[idx]] = pickle.load(open(data_path + name, 'rb'))
  
# Boxplots 
box_width = 0.2
box_dist = 0.1
pl.figure('Boxplots error')
ax = pl.subplot(111)
boxplot_data_trans = []
boxplot_data_rot = []
boxplot_positions_trans = []
boxplot_positions_rot = []

bps_trans = []
bps_rot = []
for idx, data in enumerate(data_storage.values()):
  bps_trans.append(ax.boxplot(data['vel_trans_diff'], positions=[idx + 1 - box_width/2 - box_dist/2]))
  bps_rot.append(ax.boxplot(data['vel_rot_diff'], positions=[idx + 1 + box_width/2 + box_dist/2]))
   
color_fill = 'r'
color_median = 'm'
for bp in bps_trans:
  pl.setp(bp['boxes'], color=color_fill, lw=1.0, alpha=1.0)
  pl.setp(bp['whiskers'], color=color_fill, ls='-')
  pl.setp(bp['fliers'], marker='o', markerfacecolor=color_fill, markersize=1.0, alpha=1.0, markeredgecolor=color_fill, markeredgewidth=0.1)
  pl.setp(bp['caps'], color=color_fill)
  pl.setp(bp['medians'], color=color_median, lw=1.5)
   
color_fill = 'b'
color_median = 'c'
for bp in bps_rot:
  pl.setp(bp['boxes'], color=color_fill, lw=1.0, alpha=1.0)
  pl.setp(bp['whiskers'], color=color_fill, ls='-')
  pl.setp(bp['fliers'], marker='o', markerfacecolor=color_fill, markersize=1.0, alpha=1.0, markeredgecolor=color_fill, markeredgewidth=0.1)
  pl.setp(bp['caps'], color=color_fill)
  pl.setp(bp['medians'], color=color_median, lw=1.5)
             
ax.set_xlim(0, len(names)+1)
ax.set_ylabel('[m/s] / [rad/s]')
x_tick_pos = np.arange(len(names)) + 1
pl.xticks(x_tick_pos, names, rotation=0, fontsize=fontsize)
h = []
h.append(ax.plot([],color='r', lw=7, label='trans. error'))
h.append(ax.plot([],color='b', lw=7, label='rot. error'))
ax.legend(loc='upper left', fancybox = True, framealpha = 0.5)
pl.subplots_adjust(left=0.17, right=0.95, top=0.92, bottom=0.3)

print('Saving figure.')
pl.savefig(os.path.join(figure_path, 'error_boxplots.pdf'))

pl.show(block=False)
  
  