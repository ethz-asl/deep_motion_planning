import pickle
import rospy
import rospkg


import pylab as pl
import numpy as np

from time_msg_container import *

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

# pl.close('all')
pl.rc('text', usetex=True)
pl.rc('font', family='serif')
fontsize = 14

data_path = rospkg.RosPack().get_path('planner_comparison') + "/data/"
filenames_training = ['FC_BS128_S500000_noDrop_LRconst_training.pkl','FC_BS128_S500000_LRconst_training.pkl', 'FC_BS128_S500000_REG_LRconst_training.pkl', 'FC_4Layers_BS128_S2000000_REG_LRconst_training.pkl']
filenames_eval = ['FC_BS128_S500000_noDrop_LRconst_eval.pkl','FC_BS128_S500000_LRconst_eval.pkl', 'FC_BS128_S500000_REG_LRconst_eval.pkl', 'FC_4Layers_BS128_S2000000_REG_LRconst_eval.pkl']
names = ['fully connected','dropout', 'dropout+reg.', '4 Layers']

data_storage_training = []
data_storage_eval = []

for name in filenames_training:
  data_storage_training.append(pickle.load(open(data_path + name, 'rb')))
  
for name in filenames_eval:
  data_storage_eval.append(pickle.load(open(data_path + name, 'rb')))

err_container = []

for idx, data in enumerate(data_storage_training):
  err = ErrorValues(names[idx])
  # Training data
  err.rot_mean_training = data_storage_training[idx]['rot_error'][0]
  err.rot_std_training = data_storage_training[idx]['rot_error'][1]
  err.trans_mean_training = data_storage_training[idx]['trans_error'][0]
  err.trans_std_training = data_storage_training[idx]['trans_error'][1]
  
  # Evaluation data
  err.rot_mean_eval = data_storage_eval[idx]['rot_error'][0]
  err.rot_std_eval = data_storage_eval[idx]['rot_error'][1]
  err.trans_mean_eval = data_storage_eval[idx]['trans_error'][0]
  err.trans_std_eval = data_storage_eval[idx]['trans_error'][1]
  
  err_container.append(err)
  
pl.figure('Error Plot')
ax = pl.subplot(111)
width = 0.2
for idx,err in enumerate(err_container):
  train_bar_trans = ax.bar(idx+1-width/2, err.trans_mean_training, width=width, color='b', align='center')
  train_bar_rot = ax.bar(idx+1-width/2, err.rot_mean_training, width=width, color='b', alpha= 0.6, align='center', bottom=err.trans_mean_training)
  eval_bar_trans = ax.bar(idx+1+width/2, err.trans_mean_eval, width=width, color='r', align='center')
  eval_bar_rot = ax.bar(idx+1+width/2, err.rot_mean_eval, width=width, color='r', alpha=0.6, align='center', bottom=err.trans_mean_eval)
ax.set_ylabel('Error [-]', fontsize=fontsize)
ax.grid('on')
x_tick_pos = np.arange(len(err_container)) + 1
pl.xticks(x_tick_pos, names, rotation=45, fontsize=fontsize)
pl.legend((train_bar_trans, train_bar_rot, eval_bar_trans, eval_bar_rot), 
          ('training map trans.', 'training map rot.', 'eval map trans.', 'eval map rot.'), loc='best', fancybox = True, framealpha = 0.5, fontsize=fontsize)
pl.tight_layout()

pl.show(block=False)
  
  