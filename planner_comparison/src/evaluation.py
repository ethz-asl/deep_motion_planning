import rospy 
import rosbag
import rospkg
import argparse
import bisect
import numpy as np
import pylab as pl
import time
import logging

# Messages
from geometry_msgs.msg import Twist, PoseStamped, Point, Quaternion
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Path
from move_base_msgs.msg import MoveBaseActionFeedback

# Deep motion planner
from deep_motion_planner.tensorflow_wrapper import TensorflowWrapper 
from deep_motion_planner.deep_motion_planner import DeepMotionPlanner
import deep_motion_planner.util as dmp_util


class TimeMsgContainer():
  
  def __init__(self):
    self.times = []
    self.msgs = []
    
  def __len__(self):
    if (len(self.times) == len(self.msgs)):
      return len(self.times)
    else:
      raise Exception("There's something wrong with the vector lengths.")

def parse_args():
    """
    Parse input arguments.
    """
    parser = argparse.ArgumentParser(description='Compare different motion planners')
    parser.add_argument('-p', '--logPath', help='path to the logfile that should be analyzed', type=str, default="")
    parser.add_argument('-m', '--modelPath', help='path to the deep model', type=str, default=(rospkg.RosPack().get_path('deep_motion_planner'))+'/models/')
    parser.add_argument('-f', '--protobufFile', help='name of the protobuf file that comprises the model structure', type=str, default="model.pb")
    args = parser.parse_args()
    return args
  
def plot_velocities(ax_trans, ax_rot, velocities, color='b', linestyle='-', label=None):
  t_vec = [rt.to_sec() for rt in velocities.times]
  ax_trans.plot(t_vec, [v.linear.x for v in velocities.msgs], color=color, linestyle=linestyle, label=label)
  ax_rot.plot(t_vec, [v.angular.z for v in velocities.msgs], color=color, linestyle=linestyle, label=label)
  
def plot_trajectory(ax, loc_msgs, color='r', linestyle='-'):
  """
  Plot trajectory of the robot in Euclidean coordinates
  """
  x_vec = [l.feedback.base_position.pose.position.x for l in loc_msgs.msgs]
  y_vec = [l.feedback.base_position.pose.position.y for l in loc_msgs.msgs]
  ax.plot(x_vec, y_vec, color=color, linestyle=linestyle)
  ax.set_xlabel('x [m]')
  ax.set_ylabel('y [m]')

def get_closest_msg_in_past(t_desired, time_msg_list):
  idx_des = bisect.bisect(time_msg_list.times, t_desired)-1
  return time_msg_list.msgs[idx_des]


def compute_deep_plan(tensorflow_wrapper, scan_ranges, relative_target):
  """
  Compute the result of the deep motion planner for a specific timestamp
  """
  input_data = list(scan_ranges) + list(relative_target)
  linear_x, angular_z = tensorflow_wrapper.inference(input_data)
  cmd = Twist()
  cmd.linear.x = linear_x
  cmd.angular.z = angular_z
  return cmd

################## Setup #####################
pl.close('all')
args = parse_args()
plot_velocities_switch = True
plot_trajectory_switch = False
plot_errors_swtich = True
logger = logging.Logger('deep_evaluation', level=20) # INFO: 20 | DEBUG: 10
##############################################

# Get data
bag = rosbag.Bag(args.logPath)

# Get ros planner commands
vel_cmd_ros_msgs = TimeMsgContainer()
for topic, msg, t in bag.read_messages(topics=['/ros_planner/cmd_vel']):
  vel_cmd_ros_msgs.times.append(t)
  vel_cmd_ros_msgs.msgs.append(msg)

# Get deep planenr commands
vel_cmd_deep_msgs = TimeMsgContainer()
for topic, msg, t in bag.read_messages(topics=['/deep_planner/cmd_vel']):
  vel_cmd_deep_msgs.times.append(t)
  vel_cmd_deep_msgs.msgs.append(msg)

# Get scans 
scan_msgs = TimeMsgContainer()
for topic, msg, t in bag.read_messages(topics=['/base_scan']):
  scan_msgs.times.append(t)
  scan_msgs.msgs.append(msg)

# Get published goal positions 
goal_msgs = TimeMsgContainer()
for topic, msg, t in bag.read_messages(topics=['/move_base/current_goal']):
  goal_msgs.times.append(t)
  goal_msgs.msgs.append(msg)

  
# Get position/localization messages
loc_msgs = TimeMsgContainer()
for topic, msg, t in bag.read_messages(topics=['/move_base/feedback']):
  loc_msgs.times.append(t)
  loc_msgs.msgs.append(msg)


# Compute deep plans for timestamps
time_vec = vel_cmd_ros_msgs.times
vel_cmd_deep = TimeMsgContainer()
with TensorflowWrapper(args.modelPath, args.protobufFile, False) as tf_wrapper:
  t_start = time.time()
  for t in time_vec:
    # Get appropriate messages
    current_pos = get_closest_msg_in_past(t, loc_msgs)
    current_scan = get_closest_msg_in_past(t, scan_msgs)
    current_goal = get_closest_msg_in_past(t, goal_msgs)
    
    # Compute relative target (in robot frame)
    relative_target = dmp_util.compute_relative_target_raw(current_pos.feedback.base_position, current_goal)
    vel_cmd_deep.times.append(t)
    vel_cmd_deep.msgs.append(compute_deep_plan(tf_wrapper, scan_ranges=current_scan.ranges, relative_target=relative_target))
  print("Avg. model query time was {0} ms".format((time.time()-t_start) * 1000.0 / len(time_vec)))


# Run evaluation
vel_trans_diff = np.zeros([len(vel_cmd_deep),1])
vel_rot_diff = np.zeros([len(vel_cmd_deep),1])
cnt = 0
for v_ros, v_deep in zip(vel_cmd_ros_msgs.msgs, vel_cmd_deep.msgs):
  vel_trans_diff[cnt] = np.abs(v_ros.linear.x - v_deep.linear.x)
  vel_rot_diff[cnt] = np.abs(v_ros.angular.z - v_deep.angular.z) 
  cnt += 1
  
mean_trans_error = np.mean(vel_trans_diff)
std_trans_error = np.std(vel_trans_diff)
mean_rot_error = np.mean(vel_rot_diff)
std_rot_error = np.std(vel_rot_diff)

print("Translational velocities: ")
print("\tMean error: {0}".format(mean_trans_error))
print("\tStandard dev: {0}".format(std_trans_error))
print("\nRotational velocities: ")
print("\tMean error: {0}".format(mean_rot_error))
print("\tStandard dev: {0}".format(std_rot_error))

# Plotting 
if plot_velocities_switch:
  pl.figure('Velocity Command Comparison')
  ax_trans = pl.subplot(211)
  ax_rot = pl.subplot(212)
  plot_velocities(ax_trans, ax_rot, vel_cmd_ros_msgs, color='r', linestyle='-', label='ros')
#   plot_velocities(ax_trans, ax_rot, vel_cmd_deep_msgs, color='b', linestyle='-', label='deep_msg')
  plot_velocities(ax_trans, ax_rot, vel_cmd_deep, color='g', linestyle='-', label='deep')
  ax_trans.set_ylim([-1., 1.])
  ax_trans.set_ylabel('trans_vel [m/s]')
  ax_trans.grid('on')
  ax_rot.set_ylim([-1., 1.])
  ax_rot.set_ylabel('rot_vel [rad/s]')
  ax_rot.set_xlabel('time [s]')
  ax_rot.grid('on')

# Trajectory
if plot_trajectory_switch:
  pl.figure('Robot trajectory')
  ax = pl.subplot(111)
  plot_trajectory(ax, loc_msgs)
  ax.grid('on')
  
if plot_errors_swtich:
  pl.figure('Error plots')
  ax = pl.subplot(211)
  cl = 'r'
  ax.plot([t.to_sec() for t in time_vec], vel_trans_diff, color=cl)
  ax.plot([time_vec[0].to_sec(), time_vec[-1].to_sec()], [mean_trans_error, mean_trans_error], color=cl, linestyle='--', lw=0.5)
  ax.fill_between([time_vec[0].to_sec(), time_vec[-1].to_sec()], [mean_trans_error-std_trans_error, mean_trans_error-std_trans_error], 
                  [mean_trans_error+std_trans_error, mean_trans_error+std_trans_error], alpha=0.4, facecolor=cl)
  ax.set_ylabel('Diff vel_trans [m/s]')
  ax.grid('on')
  
  ax = pl.subplot(212)
  cl = 'b'
  ax.plot([t.to_sec() for t in time_vec], vel_rot_diff, color=cl)
  ax.plot([time_vec[0].to_sec(), time_vec[-1].to_sec()], [mean_rot_error, mean_rot_error], color=cl, linestyle='--', lw=0.5)
  ax.fill_between([time_vec[0].to_sec(), time_vec[-1].to_sec()], [mean_rot_error-std_rot_error, mean_rot_error-std_rot_error], 
                  [mean_rot_error+std_rot_error, mean_rot_error+std_rot_error], alpha=0.4, facecolor=cl)
  ax.set_ylabel('Diff vel_rot [rad/s]')
  ax.set_xlabel('time [s]')
  ax.grid('on')

pl.show(block=False)
