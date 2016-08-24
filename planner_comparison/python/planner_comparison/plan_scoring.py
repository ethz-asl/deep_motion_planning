import rospy 
import bisect
import numpy as np
import time
import logging
import math
from time_msg_container import *
from plan_scoring import *

# Messages
from geometry_msgs.msg import Twist, PoseStamped, Point, Quaternion
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Path, Odometry
from move_base_msgs.msg import MoveBaseActionFeedback
from std_msgs.msg import Empty


class Mission():
  
  def __init__(self):
    self.loc_msgs = TimeMsgContainer()
    self.odom_msgs = TimeMsgContainer()
    self.vel_cmd_msgs = TimeMsgContainer()
    self.scan_msgs = TimeMsgContainer()
    self.goal = PoseStamped
    self.start_time = rospy.Time()
    self.end_time = rospy.Time()
      
    self.params = {'threshold_range': 0.2, 
                   'rotation_energy_weight': 3.0,
                   'translation_energy_weight': 1.0}
    
  def duration(self):
    return (self.end_time - self.start_time).to_sec()
    
  def obstacle_closeness(self):
    cost = 0.0
    if not self.params['threshold_range']:
      raise('Please define a threshold range for the object closeness feature.')
    for scan in self.scan_msgs.msgs:
      min_range = min(scan.ranges)
      if min_range < self.params['threshold_range']:
        cost += 1/min_range
    return cost

  def distance(self):
    dist = 0.0
    position_prev = self.odom_msgs.msgs[0].pose.pose.position
    for ii in range(1, len(self.odom_msgs)):
      position_new = self.odom_msgs.msgs[ii].pose.pose.position
      dist += math.hypot(position_new.x - position_prev.x, position_new.y - position_prev.y)
      position_prev = position_new
    return dist
  
  def avg_speed(self):
    if self.duration() > 0.0:
      return self.distance()/self.duration()
    else: 
      return 0.0
    
  def acceleration_vector(self):
    acc_trans = np.zeros([len(self.odom_msgs)])
    acc_rot = np.zeros([len(self.odom_msgs)])
    t_prev = self.odom_msgs.times[0]
    v_trans_prev = math.hypot(self.odom_msgs.msgs[0].twist.twist.linear.x, self.odom_msgs.msgs[0].twist.twist.linear.y)
    v_rot_prev = self.odom_msgs.msgs[0].twist.twist.angular.z
    for ii in range(1,len(self.odom_msgs)):
      t = self.odom_msgs.times[ii]
      v_trans = math.hypot(self.odom_msgs.msgs[ii].twist.twist.linear.x, self.odom_msgs.msgs[ii].twist.twist.linear.y)
      v_rot = self.odom_msgs.msgs[ii].twist.twist.angular.z
      t_diff = (t - t_prev).to_sec()
      acc_trans[ii] = (v_trans - v_trans_prev) / t_diff
      acc_rot[ii] = (v_rot - v_rot_prev) / t_diff
    return acc_trans, acc_rot
  
  def energy(self):
    """
    acceleration based energy during mission
    """
    acc_trans, acc_rot = self.acceleration_vector()
    return self.params['translation_energy_weight'] * np.sum(np.abs(acc_trans)) + self.params['rotation_energy_weight'] * np.sum(np.abs(acc_trans))

  def normalized_energy(self):
    return self.energy()/self.duration()
  
  def final_goal_dist(self):
    return math.hypot(self.odom_msgs.msgs[-1].pose.pose.position.x - self.goal.pose.position.x, 
                      self.odom_msgs.msgs[-1].pose.pose.position.y - self.goal.pose.position.y)

  def compute_mission_cost(self):
    feature_list = [self.energy, self.object_closeness, self.final_goal_dist, self.avg_speed]
    feature_weights = [1] * len(feature_list)
    
    cost = 0.0
    for f, w in zip(feature_list, feature_weights):
      cost += w * f()
      
    return cost
    

def adjust_start_stop_msgs(start_msgs, stop_msgs):
  # All trajectories are complete
  start_adjusted = TimeMsgContainer()
  stop_adjusted = TimeMsgContainer()
  # recording started during driving and ended after end of mission -> delete first stop message
  if len(start_msgs) < len(stop_msgs) and start_msgs.times[0] > stop_msgs.times[0]:
    start_adjusted = start_msgs
    stop_adjusted.times = stop_msgs.times[1:]
    stop_adjusted.msgs = stop_msgs.msgs[1:]
  # recording started during driving and ended during driving -> delete last start and first stop message
  elif len(start_msgs) == len(stop_msgs) and start_msgs.times[0] > stop_msgs.times[0] and start_msgs.times[-1] > stop_msgs.times[-1]:
    start_adjusted.times = start_msgs.times[:-1]
    start_adjusted.msgs = start_msgs.msgs[:-1]
    stop_adjusted.times = stop_msgs.times[1:]
    stop_adjusted.msgs = stop_msgs.msgs[1:]
  # recording started before driving and ended during driving -> delete last start message
  elif len(start_msgs) > len(stop_msgs) and start_msgs.times[-1] > stop_msgs.times[-1]:
    stop_adjusted = stop_msgs
    start_adjusted.times = start_msgs.times[:-1]
    start_adjusted.msgs = start_msgs.msgs[:-1]
  else:
    start_adjusted = start_msgs 
    stop_adjusted = stop_msgs
    
  return start_adjusted, stop_adjusted

def extract_missions(msg_container):
  # Only considerd "full" trajectories from start to goal
  start_msgs, stop_msgs = adjust_start_stop_msgs(msg_container['start'], msg_container['stop'])
  
  missions = []
  
  for ii in range(len(start_msgs)):
    data = Mission()
    data.start_time = start_msgs.times[ii]
    data.end_time = stop_msgs.times[ii]
    data.loc_msgs = msg_container['loc'].get_data_for_interval(data.start_time, data.end_time)
    data.odom_msgs = msg_container['odom'].get_data_for_interval(data.start_time, data.end_time)
    data.vel_cmd_msgs = msg_container['vel_cmd'].get_data_for_interval(data.start_time, data.end_time)
    data.scan_msgs = msg_container['scan'].get_data_for_interval(data.start_time, data.end_time)
    data.goal = msg_container['goal'].get_previous_msg(data.start_time)
    missions.append(data)
  return missions

 