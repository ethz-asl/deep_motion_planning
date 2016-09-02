import rosbag 
import pickle
import os

# Messages
from geometry_msgs.msg import Twist, PoseStamped, Point, Quaternion
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Path, Odometry
from move_base_msgs.msg import MoveBaseActionFeedback
from std_msgs.msg import Empty
from actionlib_msgs.msg import GoalStatus

from time_msg_container import *

class RosbagInterface():
  
  def __init__(self, log_path):
    self.log_path = log_path
    self.bag = rosbag.Bag(self.log_path)
    self.msg_container = {}
    self.msg_container = self.load_messages()

  def load_messages(self, topics=None):
    if topics is None:
      # Get ros planner commands
      vel_cmd_msgs = TimeMsgContainer()
      for topic, msg, t in self.bag.read_messages(topics=['/cmd_vel']):
        vel_cmd_msgs.times.append(t)
        vel_cmd_msgs.msgs.append(msg)
      self.msg_container['vel_cmd'] = vel_cmd_msgs
      if len(vel_cmd_msgs) == 0:
        vel_cmd_msgs = TimeMsgContainer()
        for topic, msg, t in self.bag.read_messages(topics=['/cmd_vel_mux/input/navi']):
          vel_cmd_msgs.times.append(t)
          vel_cmd_msgs.msgs.append(msg)
        self.msg_container['vel_cmd'] = vel_cmd_msgs
      
      # Get scans 
      scan_msgs = TimeMsgContainer()
      for topic, msg, t in self.bag.read_messages(topics=['/base_scan']):
        scan_msgs.times.append(t)
        scan_msgs.msgs.append(msg)
      self.msg_container['scan'] = scan_msgs
      
      # Get published goal positions 
      goal_msgs = TimeMsgContainer()
      for topic, msg, t in self.bag.read_messages(topics=['/move_base/current_goal']):
        goal_msgs.times.append(t)
        goal_msgs.msgs.append(msg)
      if len(goal_msgs) > 0:
        self.msg_container['goal'] = goal_msgs
      else:
        goal_msgs = TimeMsgContainer()
        for topic, msg, t in self.bag.read_messages(topics=['/deep_move_base/goal']):
          goal_msgs.times.append(t)
          goal_msgs.msgs.append(msg)
        self.msg_container['goal'] = goal_msgs
        
      # Get position/localization messages
      loc_msgs = TimeMsgContainer()
      for topic, msg, t in self.bag.read_messages(topics=['/amcl_pose']):
        loc_msgs.times.append(t)
        loc_msgs.msgs.append(msg)
      self.msg_container['loc'] = loc_msgs
      if len(loc_msgs) == 0:
        loc_msgs = TimeMsgContainer()
        for topic, msg, t in self.bag.read_messages(topics=['/move_base/feedback']):
          loc_msgs.times.append(t)
          loc_msgs.msgs.append(msg)
        self.msg_container['loc'] = loc_msgs
      
      # Odometry data
      odom_msgs = TimeMsgContainer()
      for topic, msg, t in self.bag.read_messages(topics=['/base_pose_ground_truth']):
        odom_msgs.times.append(t)
        odom_msgs.msgs.append(msg)
      self.msg_container['odom'] = odom_msgs
      if len(odom_msgs) == 0:
        odom_msgs = TimeMsgContainer()
        for topic, msg, t in self.bag.read_messages(topics=['/odom']):
          odom_msgs.times.append(t)
          odom_msgs.msgs.append(msg)
        self.msg_container['odom'] = odom_msgs
      
      # Mission start
      start_msgs = TimeMsgContainer()
      for topic, msg, t in self.bag.read_messages(topics=['/start']):
        start_msgs.times.append(t)
        start_msgs.msgs.append(msg)
      self.msg_container['start'] = start_msgs
      
      # Mission stop
      stop_msgs = TimeMsgContainer()
      for topic, msg, t in self.bag.read_messages(topics=['/stop']):
        stop_msgs.times.append(t)
        stop_msgs.msgs.append(msg)
      self.msg_container['stop'] = stop_msgs
      
      # Goal status messages
      goal_status_msgs = TimeMsgContainer()
      for topic, msg, t in self.bag.read_messages(topics=['/move_base/status']):
        goal_status_msgs.times.append(t)
        goal_status_msgs.msgs.append(msg)
      self.msg_container['goal_status'] = goal_status_msgs
      
      # Map msg
      map_msgs = TimeMsgContainer()
      for topic, msg, t in self.bag.read_messages(topics=['/map']):
        map_msgs.times.append(t)
        map_msgs.msgs.append(msg)
      self.msg_container['map'] = map_msgs
    else:
      for topic in topics:
        time_msg_container = TimeMsgContainer()
        for topic, msg, t in self.bag.read_messages(topics=[topic]):
          time_msg_container.times.append(t)
          time_msg_container.msgs.append(msg)
        self.msg_container[topic] = time_msg_container
        
    return self.msg_container
  
  def get_topics(self):
    return self.msg_container.keys()
    
  def save_container(self, filename, path=''):
    pickle.dump(self.msg_container, open(os.path.join(path, filename), 'wb'))