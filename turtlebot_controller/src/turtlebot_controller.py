#!/usr/bin/env python
import rospy

import numpy as np
import math

# Messages
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry

class TurtlebotController():
  """
  Controller with integration for turtlebot in order to achieve accurate velocity tracking
  """
  
  def __init__(self):
    self.measured_velocity = None
    self.integral_trans_vel = 0.0
    self.integral_rot_vel = 0.0
    self.last_control_time = rospy.Time.now()
    
    self.cmd_vel_sub = rospy.Subscriber('/deep_planner/cmd_vel', Twist, self.__cmd_vel_callback__)
    self.meas_vel_sub = rospy.Subscriber('/odom', Odometry, self.__meas_vel_callback__)
    self.controller_vel_pub = rospy.Publisher('/navigation_velocity_smoother/raw_cmd_vel', Twist, queue_size=1)
    
    # Controller weights
    self.ffwd_weight = 1.0
    self.trans_weight = 0.05
    self.rot_weight = 0.1
    
    
  def __cmd_vel_callback__(self, cmd_vel):
    new_time = rospy.Time.now()
    if self.measured_velocity.linear.x < 0.1 and self.measured_velocity.angular.z < 0.1 and self.measured_velocity is not None:
      self.integral_trans_vel += (cmd_vel.linear.x - self.measured_velocity.linear.x) / (new_time - self.last_control_time).to_sec()
      self.integral_rot_vel += (cmd_vel.angular.z - self.measured_velocity.angular.z) / (new_time - self.last_control_time).to_sec()
      cmd_vel_controller = Twist()
      cmd_vel_controller.linear.x = self.ffwd_weight*cmd_vel.linear.x + self.trans_weight*self.integral_trans_vel
      cmd_vel_controller.angular.z = self.ffwd_weight*cmd_vel.angular.z + self.rot_weight*self.integral_rot_vel
      rospy.logdebug("cmd_vel = ({0}, {1}) \t meas_vel = ({2}, {3})".format(cmd_vel.linear.x, cmd_vel.angular.z, self.measured_velocity.linear.x, self.measured_velocity.angular.z))
      rospy.logdebug("integral part = ({0}, {1})".format(self.integral_trans_vel, self.integral_rot_vel))
      rospy.logdebug("resulting command = ({0}, {1})".format(cmd_vel_controller.linear.x, cmd_vel_controller.angular.z))
      self.controller_vel_pub.publish(cmd_vel_controller)
    else:
      # Reset integrators
      rospy.logdebug("Resetting integrators.")
      self.integral_trans_vel = 0.0
      self.integral_rot_vel = 0.0
      self.controller_vel_pub.publish(cmd_vel)
    
  def __meas_vel_callback__(self, odom):
    self.measured_velocity = odom.twist.twist
    
  def __enter__(self):
    return self
      
  def __exit__(self, exc_type, exc_value, traceback):
    return self