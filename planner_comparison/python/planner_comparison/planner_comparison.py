import rospy
from geometry_msgs.msg import Twist, PoseStamped, Point, Quaternion
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Path

import logging
import tf
import math
import numpy as np


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Compare different motion planners')

    parser.add_argument('-m', '--mail', help='Send an email when training finishes',
            action='store_true')
    args = parser.parse_args()

    return args
  

class PlannerComparison():
  """
  Compare the performance of different local motion planners (with local knowledge) 
  compared to the global planner.
  """
  def __init__(self):
            
    self.current_pose = PoseStamped()
    self.ros_plan_cmd = None
    self.deep_plan_cmd = None
        
    # ROS params
    self.executed_plan = rospy.get_param('~executed_plan')
    rospy.loginfo("Executing plan that comes from %s".format(self.executed_plan))
    
    # ROS topics
    self.cmd_ros_plan_sub = rospy.Subscriber('ros_planner/cmd_vel', Twist, self.__callback_ros_plan_cmd__)
    self.cmd_deep_plan_sub = rospy.Subscriber('deep_planner/cmd_vel', Twist, self.__callback_deep_plan__)
    self.cmd_pub  = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
    self.deep_plan_pub = rospy.Publisher('/deep_planner/path', Path, queue_size=1)
    
    # Transformations 
    self.tf_listener = tf.TransformListener()
    
    
  def __callback_ros_plan_cmd__(self, data):
    self.ros_plan_cmd = data
    if self.executed_plan == 'ros':
      self.__publish_vel_cmd__(data)
      self.__callback_current_pose__()

  def __callback_deep_plan__(self, data):
    self.deep_plan_cmd = data
    self.__publish_path_from_vel_cmd__(data, rospy.Duration(1.7))
    if self.executed_plan == 'deep':
      self.__publish_vel_cmd__(data)
      self.__callback_current_pose__()
      
  def __callback_current_pose__(self):
    """
    Set the current pose of the robot (robot coordinate frame vs. odom frame)
    """
    try:
      (trans,rot) = self.tf_listener.lookupTransform('odom', 'base_footprint', rospy.Time(0))
      self.current_pose.header.stamp = self.tf_listener.getLatestCommonTime('odom', 'base_footprint')
      self.current_pose.pose.position = Point(*trans)
      self.current_pose.pose.orientation = Quaternion(*rot)
    except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
      rospy.logerr('Transformation does not exist.')
      
  def __publish_vel_cmd__(self, cmd_vel):
    """
    Publish velocity command under topic "/cmd_vel" (which the simulation listens to)
    """
    cmd = Twist()
    cmd.linear.x = cmd_vel.linear.x
    cmd.angular.z = cmd_vel.angular.z
    self.cmd_pub.publish(cmd)
      
      
  def __publish_path_from_vel_cmd__(self, cmd_vel, sim_time, dt=rospy.Duration(0.05)):
    """
    Compute plan from velocity command (assuming constant translational and rotational velocity) 
    for the succeeding time interval specified with sim_time
    """
    path = Path()
    start_time = self.current_pose.header.stamp
    final_time = start_time + sim_time
    path.header.stamp = start_time
    path.header.frame_id = 'odom'
    path.poses.append(self.current_pose)
    
    t = start_time
    trans_vel = cmd_vel.linear.x
    rot_vel = cmd_vel.angular.z
    while t < start_time + sim_time:
      t += dt
      new_pose = PoseStamped()
      new_pose.header.stamp = t
      quat = path.poses[-1].pose.orientation
      euler = tf.transformations.euler_from_quaternion([quat.x, quat.y, quat.z, quat.w])
      current_yaw = euler[2]
      new_yaw = current_yaw + dt.to_sec()*rot_vel
      yaw_mean = (current_yaw + new_yaw) / 2.0
      old_pos = path.poses[-1].pose.position
      new_pose.pose.position = Point(old_pos.x + trans_vel*math.cos(yaw_mean)*dt.to_sec(), 
                                     old_pos.y + trans_vel*math.sin(yaw_mean)*dt.to_sec(), 
                                     old_pos.z + 0.0)
      new_pose.pose.orientation = Quaternion(*tf.transformations.quaternion_from_euler(0.0, 0.0, new_yaw))
      path.poses.append(new_pose)
      
    self.deep_plan_pub.publish(path)
