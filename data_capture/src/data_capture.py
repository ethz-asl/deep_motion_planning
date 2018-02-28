import sys
sys.path.insert(0, '/usr/local/lib/python2.7/dist-packages')


import rospy
import message_filters
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Empty
from geometry_msgs.msg import TwistStamped, PoseStamped
from nav_msgs.msg import Odometry

import tf

import csv
import os
from datetime import datetime
import numpy as np

import support as sup


class DataCapture():
  """Class that captures various ROS messages and saves them into a .csv file"""
  def __init__(self, storage_path):

    # Prepare the generation of the storage folder
    date_str = datetime.strftime(datetime.now(), '%Y-%m-%d_%H-%M-%S')
    self.storage_path = os.path.join(storage_path, date_str)

    rospy.loginfo(self.storage_path)

    self.enable_capture = False
    self.data_buffer = list()
    self.target_count = 1 #sequential number for the file name
    self.first_file = True
    self.target_global_frame = PoseStamped()

    # ROS topics
    rospy.Subscriber('/start', Empty, self.start_callback)
    rospy.Subscriber('/stop', Empty, self.stop_callback)
    rospy.Subscriber('/abort', Empty, self.abort_callback)
    rospy.Subscriber('/odom', Odometry, self.odom_callback)
    rospy.Subscriber('/move_base/current_goal', PoseStamped, self.global_target_callback)

    # Synchronized messages
    scan_sub = message_filters.Subscriber('scan', LaserScan)
    cmd_sub = message_filters.Subscriber('cmd_vel', TwistStamped)
    odom_sub = message_filters.Subscriber('odom', Odometry)
    target_sub = message_filters.Subscriber('relative_target', PoseStamped)

    self.synchonizer = message_filters.TimeSynchronizer([scan_sub, cmd_sub, odom_sub, target_sub], 10)
    self.synchonizer.registerCallback(self.sync_callback)

  def start_callback(self, data):
    """
    Enable the data capture
    """
    if not self.enable_capture:
        rospy.loginfo('Start data capture')
        self.enable_capture = True

  def stop_callback(self, data):
    """
    Disable data capture and write the cached messages into a file
    """
    if self.enable_capture:
      rospy.loginfo('Stop data capture')
      self.enable_capture = False

      self.__write_data_to_file__()

  def global_target_callback(self, data):
    """
    Store current target.
    """
    self.target_global_frame = data


  def abort_callback(self, data):
    """
    Disable data capture and clear cache without writing it into a file
    """
    if self.enable_capture:
      rospy.loginfo('Abort and clear buffered data')
      self.data_buffer = list()
      self.enable_capture = False

  def odom_callback(self, data):
    current_time = rospy.get_time()
    odom_time = data.header.stamp.to_sec()
    rospy.logdebug("ROS time: {} \tOdometry time: {}".format(current_time, odom_time))
    if current_time - odom_time > 0.01:
      rospy.logdebug("Odometry message delayed by {}".format(current_time - odom_time))


  def sync_callback(self, scan, cmd, odom, target):
    """
    Callback for the syncronizer that caches the syncronized messages
    """
    if self.enable_capture:
      # concatenate the data and add it to the buffer
      orientation = [target.pose.orientation.x, target.pose.orientation.y,
              target.pose.orientation.z, target.pose.orientation.w]
      yaw = tf.transformations.euler_from_quaternion(orientation)[2]

      # Global target data
      target_global_frame_x = self.target_global_frame.pose.position.x
      target_global_frame_y = self.target_global_frame.pose.position.y
      target_global_frame_yaw = tf.transformations.euler_from_quaternion([self.target_global_frame.pose.orientation.x,
                                                                          self.target_global_frame.pose.orientation.y,
                                                                          self.target_global_frame.pose.orientation.z,
                                                                          self.target_global_frame.pose.orientation.w])[2]

      # Current position
      robot_pose_x = odom.pose.pose.position.x
      robot_pose_y = odom.pose.pose.position.y
      robot_pose_yaw = tf.transformations.euler_from_quaternion([odom.pose.pose.orientation.x, odom.pose.pose.orientation.y,
                                                                 odom.pose.pose.orientation.z, odom.pose.pose.orientation.w])[2]

      new_row = [cmd.header.stamp.to_nsec(), cmd.twist.linear.x, cmd.twist.angular.z] + \
                 list(scan.ranges) + [odom.twist.twist.linear.x, odom.twist.twist.angular.z,
                                      target.pose.position.x, target.pose.position.y, yaw,
                                      target_global_frame_x, target_global_frame_y, target_global_frame_yaw,
                                      robot_pose_x, robot_pose_y, robot_pose_yaw]

      rospy.logdebug("Current position: ({}, {}, {})".format(robot_pose_x, robot_pose_y, robot_pose_yaw * 180.0 / np.pi))
      rospy.logdebug("Global target: ({}, {}, {})".format(target_global_frame_x,
                                                         target_global_frame_y,
                                                         target_global_frame_yaw * 180. / np.pi))
      rospy.logdebug("Relative target: ({}, {}, {})".format(target.pose.position.x, target.pose.position.y, yaw * 180. / np.pi))

      self.data_buffer.append(new_row)

  def __write_data_to_file__(self):
    """
    Write the cached data into a .csv file
    """
    rospy.loginfo('Write data to file: {} items'.format(len(self.data_buffer)))

    # Prevent creation of empty files
    if len(self.data_buffer) == 0:
      rospy.loginfo('Received no messages: No csv file is written')
      return

    # Create the storage folder when writing the first file
    if self.first_file:
      os.mkdir(self.storage_path)
      self.first_file = False

    # Create the first line of the csv file with column names
    # We define the length of the laser by the length of captured data, minus the fields
    # that are not related to the laser (stamp, commands and target position)
    column_line = ['stamp','linear_x_command','angular_z_command'] + \
                  ['laser_' + str(i) for i in range(len(self.data_buffer[0]) - 14)] + ['linear_x_odom', 'angular_z_odom', 'target_x',
                  'target_y', 'target_yaw', 'target_global_frame_x', 'target_global_frame_y', 'target_global_frame_yaw',
                  'robot_pose_global_frame_x', 'robot_pose_global_frame_y', 'robot_pose_global_frame_yaw']

    # write the data into a csv file and reset the buffer
    with open(os.path.join(self.storage_path,('target_' + str(self.target_count) + '.csv')), \
      'wb') as output_file:

      writer = csv.writer(output_file, delimiter=',')
      writer.writerow(column_line)
      for l in self.data_buffer:
          writer.writerow(l)

    self.data_buffer = list()
    self.target_count += 1

