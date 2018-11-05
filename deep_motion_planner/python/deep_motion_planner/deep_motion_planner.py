import sys
sys.path.insert(0, '/usr/local/lib/python2.7/dist-packages')

import rospy
import actionlib
import tf
import threading
import time
import os
import math
import copy
import csv
from datetime import datetime

# Messages
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, PoseStamped, Point, Quaternion
from move_base_msgs.msg import MoveBaseGoal, MoveBaseAction, MoveBaseFeedback
from sensor_msgs.msg import Joy
from nav_msgs.msg import Path
import rospkg
# print("Current path: {}".format(os.path.dirname(os.path.abspath(__file__))))
import support as sup


# Tensorflow
from tensorflow_wrapper import TensorflowWrapper

import numpy as np

# Utils
import util

class DeepMotionPlanner():
  """Use a deep neural network for motion planning"""
  def __init__(self):

    self.target_pose = None
    self.last_scan = None
    self.freq = 6.0
    self.send_motion_commands = True
    self.base_position = None
    self.base_orientation = None
    self.max_laser_range = 20.0
    self.num_subsampled_scans = 1080
    self.input_dim = self.num_subsampled_scans + 2
    self.num_raw_laser_scans = 1080
    self.time_last_call = rospy.get_rostime()
#     self.column_line = ['count'] + \
#                        ['laser_raw' + str(i) for i in range(self.num_raw_laser_scans)] + \
#                        ['target_global_frame_x', 'target_global_frame_y', 'target_global_frame_yaw',
#                         'robot_pose_global_frame_x', 'robot_pose_global_frame_y', 'robot_pose_global_frame_yaw'] + \
#                        ['laser_input_model' + str(i) for i in range(self.num_subsampled_scans)] + \
#                        ['goal_distance', 'goal_angle']
#
#     date_str = datetime.strftime(datetime.now(), '%Y-%m-%d_%H-%M-%S')
#     self.storage_path = os.path.join('/home/pfmark/Desktop/dump', date_str)
#     print("Logging under: {}".format(self.storage_path))
#     os.mkdir(self.storage_path)
#
#     self.output_file = open(os.path.join(self.storage_path, 'logging_data.csv'), 'wb')
#     self.writer= csv.writer(self.output_file, delimiter=',')
#     self.writer.writerow(self.column_line)

    # Load various ROS parameters
    if not rospy.has_param('~model_path'):
      rospy.logerr('Missing parameter: ~model_path')
      exit()

    self.laser_scan_stride = rospy.get_param('~laser_scan_stride', default=1) # Take every ith element
    self.n_laser_scans = rospy.get_param('~n_laser_scans', default=1080) # Cut n elements from the center to adjust the length
    self.model_path = rospy.get_param('~model_path')
    self.pickle_weights_path = rospy.get_param('~pickle_weights_path', default=None)
    self.use_pickle_weights = rospy.get_param('~use_pickle_weights', default=False)
    self.protobuf_file = rospy.get_param('~protobuf_file', 'graph.pb')
    self.use_checkpoints = rospy.get_param('~use_checkpoints', default=False)
    if not os.path.exists(self.model_path):
      rospy.logerr('Model path does not exist: {}'.format(self.model_path))
      rospy.logerr('Please check the parameter: {}'.format(rospy.resolve_name('~model_path')))
      exit()
    if not os.path.exists(os.path.join(self.model_path, self.protobuf_file)):
      rospy.logerr('Protobuf file does not exist: {}'.format(os.path.join(self.model_path, self.protobuf_file)))
      rospy.logerr('Please check the parameter: {}'.format(rospy.resolve_name('~protobuf_file')))
      exit()

    # Use a separate thread to process the received data
    self.interrupt_event = threading.Event()
    self.processing_thread = threading.Thread(target=self.processing_data)
    self.scan_lock = threading.Lock()

    # ROS topics
    scan_sub = rospy.Subscriber('scan', LaserScan, self.scan_callback)
    goal_sub = rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.goal_topic_callback)
    joystick_sub = rospy.Subscriber('/joy', Joy, self.joystick_callback)
    self.cmd_pub  = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
    self.deep_plan_pub = rospy.Publisher('/deep_planner/path', Path, queue_size=1)
    self.relative_target_pub  = rospy.Publisher('/relative_target', PoseStamped, queue_size=1)
    self.input_goal_pub = rospy.Publisher('/deep_planner/input/goal', Float32MultiArray, queue_size=1)
    self.input_laser_pub = rospy.Publisher('/deep_planner/input/laser', LaserScan, queue_size=1)


    # We over the same action api as the move base package
    self._as = actionlib.SimpleActionServer('deep_move_base', MoveBaseAction, auto_start = False)
    self._as.register_goal_callback(self.goal_callback)
    self._as.register_preempt_callback(self.preempt_callback)

    self.transform_listener = tf.TransformListener()

    self.processing_thread.start()
    self._as.start()

    self.navigation_client = actionlib.SimpleActionClient('deep_move_base', MoveBaseAction)
    while not self.navigation_client.wait_for_server(rospy.Duration(5)):
      rospy.loginfo('Waiting for deep_move_base action server')

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    # Make sure to stop the thread properly
    self.interrupt_event.set()
    self.processing_thread.join()

  def scan_callback(self, data):
    """
    Callback function for the laser scan messages
    """
    self.scan_lock.acquire()
    self.last_scan = data
    self.scan_lock.release()


  def processing_data(self):
    """
    Process the received sensor data and publish a new command

    The function does not return, it is used as thread function and
    runs until the interrupt_event is set
    """
    # Get a handle for the tensorflow interface
    if self.use_pickle_weights:
      filename_weights = self.pickle_weights_path
    else:
      filename_weights = None
    with TensorflowWrapper(self.model_path, protobuf_file=self.protobuf_file, use_checkpoints=self.use_checkpoints,
                           filename_weights=filename_weights, input_dim=self.input_dim) as tf_wrapper:
      next_call = time.time()
      # Stop if the interrupt is requested
      cnt = 1

      while not self.interrupt_event.is_set():

        # Run processing with the correct frequency
#         next_call = next_call+1.0/self.freq
#         sleep_time = next_call - time.time()
#         if sleep_time > 0.0:
#           time.sleep(sleep_time)
#         else:
#           rospy.logerr('Missed control loop frequency')

        if rospy.get_rostime() - self.time_last_call >= rospy.Duration(1.0 / self.freq):
          self.time_last_call = rospy.get_rostime()
          # Make sure, we have goal
          if not self._as.is_active():
            continue
          # Make sure, we received the first laser scan message
          if not self.target_pose or not self.last_scan:
            continue
          # Get the relative target pose
          target = self.compute_relative_target()
          if not target:
            continue

          self.scan_lock.acquire()
          scan_msg = copy.copy(self.last_scan)
          self.scan_lock.release()

          cropped_scans = util.adjust_laser_scans_to_model(self.last_scan.ranges, self.laser_scan_stride, self.n_laser_scans, perception_radius = 100.0)

          # Convert scans to numpy array
          cropped_scans_np = np.atleast_2d(np.array(cropped_scans))
          transformed_scans = sup.transform_laser(cropped_scans_np, self.num_subsampled_scans)

          if any(np.isnan(cropped_scans)) or any(np.isinf(cropped_scans)):
            rospy.logerr('Scan contained invalid float (nan or inf)')

          # Publish the scan data fed into the network
          cropped_scan_msg = LaserScan()
          cropped_scan_msg.header = scan_msg.header
          cropped_scan_msg.angle_increment = scan_msg.angle_increment / self.laser_scan_stride
          cropped_scan_msg.angle_min = -self.n_laser_scans * cropped_scan_msg.angle_increment / 2
          cropped_scan_msg.angle_max = self.n_laser_scans * cropped_scan_msg.angle_increment / 2
          cropped_scan_msg.time_increment = scan_msg.time_increment
          cropped_scan_msg.scan_time = scan_msg.scan_time
          cropped_scan_msg.scan_time = scan_msg.scan_time
          cropped_scan_msg.range_min = scan_msg.range_min
          cropped_scan_msg.range_max = scan_msg.range_max
          cropped_scan_msg.ranges = cropped_scans
          self.input_laser_pub.publish(cropped_scan_msg)


          # Prepare the input vector, perform the inference on the model
          # and publish a new command
          goal = np.array(target)
          angle = np.arctan2(goal[1],goal[0])
          norm = np.minimum(np.linalg.norm(goal[0:2], ord=2), self.max_laser_range)

          # Normalize / transform
          transformed_angle = sup.transform_target_angle(angle, norm_angle=np.pi)
          transformed_norm = sup.transform_target_distance(norm, norm_range=self.max_laser_range)

          data = np.stack((transformed_angle, transformed_norm, goal[2]))
  #         data = np.stack((angle, norm, goal[2]))

          # Publish the goal pose fed into the network
          goal_msg = Float32MultiArray()
          goal_msg.data = data
          self.input_goal_pub.publish(goal_msg)

          input_data = list(transformed_scans.tolist()[0]) + data.tolist()[0:2]

          (base_position,base_orientation) = self.transform_listener.lookupTransform('/map', '/base_link', rospy.Time())

          linear_x, angular_z = tf_wrapper.inference(input_data)

          cmd = Twist()
          cmd.linear.x = linear_x
          cmd.angular.z = angular_z
          if self.send_motion_commands:
            self.cmd_pub.publish(cmd)
            self.publish_predicted_path(cmd)

          # Check if the goal pose is reached
          self.check_goal_reached(target)

  def check_goal_reached(self, target):
    """
    Check if the position and orientation are close enough to the target.
    If this is the case, set the current goal to succeeded.
    """
    position_tolerance = 0.2
    orientation_tolerance = 10.0 #0.1
    if abs(target[0]) < position_tolerance \
        and abs(target[1]) < position_tolerance \
        and abs(target[2]) < orientation_tolerance:
      self._as.set_succeeded()

  def compute_relative_target(self):
    """
    Compute the target pose in the base_link frame and publish the current pose of the robot
    """
    try:
      # Get the base_link transformation
      (base_position,base_orientation) = self.transform_listener.lookupTransform('/map', '/base_link',
                                  rospy.Time())
    except (tf.LookupException, tf.ConnectivityException,
            tf.ExtrapolationException):
      rospy.logwarn("Lookup exception in transform for relative target computation.")
      return None

    # Publish feedback (the current pose)
    feedback = MoveBaseFeedback()
    feedback.base_position.header.stamp = rospy.Time().now()
    feedback.base_position.pose.position.x = base_position[0]
    feedback.base_position.pose.position.y = base_position[1]
    feedback.base_position.pose.position.z = base_position[2]
    feedback.base_position.pose.orientation.x = base_orientation[0]
    feedback.base_position.pose.orientation.y = base_orientation[1]
    feedback.base_position.pose.orientation.z = base_orientation[2]
    feedback.base_position.pose.orientation.w = base_orientation[3]
    self.base_position = base_position
    self.base_orientation = base_orientation
    self._as.publish_feedback(feedback)

    # Compute the relative goal position

    # Get the current orientation and the goal orientation
    current_orientation = feedback.base_position.pose.orientation
    p = [current_orientation.x, current_orientation.y, current_orientation.z, \
        current_orientation.w]
    goal_orientation = self.target_pose.target_pose.pose.orientation
    q = [goal_orientation.x, goal_orientation.y, goal_orientation.z, \
        goal_orientation.w]

    rel_target = sup.get_target_in_robot_frame(np.array([feedback.base_position.pose.position.x,
                                                         feedback.base_position.pose.position.y,
                                                         sup.get_yaw_from_quat(current_orientation)]),
                                               np.array([self.target_pose.target_pose.pose.position.x,
                                                         self.target_pose.target_pose.pose.position.y,
                                                         sup.get_yaw_from_quat(goal_orientation)]))
    self.relative_target_pub.publish(rel_target)

    return (rel_target.pose.position.x, rel_target.pose.position.y, sup.get_yaw_from_quat(rel_target.pose.orientation))

  def goal_callback(self):
    """
    Callback function when a new goal pose is requested
    """
    self.abort_planning = False
    goal = self._as.accept_new_goal()
    self.target_pose = goal

  def preempt_callback(self):
    """
    Callback function when the current action is preempted
    """
    rospy.logerr('Action preempted')
    self._as.set_preempted(result=None, text='External preemption')

  def goal_topic_callback(self, data):
    # Generate a action message
    goal = MoveBaseGoal()

    goal.target_pose.header.frame_id = 'map'
    goal.target_pose.pose.position.x = data.pose.position.x
    goal.target_pose.pose.position.y = data.pose.position.y


    goal.target_pose.pose.orientation.x = data.pose.orientation.x
    goal.target_pose.pose.orientation.y = data.pose.orientation.y
    goal.target_pose.pose.orientation.z = data.pose.orientation.z
    goal.target_pose.pose.orientation.w = data.pose.orientation.w

    # Send the waypoint
    self.navigation_client.send_goal(goal)

  def joystick_callback(self, data):

    # Pause planning
    if data.buttons[4] == 0:
      rospy.logdebug("Sending motion commands: ON")
      self.send_motion_commands = True
    else:
      self.send_motion_commands = False
      rospy.logdebug("Sending motion commands: OFF")


    # Abort planning
    if data.buttons[4] == 1 and data.buttons[5] == 1:
      rospy.loginfo("Planning aborted!")
      self._as.set_succeeded()

  def publish_predicted_path(self, cmd_vel, sim_time=rospy.Duration(1.7), dt=rospy.Duration(0.05)):
    """
    Compute plan from velocity command (assuming constant translational and rotational velocity)
    for the succeeding time interval specified with sim_time
    """
    path = Path()
    start_time = rospy.get_rostime()
    final_time = start_time + sim_time
    path.header.stamp = start_time
    path.header.frame_id = 'map'
    p = PoseStamped()
    p.pose.position.x = self.base_position[0]
    p.pose.position.y = self.base_position[1]
    p.pose.orientation.x = self.base_orientation[0]
    p.pose.orientation.y = self.base_orientation[1]
    p.pose.orientation.z = self.base_orientation[2]
    p.pose.orientation.w = self.base_orientation[3]
    path.poses.append(p)

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
