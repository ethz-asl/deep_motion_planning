import sys
sys.path.insert(0, '/usr/local/lib/python2.7/dist-packages')

import numpy as np
import pandas as pd
import os
import csv
import support as sup

# ROS imports
import rospy
import actionlib
import rospkg
import tf
from move_base_msgs.msg import MoveBaseAction
from std_msgs.msg import Empty, Int8
from geometry_msgs.msg import PoseStamped, Pose2D
from actionlib_msgs.msg import GoalStatus
import std_srvs.srv


class EvaluationRunner():

  def __init__(self):
    self.mission_file = rospy.get_param('~mission_file')
    use_deep_motion_planner = rospy.get_param('~deep_motion_planner', default=True)
    self.evaluation_info = rospy.get_param('~evaluation_info', default="map_network")
    if not os.path.exists(self.mission_file):
      rospy.logerr('Mission file not found: {}'.format("mission_file"))
      exit()
    self.mission_data = pd.read_csv(self.mission_file)
    self.trajectory_idx = 0
    self.command_start = rospy.Time.now().to_sec()
    self.command_timeout_threshold = 300.0
    self.current_target = [0, 0, 0]
    self.crash_time = rospy.Time.now().to_sec()
    self.stalled_old= 0

    # Counters
    self.n_crash = 0
    self.n_timeout = 0
    self.n_success = 0

    self.tf_broadcaster = tf.TransformBroadcaster()

    # Subscribers
    self.crashed_sub = rospy.Subscriber('/stalled', Int8, self.__stalled_callback__)

    # Publishers
    self.start_pub = rospy.Publisher('/start', Empty, queue_size=1)
    self.stop_pub = rospy.Publisher('/stop', Empty, queue_size=1)
    self.abort_pub = rospy.Publisher('/abort', Empty, queue_size=1)
    self.target_pub = rospy.Publisher('/relative_target', PoseStamped, queue_size=1)
    self.respawn_pub = rospy.Publisher('/cmd_pose', Pose2D, queue_size=1)

    # Services
    rospy.wait_for_service('/reset_positions')

    # Set up action client
    if use_deep_motion_planner:
      self.navigation_client = actionlib.SimpleActionClient('deep_move_base', MoveBaseAction)
      while not self.navigation_client.wait_for_server(rospy.Duration(5)):
        rospy.loginfo('Waiting for deep_move_base action server')
    else:
      self.navigation_client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
      while not self.navigation_client.wait_for_server(rospy.Duration(5)):
        rospy.loginfo('Waiting for move_base action server')

    if len(self.mission_data) > 0:
      rospy.loginfo("Starting evaluation mission with {} trajectories.".format(len(self.mission_data)))
      self.__send_next_command__()
    else:
      rospy.logerr("Mission file empty.")
      rospy.signal_shutdown("Mission finished.")

  def __send_next_command__(self):
    """
    Send the next command in the mission list
    """
    rospy.loginfo("Running trajectory number {} / {}.".format(self.trajectory_idx, len(self.mission_data)))

    rospy.loginfo("{:.3f} % success, \t{:.3f} % timeout, \t{:.3f} % crash".format(100*(self.n_success / float(np.maximum(self.trajectory_idx, 1))),
                                                                                  100*(self.n_timeout / float(np.maximum(self.trajectory_idx, 1))),
                                                                                  100*(self.n_crash / float(np.maximum(self.trajectory_idx, 1)))))

    if self.trajectory_idx >= len(self.mission_data):
      self.__write_result_to_file__()
      rospy.loginfo("Mission Finished")
      rospy.signal_shutdown("Mission Finished")

    self.__reset_robot_to_pose__([self.mission_data['start_x'][self.trajectory_idx],
                                  self.mission_data['start_y'][self.trajectory_idx],
                                  self.mission_data['start_yaw'][self.trajectory_idx]])


    target_coordinates = [self.mission_data['final_x'][self.trajectory_idx],
                          self.mission_data['final_y'][self.trajectory_idx],
                          self.mission_data['final_yaw'][self.trajectory_idx]]

    self.__goto_waypoint__(target_coordinates)


  def __goto_waypoint__(self, coordinates):
    """
    Send the goal given by coordinates to the move_base node
    """
    rospy.loginfo("Goto waypoint: {}".format(coordinates))

    # Generate a action message
    goal = MoveBaseAction()
    goal.action_goal.goal.target_pose.header.stamp = rospy.Time.now()

    goal.action_goal.goal.target_pose.header.frame_id = 'map'
    goal.action_goal.goal.target_pose.pose.position.x = coordinates[0]
    goal.action_goal.goal.target_pose.pose.position.y = coordinates[1]

    # Get the quaternion from the orientation in degree
    yaw = coordinates[2]
    goal.action_goal.goal.target_pose.pose.orientation.z = np.sin(yaw)
    goal.action_goal.goal.target_pose.pose.orientation.w = np.cos(yaw)

    self.current_target = coordinates

    self.command_start = rospy.Time.now().to_sec()

    # Send the waypoint
    self.navigation_client.send_goal(goal.action_goal.goal, self.__done_callback__, \
            self.__active_callback__, self.__feedback_callback__)


  def __feedback_callback__(self, feedback):
    """
    Callback for the feedback during the execution of __goto_waypoint__

    We compute the relative pose of the global target pose within the base frame and
    publish it as ROS topic
    """

    # Check if we reached the timeout
    if (rospy.Time.now().to_sec() - self.command_start) > self.command_timeout_threshold:
      rospy.loginfo("Timeout for command execution")
      self.n_timeout += 1
      self.navigation_client.cancel_goal()
      self.abort_pub.publish(Empty())
      return

    # Compute the relative goal pose within the robot base frame
    target = PoseStamped()
    target.header.stamp = rospy.Time.now()

    # Get the quaternion from the current goal
    yaw = self.current_target[2]
    q = tf.transformations.quaternion_from_euler(0., 0., yaw)

    current_orientation = feedback.base_position.pose.orientation
    p = [current_orientation.x, current_orientation.y, current_orientation.z, \
            current_orientation.w]

    target = sup.get_target_in_robot_frame(np.array([feedback.base_position.pose.position.x,
                                                     feedback.base_position.pose.position.y,
                                                     sup.get_yaw_from_quat(feedback.base_position.pose.orientation)]),
                                           np.array([self.current_target[0],
                                                     self.current_target[1],
                                                     self.current_target[2]]))
    target.header.stamp = rospy.Time.now()

    self.target_pub.publish(target)

    self.tf_broadcaster.sendTransform((self.current_target[0], self.current_target[1], 0),
                    q,
                    rospy.Time.now(),
                    'goal',
                    'map')

  def __done_callback__(self, state, result):
    """
    Callback when the execution of __goto_waypoint__ has finished

    We check if the execution was successful and trigger the next command
    """
    if state == GoalStatus.SUCCEEDED:
      # Publish stop message and reduce number of random waypoints
      rospy.loginfo("Reached waypoint")
      self.n_success += 1
      self.stop_pub.publish(Empty())

    else:
      # Execution was not successful, so abort execution and reset the simulation
      rospy.loginfo("Action returned: {}".format(GoalStatus.to_string(state)))
      self.abort_pub.publish(Empty())

    # Wait shortly before publishing the next command
    rospy.sleep(0.5)

    self.trajectory_idx += 1
    if self.trajectory_idx >= len(self.mission_data):
      self.__write_result_to_file__()
      rospy.loginfo("Mission finished")
      rospy.signal_shutdown("Maximum number of waypoints reached. Mission finished.")

    self.__send_next_command__()

  def __active_callback__(self):
    """
    Callback when the execution of __goto_waypoint__ starts

    We publish this event as ROS topic
    """
    self.start_pub.publish(Empty())


  def __reset_robot_to_pose__(self, pose):
    """
    Re-spawn robot at pre-defined pose.
    """
    spawn_pose = Pose2D()
    spawn_pose.x = pose[0]
    spawn_pose.y = pose[1]
    spawn_pose.theta = pose[2]

    rospy.loginfo("Spawning robot at ({}, {}, {})".format(spawn_pose.x, spawn_pose.y, spawn_pose.theta))
    self.respawn_pub.publish(spawn_pose)


  def __stalled_callback__(self, data):
    if data.data == 1 and self.stalled_old == 0:
      rospy.loginfo("Robot crashed. Continuing with next trajectory.")
      self.n_crash += 1
      self.trajectory_idx += 1
      self.abort_pub.publish(Empty())
      self.__reset_simulation__()
      self.__send_next_command__()
    self.stalled_old = data.data

  def __reset_simulation__(self):
    """
    Reset the simulation by calling the ROS service
    """
    try:
      reset_simulation = rospy.ServiceProxy('/reset_positions', std_srvs.srv.Empty)
      reset_simulation()
    except rospy.ServiceException, e:
      print('Service call failed: {}'.format(e))


  def __write_result_to_file__(self):
    map_name = self.evaluation_info[:self.evaluation_info.find('_')]
    file_name = self.evaluation_info + '_' + str(len(self.mission_data)) + '.csv'
    storage_path = os.path.join(rospkg.RosPack().get_path('evaluation_tools'), 'evaluation_results', file_name)
    rospy.loginfo("Writing results to '{}'".format(storage_path))
    output_file = open(storage_path, 'w')
    writer= csv.writer(output_file, delimiter=',')
    writer.writerow(['map_name', 'num_trajectories', 'num_success', 'num_timeout', 'num_crash'])
    writer.writerow([map_name, len(self.mission_data), self.n_success, self.n_timeout, self.n_crash])
    output_file.flush()
    output_file.close()


