
import rospy
import actionlib
from move_base_msgs.msg import MoveBaseAction
from actionlib_msgs.msg import GoalStatus
from std_msgs.msg import Empty
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid
from map_msgs.msg import OccupancyGridUpdate
import std_srvs.srv

import tf

import os
import math
import random
import numpy as np

from mission_file_parser import MissionFileParser

class MissionControl():

    """Docstring for MissionControl. """

    def __init__(self):
        
        # Load a mission and parse the file
        mission_file = rospy.get_param('~mission_file')
        if not os.path.exists(mission_file):
            rospy.logerr('Mission file not found: {}'.format(mission_file))
            exit()

        self.mission = MissionFileParser(mission_file).get_mission()
        self.mission_index = 0
        self.random_waypoint_number = 0
        self.current_target = [0,0,0]

        self.command_start = rospy.Time.now().to_sec()
        self.command_timeout = rospy.get_param('~command_timeout', default=360.0)

        self.costmap = None

        self.start_pub = rospy.Publisher('/start', Empty, queue_size=1)
        self.stop_pub = rospy.Publisher('/stop', Empty, queue_size=1)
        self.abort_pub = rospy.Publisher('/abort', Empty, queue_size=1)
        self.target_pub = rospy.Publisher('/relative_target', PoseStamped, queue_size=1)

        self.costmap_sub = rospy.Subscriber('/move_base/global_costmap/costmap', OccupancyGrid, \
                self.__costmap_callback__)
        self.costmap_update_sub = rospy.Subscriber('/move_base/global_costmap/costmap_updates', \
                OccupancyGridUpdate, self.__costmap_update_callback__)

        self.navigation_client = actionlib.SimpleActionClient('move_base', MoveBaseAction)

        while not self.navigation_client.wait_for_server(rospy.Duration(5)):
            rospy.loginfo('Waiting for move_base action server')

        rospy.wait_for_service('/reset_positions')

        if len(self.mission) > 0:
            rospy.loginfo('Start mission')
            self.__send_next_command__()
        else:
            rospy.logerr('Mission file contains no commands')
            rospy.signal_shutdown('Mission Finished')

    def __send_next_command__(self):

        if len(self.mission) <= self.mission_index:
            rospy.loginfo('Mission Finished')
            rospy.signal_shutdown('Mission Finished')

        call = {'wp': self.__goto_waypoint__, \
                'cmd': self.__execute_command__, \
                'rd': self.__goto_random__ }

        item = self.mission[self.mission_index]
        call[item[0]](item[1])

    def __goto_waypoint__(self, coordinates):
        rospy.loginfo('Goto waypoint: {}'.format(coordinates))

        goal = MoveBaseAction()
        goal.action_goal.goal.target_pose.header.stamp = rospy.Time.now()

        goal.action_goal.goal.target_pose.header.frame_id = 'map'
        goal.action_goal.goal.target_pose.pose.position.x = coordinates[0]
        goal.action_goal.goal.target_pose.pose.position.y = coordinates[1]

        yaw = coordinates[2] * math.pi/ 360.0
        goal.action_goal.goal.target_pose.pose.orientation.z = math.sin(yaw)
        goal.action_goal.goal.target_pose.pose.orientation.w = math.cos(yaw)

        self.current_target = coordinates

        self.command_start = rospy.Time.now().to_sec()

        self.navigation_client.send_goal(goal.action_goal.goal, self.__done_callback__, \
                self.__active_callback__, self.__feedback_callback__)

    def __goto_random__(self, parameters):
        # first call
        if self.random_waypoint_number == 0:
            self.random_waypoint_number = parameters[0]

        rospy.loginfo('Goto random waypoint: {} remaining'.format(self.random_waypoint_number))

        found_valid_sample = False
        while not found_valid_sample:
            target = [0.0] * 3
            target[0] = random.uniform(parameters[1], parameters[2])
            target[1] = random.uniform(parameters[3], parameters[4])
            target[2] = random.uniform(0.0, 360.0)

            found_valid_sample = self.__check_target_validity__(target)

        self.__goto_waypoint__(target)

    def __feedback_callback__(self, feedback):

        if (rospy.Time.now().to_sec() - self.command_start) > self.command_timeout:
            rospy.loginfo('Timeout for command execution')

            self.navigation_client.cancel_goal()
            return

        target = PoseStamped()
        target.header.stamp = rospy.Time.now()

        goal_position_difference = [self.current_target[0] - feedback.base_position.pose.position.x,
                                    self.current_target[1] - feedback.base_position.pose.position.y]

        yaw = self.current_target[2] * math.pi/ 180.0
        q = tf.transformations.quaternion_from_euler(0,0,yaw)

        current_orientation = feedback.base_position.pose.orientation
        p = [current_orientation.x, current_orientation.y, current_orientation.z, \
                current_orientation.w]

        goal_position_base_frame = tf.transformations.quaternion_multiply(tf.transformations.quaternion_inverse(p),
                tf.transformations.quaternion_multiply([goal_position_difference[0],
                    goal_position_difference[1], 0, 0], p))

        orientation_to_target = tf.transformations.quaternion_multiply(q, \
                tf.transformations.quaternion_inverse(p))
        target.pose.orientation.x = orientation_to_target[0]
        target.pose.orientation.y = orientation_to_target[1]
        target.pose.orientation.z = orientation_to_target[2]
        target.pose.orientation.w = orientation_to_target[3]

        target.pose.position.x = goal_position_base_frame[0]
        target.pose.position.y = -goal_position_base_frame[1]

        self.target_pub.publish(target)

    def __done_callback__(self, state, result):
        # result is empty
        if state == GoalStatus.SUCCEEDED:
            rospy.loginfo('Reached waypoint')

            self.stop_pub.publish(Empty())

            # Sample was valid, so reduce count by one
            if self.random_waypoint_number > 0:
                self.random_waypoint_number -= 1

        else:
            rospy.loginfo('Action returned: {}'.format(GoalStatus.to_string(state)))
            self.abort_pub.publish(Empty())
            self.__reset_simulation__()

            if self.random_waypoint_number > 0:
                rospy.loginfo('Resample this random waypoint')

        # Wait shortly before publishing the next command
        rospy.sleep(0.5)

        if self.random_waypoint_number > 0:
            item = self.mission[self.mission_index]
            self.__goto_random__(item[1])
        else:
            self.mission_index += 1
            self.__send_next_command__()

    def __active_callback__(self):
        self.start_pub.publish(Empty())

    def __execute_command__(self, cmd):
        rospy.loginfo('Execute command: {}'.format(cmd))

        self.command_start = rospy.Time.now()

        self.mission_index += 1
        self.__send_next_command__()

    def __costmap_callback__(self, data):
        self.costmap = data

    def __costmap_update_callback__(self, data):
        if self.costmap:
            update = np.array(data.data).reshape([data.height,data.width])

            current = np.array(self.costmap.data).reshape([self.costmap.info.height,
                self.costmap.info.width])
            current[data.y:data.y+data.height,data.x:data.x+data.width] = update

            self.costmap.data = current.flatten()
            
    def __check_target_validity__(self, target):
        threshold = 50

        if self.costmap:
            x_pixel = int((target[0] - self.costmap.info.origin.position.x) /
                    self.costmap.info.resolution)
            y_pixel = int((target[1] - self.costmap.info.origin.position.y) /
                    self.costmap.info.resolution)

            return self.costmap.data[int(x_pixel + self.costmap.info.width * y_pixel)] < threshold
        else:
            rospy.logwarn('No costmap available')
            return False

    def __reset_simulation__(self):
        try:
            reset_simulation = rospy.ServiceProxy('/reset_positions', std_srvs.srv.Empty)
            reset_simulation()
	except rospy.ServiceException, e:
	    print('Service call failed: {}'.format(e))


