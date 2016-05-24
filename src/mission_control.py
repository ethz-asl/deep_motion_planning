
import rospy
import actionlib
from move_base_msgs.msg import MoveBaseAction
from actionlib_msgs.msg import GoalStatus
from std_msgs.msg import Empty
from geometry_msgs.msg import PoseStamped

import tf

import os
import math

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

        self.start_pub = rospy.Publisher('/start', Empty, queue_size=1)
        self.stop_pub = rospy.Publisher('/stop', Empty, queue_size=1)
        self.abort_pub = rospy.Publisher('/abort', Empty, queue_size=1)
        self.target_pub = rospy.Publisher('/relative_target', PoseStamped, queue_size=1)

        self.navigation_client = actionlib.SimpleActionClient('move_base', MoveBaseAction)

        while not self.navigation_client.wait_for_server(rospy.Duration(5)):
            rospy.loginfo('Waiting for move_base action server')

        rospy.loginfo('Start mission')
        self.__send_next_command__()
        

    def __send_next_command__(self):

        if len(self.mission) <= self.mission_index:
            rospy.loginfo('Mission Finished')
            rospy.signal_shutdown('Mission Finished')

        call = {'wp': self.__goto_waypoint__, 'cmd': self.__execute_command__}

        item = self.mission[self.mission_index]
        call[item[0]](item[1])
        self.mission_index += 1

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

        self.navigation_client.send_goal(goal.action_goal.goal, self.__done_callback__, \
                self.__active_callback__, self.__feedback_callback__)

    def __feedback_callback__(self, feedback):
        target = PoseStamped()
        target.header.stamp = rospy.Time.now()

        goal = self.mission[self.mission_index - 1][1]
        target.pose.position.x = goal[0] - feedback.base_position.pose.position.x
        target.pose.position.y = goal[1] - feedback.base_position.pose.position.y

        yaw = goal[2] * math.pi/ 180.0
        q = tf.transformations.quaternion_from_euler(0,0,yaw)

        current_orientation = feedback.base_position.pose.orientation
        p = [current_orientation.x, current_orientation.y, current_orientation.z, \
                current_orientation.w]

        target.pose.orientation = tf.transformations.quaternion_multiply(q, \
                tf.transformations.quaternion_inverse(p))

        self.target_pub.publish(target)

    def __done_callback__(self, state, result):
        # result is empty
        if state == GoalStatus.SUCCEEDED:
            rospy.loginfo('Reached waypoint')

            self.stop_pub.publish(Empty())
        else:
            rospy.loginfo('Action returned: {}'.format(GoalStatus.to_string(state)))
            self.abort_pub.publish(Empty())

        self.__send_next_command__()

    def __active_callback__(self):
        self.start_pub.publish(Empty())


    def __execute_command__(self, cmd):
        rospy.loginfo('Execute command: {}'.format(cmd))
        self.__send_next_command__()
            
