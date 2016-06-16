
import rospy
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, PoseStamped

import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseFeedback, MoveBaseResult

import tf

import threading, time

from tensorflow_wrapper import TensorflowWrapper

class DeepMotionPlanner():
    """Use a deep neural network for motion planning"""
    def __init__(self):

        self.target_pose = None
        self.last_scan = None
        self.freq = 25.0
       
        # ROS topics
        scan_sub = rospy.Subscriber('scan', LaserScan, self.scan_callback)
        self.cmd_pub  = rospy.Publisher('/cmd_vel', Twist, queue_size=1)

        self._as = actionlib.SimpleActionServer('deep_move_base',
                MoveBaseAction, auto_start =
                False)
        self._as.register_goal_callback(self.goal_callback)
        self._as.register_preempt_callback(self.preempt_callback)

        self.transform_broadcaster = tf.TransformBroadcaster()
        self.transform_listener = tf.TransformListener()

        self.interrupt_event = threading.Event()
        self.processing_thread = threading.Thread(target=self.processing_data)

        self.processing_thread.start()
        self._as.start()

    def __enter__(self):
        return self
      
    def __exit__(self, exc_type, exc_value, traceback):
        self.interrupt_event.set()
        self.processing_thread.join()

    def scan_callback(self, data):
        self.last_scan = data

    def processing_data(self):
        with TensorflowWrapper() as tf_wrapper:
            next_call = time.time()
            while not self.interrupt_event.is_set():

                next_call = next_call+1.0/self.freq
                time.sleep(next_call - time.time())

                if not self._as.is_active():
                    continue

                if not self.target_pose or not self.last_scan:
                    continue

                target = self.compute_relative_target()
                if not target:
                    continue
                        
                input_data = list(self.last_scan.ranges) + list(target)

                linear_x, angular_z = tf_wrapper.inference(input_data)

                cmd = Twist()
                cmd.linear.x = linear_x
                cmd.angular.z = angular_z
                self.cmd_pub.publish(cmd)

                self.check_goal_reached(target)

    def check_goal_reached(self, target):
        """
        Check if the position and orientation are close enough to the target.
        If this is the case, set the current goal to succeeded.
        """
        position_tolerance = 0.1
        orientation_tolerance = 0.1
        if abs(target[0]) < position_tolerance \
                and abs(target[1]) < position_tolerance \
                and abs(target[2]) < orientation_tolerance:
            self._as.set_succeeded()


    def compute_relative_target(self):
        """
        Compute the target pose in the base_link frame and publish the current pose of the robot
        """
        try:
            (base_position,base_orientation) = self.transform_listener.lookupTransform('/map', '/base_link',
                                                                    rospy.Time())
        except (tf.LookupException, tf.ConnectivityException,
                        tf.ExtrapolationException):
            return None

        # Publish feedback
        feedback = MoveBaseFeedback()
        feedback.base_position.header.stamp = rospy.Time().now()
        feedback.base_position.pose.position.x = base_position[0]
        feedback.base_position.pose.position.y = base_position[1]
        feedback.base_position.pose.position.z = base_position[2]
        feedback.base_position.pose.orientation.x = base_orientation[0]
        feedback.base_position.pose.orientation.y = base_orientation[1]
        feedback.base_position.pose.orientation.z = base_orientation[2]
        feedback.base_position.pose.orientation.w = base_orientation[3]
        self._as.publish_feedback(feedback)

        goal_position_difference = [self.target_pose.target_pose.pose.position.x - feedback.base_position.pose.position.x,
                                    self.target_pose.target_pose.pose.position.y - feedback.base_position.pose.position.y]

        current_orientation = feedback.base_position.pose.orientation
        p = [current_orientation.x, current_orientation.y, current_orientation.z, \
                current_orientation.w]
        goal_orientation = self.target_pose.target_pose.pose.orientation
        q = [goal_orientation.x, goal_orientation.y, goal_orientation.z, \
                goal_orientation.w]

        # Rotate the relative goal position into the base frame
        goal_position_base_frame = tf.transformations.quaternion_multiply(
                tf.transformations.quaternion_inverse(p),
                tf.transformations.quaternion_multiply([goal_position_difference[0],
                    goal_position_difference[1], 0, 0], p))

        # Compute the difference to the goal orientation
        orientation_to_target = tf.transformations.quaternion_multiply(q, \
                tf.transformations.quaternion_inverse(p))
        yaw = tf.transformations.euler_from_quaternion(orientation_to_target)[2]

        return (goal_position_base_frame[0], -goal_position_base_frame[1], yaw)

    def goal_callback(self):
        goal = self._as.accept_new_goal()
        self.target_pose = goal

    def preempt_callback(self):
        rospy.logerr('Action preempted')
        self._as.set_preempted(result=None, text='External preemption')

