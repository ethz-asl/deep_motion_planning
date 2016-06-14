
import rospy
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import TwistStamped, PoseStamped

import actionlib
from move_base_msgs.msg import MoveBaseAction

import tf

import threading, time

class DeepMotionPlanner():
    """Use a deep neural network for motion planning"""
    def __init__(self):

        self.target_pose = None
        self.freq = 25.0
       
        # ROS topics
        scan_sub = rospy.Subscriber('scan', LaserScan, self.scan_callback)

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
        pass

    def processing_data(self):
        next_call = time.time()
        while not self.interrupt_event.is_set():

            next_call = next_call+1.0/self.freq
            time.sleep(next_call - time.time())

            if not self.target_pose:
                continue
            goal = self.target_pose
            self.transform_broadcaster.sendTransform(
                    (goal.target_pose.pose.position.x, goal.target_pose.pose.position.y, 
                        goal.target_pose.pose.position.z),
                    (goal.target_pose.pose.orientation.x,goal.target_pose.pose.orientation.y,
                        goal.target_pose.pose.orientation.z,goal.target_pose.pose.orientation.w),
                    rospy.Time().now() , 'target_pose', goal.target_pose.header.frame_id)

            try:
                (trans,rot) = self.transform_listener.lookupTransform('/base_link', '/target_pose',
                                                                        rospy.Time().now())
            except (tf.LookupException, tf.ConnectivityException,
                            tf.ExtrapolationException):
                continue
            print(trans)
            print(rot)


    def goal_callback(self):
        goal = self._as.accept_new_goal()
        self.target_pose = goal

    def preempt_callback(self):
        rospy.logerr('Action preempted')
        self._as.set_preempted(result=None, text='External preemption')

