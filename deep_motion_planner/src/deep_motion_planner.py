
import rospy
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Empty
from geometry_msgs.msg import TwistStamped, PoseStamped

import tf

class DeepMotionPlanner():
    """Use a deep neural network for motion planning"""
    def __init__(self):
       
        # ROS topics
        rospy.Subscriber('/start', Empty, self.start_callback)
        rospy.Subscriber('/stop', Empty, self.stop_callback)

        scan_sub = rospy.Subscriber('scan', LaserScan, self.scan_callback)

    def start_callback(self, data):
        rospy.loginfo('Start deep motion planning')

    def stop_callback(self, data):
        rospy.loginfo('Stop deep motion planning')

    def scan_callback(self, data):
        rospy.loginfo('Receied scan')

