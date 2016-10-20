#!/usr/bin/env python2

import numpy as np

import rospy
import tf
from visualize_attention import VisualizeAttention

# Messages
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseStamped

class AttentionHandler():
    """Wrapper for the ROS input/output handling"""

    def __init__(self):
        self.vis = VisualizeAttention()
        rospy.Subscriber('/base_scan', LaserScan, self.laser_callback)
        rospy.Subscriber('/relative_target', PoseStamped, self.target_callback)
        rospy.Subscriber('/deep_planner/sensor_attention', Float32MultiArray,
                self.sensor_attention_callback)

    def laser_callback(self, data):
        self.vis.set_laser_data(data.ranges)

    def target_callback(self, data):

        # Prepare the target for the visualization
        x = data.pose.position.x
        y = data.pose.position.y
        q = (data.pose.orientation.x, data.pose.orientation.y, data.pose.orientation.z,
                data.pose.orientation.w)
        h = tf.transformations.euler_from_quaternion(q)[2]

        self.vis.set_target_pose((x,y,h))

    def sensor_attention_callback(self, sensor_attention):

        # Recreate the numpy array with the correct dimensions
        att = np.array(sensor_attention.data)
        att = np.reshape(att, [sensor_attention.layout.dim[0].size,sensor_attention.layout.dim[1].size])

        self.vis.set_sensor_attention(att)

    def __enter__(self):
        return self
      
    def __exit__(self, exc_type, exc_value, traceback):
        self.vis.close()

def main():
    
    rospy.init_node('attention_visualization')

    with AttentionHandler() as ath:
        rospy.spin()

if __name__ == "__main__":
    main()
