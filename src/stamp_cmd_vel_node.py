#!/usr/bin/env python2

import rospy
from geometry_msgs.msg import Twist, TwistStamped

class StampTwistMsgs():

    def __init__(self):
        self.cmd_vel_sub = rospy.Subscriber('/cmd_vel', Twist, self.callback)
        self.cmd_vel_stamped_pub = rospy.Publisher('/cmd_vel_stamped', TwistStamped, queue_size=1)


    def callback(self, data):
        msg = TwistStamped()
        msg.twist = data
        msg.header.stamp = rospy.Time.now()

        self.cmd_vel_stamped_pub.publish(msg)

def main():

    rospy.init_node('stamp_cmd_vel_node')

    StampTwistMsgs()

    rospy.spin()

if __name__ == "__main__":
    main()
