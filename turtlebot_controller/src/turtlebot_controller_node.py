#!/usr/bin/env python2

import rospy
from turtlebot_controller import TurtlebotController

def main():
    
    rospy.init_node('turtlebot_controller')

    with TurtlebotController() as controller:
        rospy.spin()

if __name__ == "__main__":
    main()
