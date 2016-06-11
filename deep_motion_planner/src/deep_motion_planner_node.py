#!/usr/bin/env python2

import rospy

from deep_motion_planner import DeepMotionPlanner

def main():
    
    rospy.init_node('deep_motion_planner')

    planner = DeepMotionPlanner()

    rospy.spin()

if __name__ == "__main__":
    main()
