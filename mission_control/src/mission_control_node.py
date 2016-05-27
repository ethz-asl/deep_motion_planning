#!/usr/bin/env python2

import rospy
from mission_control import MissionControl

def main():

    try:
        rospy.init_node('mission_control')

        mission = MissionControl()

        rospy.spin()
    except rospy.ROSInterruptException:
        print('mission_control_node interrupted')

if __name__ == "__main__":
    main()
