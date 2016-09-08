#!/usr/bin/env python2

import rospy
from planner_comparison import PlannerComparison

def main():

    rospy.init_node('planner_comparison')

    mission = PlannerComparison()

    rospy.spin()

if __name__ == "__main__":
    main()
