#!/usr/bin/env python2

import rospy
from evaluation_runner import EvaluationRunner

def main():

    rospy.init_node('evaluation_runner')

    mission = EvaluationRunner()

    rospy.spin()

if __name__ == "__main__":
    main()
