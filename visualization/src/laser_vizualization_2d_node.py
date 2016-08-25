#!/usr/bin/env python2

import rospy
from laser_visualization_2d import LaserVisualization2d

def main():
    
    rospy.init_node('laser_visualization')

    with LaserVisualization2d() as vis:
        rospy.spin()

if __name__ == "__main__":
    main()
