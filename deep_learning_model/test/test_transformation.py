import sys
sys.path.append('../src/data')
import support as sup

import numpy as np
import pylab as pl
import tf
from geometry_msgs.msg import PoseStamped


pl.close('all')
np.set_printoptions(precision=3)

robot_pose_np = np.array([-2.0, 2.0, np.pi / 4.0])
target_pose_np = np.array([1.0, 1.0, 3*np.pi / 2.0])

robot_pose = PoseStamped()
robot_pose.pose.position.x = robot_pose_np[0]
robot_pose.pose.position.y = robot_pose_np[1]
robot_yaw_quat = tf.transformations.quaternion_from_euler(0., 0., robot_pose_np[2])
robot_pose.pose.orientation.x = robot_yaw_quat[0]
robot_pose.pose.orientation.y = robot_yaw_quat[1]
robot_pose.pose.orientation.z = robot_yaw_quat[2]
robot_pose.pose.orientation.w = robot_yaw_quat[3]

target_pose = PoseStamped()
target_pose.pose.position.x = target_pose_np[0]
target_pose.pose.position.y = target_pose_np[1]
target_yaw_quat = tf.transformations.quaternion_from_euler(0., 0., target_pose_np[2])
target_pose.pose.orientation.x = target_yaw_quat[0]
target_pose.pose.orientation.y = target_yaw_quat[1]
target_pose.pose.orientation.z = target_yaw_quat[2]
target_pose.pose.orientation.w = target_yaw_quat[3]


fig = pl.figure('coordinate transformation')
ax = pl.gca()
ax.plot(robot_pose_np[0], robot_pose_np[1], marker='.', markersize=10, color='b')
ax.arrow(robot_pose_np[0], robot_pose_np[1], np.cos(robot_pose_np[2]), np.sin(robot_pose_np[2]), color='b', head_width=0.1, head_length=0.1)
ax.arrow(robot_pose_np[0], robot_pose_np[1], -0.5*np.sin(robot_pose_np[2]), 0.5*np.cos(robot_pose_np[2]), color='b', linewidth=0.5, head_width=0.1, head_length=0.1)
ax.plot(target_pose_np[0], target_pose_np[1], marker='.', markersize=10, color='r')
ax.arrow(target_pose_np[0], target_pose_np[1], np.cos(target_pose_np[2]), np.sin(target_pose_np[2]), color='r', head_width=0.1, head_length=0.1)
ax.arrow(target_pose_np[0], target_pose_np[1], -0.5*np.sin(target_pose_np[2]), 0.5*np.cos(target_pose_np[2]), color='r', linewidth=0.5, head_width=0.1, head_length=0.1)

ax.set_xlim([-4, 4])
ax.set_ylim([-4, 4])
ax.set_aspect('equal')


# Quaternion-based transformation
target_local_frame_quat = sup.get_target_in_robot_frame(robot_pose_np, target_pose_np)
target_angle_local_frame_euler = sup.get_relative_angle_to_goal(robot_pose.pose.position, robot_pose.pose.orientation, target_pose.pose.position)
target_distance_local_frame_euler = sup.get_distance(robot_pose.pose.position, target_pose.pose.position)

target_position_local_frame = np.array([target_local_frame_quat.pose.position.x, target_local_frame_quat.pose.position.y])
yaw  = tf.transformations.euler_from_quaternion([target_local_frame_quat.pose.orientation.x,
                                                 target_local_frame_quat.pose.orientation.y,
                                                 target_local_frame_quat.pose.orientation.z,
                                                 target_local_frame_quat.pose.orientation.w])[2]
print("Target in local frame: ({:.3f}, {:.3f}, {:.3f})".format(target_local_frame_quat.pose.position.x, target_local_frame_quat.pose.position.y, yaw * 180.0 / np.pi))
print("Relative goal position: ({:.3f}, {:.3f})".format(np.linalg.norm(target_position_local_frame),
                                                        np.math.atan2(target_local_frame_quat.pose.position.y,
                                                                      target_local_frame_quat.pose.position.x) * 180 / np.pi))
print(' ')
print("Relative goal position euler transformation: ({:.3f}, {:.3f})".format(target_distance_local_frame_euler,
                                                                             target_angle_local_frame_euler * 180 / np.pi))

pl.show(block=False)




