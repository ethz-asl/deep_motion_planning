#!/usr/bin/env python

import tf

from deep_motion_planner.deep_motion_planner import DeepMotionPlanner as dmp

def compute_relative_target_raw(current_pose, target_pose):
  """
  Computes the relative target pose which has to be fed to the network as an input. 
  Both target pose and current_pose have to be in the same coordinate frame (gloabl map).
  """
  # Compute the relative goal position
  goal_position_difference = [target_pose.pose.position.x - current_pose.pose.position.x,
                              target_pose.pose.position.y - current_pose.pose.position.y]

  # Get the current orientation and the goal orientation
  current_orientation = current_pose.pose.orientation
  p = [current_orientation.x, current_orientation.y, current_orientation.z, current_orientation.w]
  goal_orientation = target_pose.pose.orientation
  q = [goal_orientation.x, goal_orientation.y, goal_orientation.z, goal_orientation.w]

  # Rotate the relative goal position into the base frame (robot frame)
  goal_position_base_frame = tf.transformations.quaternion_multiply(tf.transformations.quaternion_inverse(p),
                                                                    tf.transformations.quaternion_multiply([goal_position_difference[0],
                                                                                                            goal_position_difference[1], 
                                                                                                            0, 
                                                                                                            0], 
                                                                                                           p))

  # Compute the difference to the goal orientation
  orientation_to_target = tf.transformations.quaternion_multiply(q, tf.transformations.quaternion_inverse(p))
  yaw = tf.transformations.euler_from_quaternion(orientation_to_target)[2]

  return (goal_position_base_frame[0], -goal_position_base_frame[1], yaw)