#!/usr/bin/env python

import pylab as pl
from planner_comparison.plan_scoring import *
from bisect import *

def compute_detailed_cost_sum(mission_vec):
  detailed_cost = mission_vec[0].compute_cost()[1]
  for m in mission_vec[1:]:
    for k in detailed_cost.keys():
      detailed_cost[k] += m.compute_cost()[1][k]
  return detailed_cost

def joystick_trigger_times(mission):
  trigger_times = []
  for idx, msg in enumerate(mission.joy_msgs.msgs):
    if idx > 0:
      if msg.buttons[4] == 1 and mission.joy_msgs.msgs[idx-1].buttons[4] == 0:
        trigger_times.append(mission.joy_msgs.times[idx])
    else:
      if msg.buttons[4] == 1:  # active in the beginning
        trigger_times.append(mission.joy_msgs.times[idx])
  return trigger_times

def find_next_joystick_release_time(trigger_time, mission):
  start_idx = bisect(mission.joy_msgs.times, trigger_time)
  for idx in range(start_idx, len(mission.joy_msgs)-1):
    if mission.joy_msgs.msgs[idx+1].buttons[4] == 1 and mission.joy_msgs.msgs[idx+1].buttons[4] == 0:
      return mission.joy_msgs.times[idx]
  return mission.joy_msgs.times[-1]

def joystick_active_time_intervals(mission):
  time_intervals = []
  trigger_times = joystick_trigger_times(mission)
  for t_time in trigger_times:
    time_intervals.append((t_time, find_next_joystick_release_time(t_time, mission)))
  return time_intervals
  
def plot_mission(ax, mission, id=None, color='b', plot_numbers=False, plot_joystick=False):
  traj = mission.get_trajectory()
  
  try:
    goal = mission.goal.pose.position
  except AttributeError:
    goal = mission.goal.goal.target_pose.pose.position
  ax.plot(traj[0,:], traj[1,:], color=color)
  ax.plot(goal.x, goal.y, marker='o', mfc='r', mec='k')
  if id==0:
    ax.plot(traj[0,0], traj[1,0], marker='o', mfc='r', mec='k')
  if plot_numbers:
    if id is not None:
      ax.text(traj[0,0]-0.5, traj[1,0], str(id), bbox={'facecolor':'blue', 'alpha':0.1, 'pad':3})
  
  
def plot_error_bars(ax, cost, color='b', shift=0.0, bar_width=0.2):
  bars = []
  for idx,key in enumerate(cost.keys()):
    bar = ax.bar(idx+1+shift, cost[key], width=bar_width, color=color, align='center')
    bars.append(bar)
  return bars