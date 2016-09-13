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

def get_joystick_trajectories(mission):
  joystick_intervals = joystick_active_time_intervals(mission)
  trajectories = []
  for interval in joystick_intervals:
    trajectories.append(mission.get_trajectory_for_time_interval(interval))
  return trajectories

def plot_joystick_interference(ax, mission, color='r', alpha=1.0, linewidth=1.0 ):
  trajectories = get_joystick_trajectories(mission)
  th = []
  for traj in trajectories:
#     t = ax.plot(traj[0,:], traj[1,:], color=color, alpha=alpha, linewidth=1, marker='o', markersize=2, mew=0.1, mec='none', mfc=color)
    t = ax.plot(traj[0,:], traj[1,:], color=color, alpha=alpha, linewidth=1)
    th.append(t)
  return th
  
  
def plot_mission(ax, mission, id=None, color='b', plot_numbers=False, plot_joystick=False, fontsize=9, alpha=1.0, linewidth=1.0):
  traj = mission.get_trajectory()
  
  try:
    goal = mission.goal.pose.position
  except AttributeError:
    goal = mission.goal.goal.target_pose.pose.position
  th = ax.plot(traj[0,:], traj[1,:], color=color, alpha=alpha)
  ax.plot(goal.x, goal.y, marker='o', mfc='r', mec='r')
  if id==0 and plot_numbers:
    ax.plot(traj[0,0], traj[1,0], marker='o', mfc='r', mec='r')
  if plot_numbers:
    if id is not None:
      ax.text(traj[0,-1], traj[1,-1], str(id), color='k', fontsize=fontsize)
  return th
  
def plot_error_bars(ax, cost, color='b', shift=0.0, bar_width=0.2):
  bars = []
  for idx,key in enumerate(cost.keys()):
    bar = ax.bar(idx+1+shift, cost[key], width=bar_width, color=color, align='center')
    bars.append(bar)
  return bars

def plot_relative_error_bars(ax, rel_cost, colors=['g', 'k'], bar_width=0.2):
  bars = []
  for idx,key in enumerate(rel_cost.keys()):
    color = colors[0] if rel_cost[key]>0 else colors[1]
    bar = ax.bar(idx+1, 100*rel_cost[key], width=bar_width, color=color, align='center')
    bars.append(bar)
    ax.plot([-100, 100], [0, 0], color='k')
    ax.set_ylabel('%')
    ax.set_xlim(0, len(rel_cost)+1)
    x_tick_pos = np.arange(len(rel_cost)) + 1
    pl.xticks(x_tick_pos, rel_cost.keys(), rotation=25, fontsize=9)
  return bars

def get_complete_missions(missions):
  complete = []
  for m in missions: 
    if len(joystick_active_time_intervals(m)) == 0:
      complete.append(True)
    else: 
      complete.append(False)
      
  return complete


def compute_joystick_distance(mission):
  joystick_distance = 0.0
  joystick_trajs = get_joystick_trajectories(mission)
  
  for traj in joystick_trajs:
    if traj.shape[1] > 1:
      pos_old = traj[:,0]
      for ii in range(1, len(traj)):
        pos_new = traj[:,ii]
        joystick_distance += np.linalg.norm(pos_new-pos_old)
    else:
      joystick_distance += 0.03
      
  return joystick_distance

def compute_autonomous_percent_mission(mission):
  if absolute_distance > 0:
    return 1.0 - compute_joystick_distance(mission) / mission.distance()
  else: 
    return 1.0
  
  