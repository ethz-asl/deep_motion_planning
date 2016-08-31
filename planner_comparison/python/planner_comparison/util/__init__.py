#!/usr/bin/env python

import pylab as pl
from planner_comparison.plan_scoring import *

def compute_detailed_cost_sum(mission_vec):
  detailed_cost = mission_vec[0].compute_cost()[1]
  for m in mission_vec[1:]:
    for k in detailed_cost.keys():
      detailed_cost[k] += m.compute_cost()[1][k]
  return detailed_cost

def plot_mission(ax, mission, id=None, color='b', plot_numbers=False):
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