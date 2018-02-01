"""
Add the src directory to the Python path
"""
import os
import sys

def add_path(path):
  if path not in sys.path:
    sys.path.insert(0, path)

project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
add_path(os.path.join(project_dir, 'src'))
add_path(os.path.join(project_dir, '..', 'deep_motion_planner', 'python'))

