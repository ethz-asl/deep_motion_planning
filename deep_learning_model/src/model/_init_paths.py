"""
Add the src directory to the Python path
"""
import os
import sys
    
project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
path = os.path.join(project_dir, 'src')

if path not in sys.path:
    sys.path.insert(0, path)
