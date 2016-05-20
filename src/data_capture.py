
import rospy

class DataCapture():
    """docstring for DataCapture"""
    def __init__(self, storage_path):
        self.storage_path = storage_path
        rospy.loginfo(self.storage_path)
        
