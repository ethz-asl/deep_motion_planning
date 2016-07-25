#include "safety_module/SafetyModuleWrapper.hpp"

namespace safety_module
{
    
  SafetyModuleWrapper::SafetyModuleWrapper()
    : _nh("~"),
      _laser_sub(_nh, "base_scan", 1), 
      _odom_sub(_nh, "odom", 1), 
      _cmd_sub(_nh, "cmd_vel_stamped", 1), 
      _sync(_laser_sub, _odom_sub, _cmd_sub, 10)
  {
    _sync.registerCallback(&SafetyModuleWrapper::syncCallback, this);
  }

  void SafetyModuleWrapper::syncCallback(const sensor_msgs::LaserScan::ConstPtr& laser,
                                         const nav_msgs::Odometry::ConstPtr& odom,
                                         const geometry_msgs::TwistStamped::ConstPtr& cmd)
  {
    ROS_INFO("call");
  }

} /* safety_module */ 
