#ifndef _SAFETY_MODULE_WRAPPER_HPP_
#define _SAFETY_MODULE_WRAPPER_HPP_

#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/LaserScan.h>
#include <geometry_msgs/TwistStamped.h>

#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>

namespace safety_module
{
  class SafetyModuleWrapper
  {
  public:
    SafetyModuleWrapper();
    ~SafetyModuleWrapper() = default;
  
  private:
    ros::NodeHandle _nh;

    message_filters::Subscriber<sensor_msgs::LaserScan> _laser_sub;
    message_filters::Subscriber<nav_msgs::Odometry> _odom_sub;
    message_filters::Subscriber<geometry_msgs::TwistStamped> _cmd_sub;
    message_filters::TimeSynchronizer<sensor_msgs::LaserScan, nav_msgs::Odometry, 
      geometry_msgs::TwistStamped> _sync;

    void syncCallback(const sensor_msgs::LaserScan::ConstPtr& laser,
                      const nav_msgs::Odometry::ConstPtr& odom,
                      const geometry_msgs::TwistStamped::ConstPtr& cmd);
  }; 
} /* safety_module */ 

#endif /* _SAFETY_MODULE_WRAPPER_HPP_ */
