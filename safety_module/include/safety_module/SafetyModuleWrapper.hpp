#ifndef _SAFETY_MODULE_WRAPPER_HPP_
#define _SAFETY_MODULE_WRAPPER_HPP_

#include <geometry_msgs/TwistStamped.h>
#include <ros/ros.h>
#include <sensor_msgs/LaserScan.h>

#include <tf/transform_listener.h>

#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>

#include "safety_module/SafetyModule.hpp"

namespace safety_module
{
class SafetyModuleWrapper
{
public:
  SafetyModuleWrapper();
  ~SafetyModuleWrapper() = default;

private:
  SafetyModule _safetyModule;

  ros::NodeHandle _nh;
  tf::TransformListener _tfListener;

  ros::Publisher _safetyInterruptPub;  //!< Publish message in an unsafe state
  ros::Publisher _cmdPub;              //!< Forward the velocity commands if it is safe

  message_filters::Subscriber<sensor_msgs::LaserScan> _laserSub;
  message_filters::Subscriber<geometry_msgs::TwistStamped> _cmdSub;
  message_filters::TimeSynchronizer<sensor_msgs::LaserScan, geometry_msgs::TwistStamped> _sync;

  void syncCallback(const sensor_msgs::LaserScan::ConstPtr& laser,
                    const geometry_msgs::TwistStamped::ConstPtr& cmd);
};
} /* safety_module */

#endif /* _SAFETY_MODULE_WRAPPER_HPP_ */
