#include "safety_module/SafetyModuleWrapper.hpp"

#include <std_msgs/Empty.h>

#include <Eigen/Geometry>
#include <vector>

namespace safety_module
{
    
  SafetyModuleWrapper::SafetyModuleWrapper()
    : _safetyModule(0.35, 0.02, 0.0, 1.0),
      _nh("~"),
      _laserSub(_nh, "/base_scan", 1), 
      _cmdSub(_nh, "/unchecked_cmd_vel_stamped", 1), 
      _sync(_laserSub, _cmdSub, 10)
  {
    _safetyInterruptPub = _nh.advertise<std_msgs::Empty>("/safety_interrupt", 1);
    _cmdPub = _nh.advertise<geometry_msgs::Twist>("/cmd_vel", 1);
    _sync.registerCallback(&SafetyModuleWrapper::syncCallback, this);
  }

  void SafetyModuleWrapper::syncCallback(const sensor_msgs::LaserScan::ConstPtr& laser,
                                         const geometry_msgs::TwistStamped::ConstPtr& cmd)
  {
    // Get the transformation between robot base and laser base
    tf::StampedTransform transform;
    _tfListener.lookupTransform("/base_link", "/base_laser_link", ros::Time(0), transform);
    Eigen::Vector2d currentPosition(transform.getOrigin().getX(), transform.getOrigin().getY());
    Eigen::Quaterniond currentOrientation(transform.getRotation().getW(), transform.getRotation().getX(),
        transform.getRotation().getY(), transform.getRotation().getZ());

    // Convert the laser measurements to 2D points in the laser frame
    std::vector<Eigen::Vector2d> pointsList;
    for(size_t i = 0; i < laser->ranges.size(); ++i)
    {
      double alpha = laser->angle_min + static_cast<double>(i) * laser->angle_increment;
      Eigen::Vector2d laserMeas(cos(alpha) * laser->ranges.at(i), sin(alpha) * laser->ranges.at(i));
      pointsList.push_back(laserMeas);
    }

    if (_safetyModule.motionIsSafe(currentPosition, currentOrientation, cmd->twist.linear.x,
          cmd->twist.angular.z, pointsList))
    {
      geometry_msgs::Twist output;
      output = cmd->twist;
      _cmdPub.publish(output);
    }
    else
    {
      std_msgs::Empty msg;
      _safetyInterruptPub.publish(msg);

      // Publish a message with zeros
      geometry_msgs::Twist output;
      _cmdPub.publish(output);
    }
    // TODO Handle reset more carefully
    _safetyModule.reset();
  }

} /* safety_module */ 
