#include "safety_module/SafetyModuleWrapper.hpp"
#include <ros/ros.h>

int main(int argc, char* argv[])
{
  ros::init(argc, argv, "safety_module_node");

  safety_module::SafetyModuleWrapper wrapper;

  ros::spin();
  return 0;
}
