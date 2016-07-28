cmake_minimum_required(VERSION 2.8.3)
project(safety_module)

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  roscpp
  sensor_msgs
  nav_msgs
  message_filters
  tf
)

## Find system dependencies
find_package(Eigen3 REQUIRED)

###########
## Build ##
###########

## Specify additional locations of header files
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  include/
)

add_executable(safety_module_node 
                  src/node.cpp
                  src/SafetyModuleWrapper.cpp
                  src/SafetyModule.cpp
)

target_link_libraries(safety_module_node
  ${catkin_LIBRARIES}
)
