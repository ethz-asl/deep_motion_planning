#ifndef SAFETYMODULE_H_
#define SAFETYMODULE_H_

#include <Eigen/Geometry>

#include <vector>

inline int sign(const double x) { return x >= 0 ? 1 : -1; }

class SafetyModule
{

public:
  typedef Eigen::Matrix<double, 2, Eigen::Dynamic> LaserMeasurementMatrix;

  SafetyModule() {}
  SafetyModule(double robotRadius, double minimumDistanceToRobot, double timeOffset,
               double maximumDeceleration);
  ~SafetyModule() {}

  bool motionIsSafe(const Eigen::Vector2d& currentPosition,
                    const Eigen::Quaterniond& currentOrientation, const double transVel,
                    const double rotVel, const std::vector<Eigen::Vector2d>& pointsLaser);

  void reset()
  {
    _motionIsSafe = true;
    _statusMsg.clear();
  }

  double computeStoppingDistance(const double transVel);

  Eigen::Vector2d transformPointPolar2KarthesianCoordinates(const double range, const double alpha);

  int getSignOfDistance(const Eigen::Hyperplane<double, 2>& hyperplane,
                        const Eigen::Vector2d& point);

  Eigen::Vector2d computeStoppingPoint(const double transVel, const double rotVel,
                                       const Eigen::Vector2d& positionStart,
                                       const Eigen::Vector2d& positionCenter);

  Eigen::Vector2d computeRotationCenterOfTrajectory(const double radius, const double transVel,
                                                    const double rotVel,
                                                    const Eigen::Vector2d& positionStart,
                                                    const Eigen::Vector2d& currentHeading);

  double computeTrajectoryRadius(const double transVel, const double rotVel);

  bool objectTooClose(const Eigen::Vector2d& laserMeas, const Eigen::Vector2d& positionRobot,
                      double& laserPointDistToCenterSquared);

  double getRotVelThreshold() const { return _rotVelThreshold; }
  void setRotVelThreshold(double rotVelThreshold = 1e-3) { _rotVelThreshold = rotVelThreshold; }

  double getMaxDeceleration() const { return _maxDeceleration; }
  void setMaxDeceleration(double maxDeceleration = 1) { _maxDeceleration = maxDeceleration; }

  double getRobotRadius() const { return _robotRadius; }
  void setRobotRadius(double robotRadius = 0.4) { _robotRadius = robotRadius; }

  double getTimeOffset() const { return _timeOffset; }
  void setTimeOffset(double timeOffset = 0.3) { _timeOffset = timeOffset; }

  double getMinDistToRobot() const { return _minimumDistanceToRobot; }
  void setMinDistToRobot(double minDistToRobot) { _minimumDistanceToRobot = minDistToRobot; }

  double getMinDistToSensor() const { return _minimumDistanceToSensor; }
  void setMinDistToSensor(double minDistToSensor) { _minimumDistanceToSensor = minDistToSensor; }

  double getMinLateralDist() const { return _minimumLateralDistance; }
  void setMinLateralDist(double minLateralDist) { _minimumLateralDistance = minLateralDist; }

  double getMinLongitudinalDist() const { return _minimumLongitudinalDistance; }
  void setMinLongitudinalDist(double minLongitudinalDist)
  {
    _minimumLongitudinalDistance = minLongitudinalDist;
  }

  const std::string& getStatusMessage() const { return _statusMsg; }

private:
  // parameters
  double _robotRadius            = 0.18;
  double _minimumDistanceToRobot = 0.01;  // m
  double _timeOffset      = 0.0;  // s (time until sent velocity command gets executed by the robot)
  double _maxDeceleration = 20.0;  // m/s^2
  double _rotVelThreshold = 1e-3;             // rad/s
  double _distSensorToEdge            = 0.02;  // m
  double _minimumDistanceToSensor     = 0.0;
  double _minimumLateralDistance      = 0.01;
  double _minimumLongitudinalDistance = 0.01;

  // safety state
  bool _motionIsSafe = true;

  std::string _statusMsg;
};

#endif /* SAFETYMODULE_H_ */
