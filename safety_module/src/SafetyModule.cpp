#include <safety_module/SafetyModule.hpp>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <ros/ros.h>

SafetyModule::SafetyModule(double robotRadius, double minimumDistanceToRobot, double timeOffset, double maximumDeceleration)
   : _robotRadius(robotRadius), _minimumDistanceToRobot(minimumDistanceToRobot), _timeOffset(timeOffset), _maxDeceleration(maximumDeceleration) {

}

bool SafetyModule::motionIsSafe(const Eigen::Vector2d& currentPosition, const Eigen::Quaterniond& currentOrientation,
                      const double transVel, const double rotVel,
                      const std::vector<Eigen::Vector2d>& pointsLaser)
{
  
  if(transVel < 0){
    ROS_WARN_STREAM_NAMED("safety_module", "TransVel is negative: " << transVel << ". This is currently unsafe!");
    return false;
  }

  std::ostringstream os;
  _distSensorToEdge = _robotRadius - abs(currentPosition.norm());
  double yaw = currentOrientation.matrix().eulerAngles(0,1,2)[2];
  Eigen::Vector2d currentHeadingVector(cos(yaw), sin(yaw));
  Eigen::Vector2d positionRobotStart, positionRobotEnd, positionCenter;
  positionRobotStart = currentPosition;

  // Compute stopping distance of the robot
  const double distStop = computeStoppingDistance(transVel);
  bool collision = false;
  bool objectIsClose = false;

  double minLaserDistSquared = std::numeric_limits<double>::max();
  double distSquared;
  for (const Eigen::Vector2d& laserMeas : pointsLaser)
  {
    objectIsClose = objectTooClose(laserMeas, positionRobotStart, distSquared);
    minLaserDistSquared = distSquared < minLaserDistSquared ? distSquared : minLaserDistSquared;
    if (objectIsClose)
      _motionIsSafe = false;
  }

  if (!_motionIsSafe)
    os << "Object too close, dist to center: " << sqrt(minLaserDistSquared) << ". ";

  // Compute center of rotation for current motion
  if (_motionIsSafe) {
    ROS_DEBUG_STREAM("distStop: " << distStop << ", transVel: " << transVel << ", rotVel: " << rotVel);
    if (abs(rotVel) <= _rotVelThreshold) {
      for (const Eigen::Vector2d& laserMeas : pointsLaser)
      {
        if (transVel>0) {
          collision = (std::abs(laserMeas(1)) < _robotRadius + _minimumLateralDistance) && (laserMeas(0) < distStop + _distSensorToEdge + _minimumLongitudinalDistance) && (laserMeas(0) > positionRobotStart(1));
          if (collision)
            ROS_DEBUG_STREAM_NAMED("safety_module", "Anticipated collision (straight forward driving). Stopping distance is " << distStop);
        } else if (transVel < 0){
          collision = (std::abs(laserMeas(1)) < _robotRadius + _minimumLateralDistance) && (laserMeas(0) < positionRobotStart(1)) && (laserMeas(0) > -(distStop + _distSensorToEdge + _minimumLongitudinalDistance));
          if (collision)
            ROS_DEBUG_STREAM_NAMED("safety_module", "Anticipated collision (straight backward driving). Stopping distance is " << distStop);
        }

        if (collision)
          _motionIsSafe = false;
      }
    } else {
      const double radiusRobotTrajectory = computeTrajectoryRadius(transVel, rotVel);

      if (radiusRobotTrajectory > _robotRadius) {
        positionCenter = computeRotationCenterOfTrajectory(radiusRobotTrajectory, transVel, rotVel, positionRobotStart, currentHeadingVector);
        positionRobotEnd = computeStoppingPoint(transVel, rotVel, positionRobotStart, positionCenter);

        // generate hyperplanes that form the limits of the critical collision region
        auto hyperCenterStart = Eigen::Hyperplane<double,2>::Through(positionCenter, positionRobotStart);
        auto hyperCenterEnd = Eigen::Hyperplane<double,2>::Through(positionCenter, positionRobotEnd);

        // find sign of the distance (critical region on which side of the halfspace) with respect to the halfspaces
        int signForCollisionCenterStart = getSignOfDistance(hyperCenterStart, positionRobotEnd);
        int signForCollisionCenterEnd = getSignOfDistance(hyperCenterEnd, positionRobotStart);

        for (const Eigen::Vector2d& laserMeas : pointsLaser)
        {
          double distCenterSquared = (laserMeas - positionCenter).squaredNorm();
          collision = (distCenterSquared <= (radiusRobotTrajectory + _robotRadius + _minimumLateralDistance) * (radiusRobotTrajectory + _robotRadius + _minimumLateralDistance)) &&  // within sector of circle
              (distCenterSquared >= (radiusRobotTrajectory - _robotRadius - _minimumLateralDistance) * (radiusRobotTrajectory - _robotRadius - _minimumLateralDistance)) &&
              (signForCollisionCenterStart == getSignOfDistance(hyperCenterStart, laserMeas)) &&  // within halfspace limits
              (signForCollisionCenterEnd == getSignOfDistance(hyperCenterEnd, laserMeas));
          if (collision) {
            _motionIsSafe = false;
            break;
          }
        }
      }
    }

    if (!_motionIsSafe)
      os << "Collision with command tv: " << transVel << ", rv: " << rotVel << ", brake dist: " << distStop << ". ";
  }

  if (_motionIsSafe)
    os << "Motion is safe. ";

  _statusMsg += os.str();
  ROS_DEBUG_STREAM_COND(!_motionIsSafe, _statusMsg);

  return _motionIsSafe;
}


double SafetyModule::computeStoppingDistance(const double transVel) {
  double absVel = std::abs(transVel);
  return absVel * _timeOffset + absVel*absVel / (2 * _maxDeceleration) + _robotRadius;
}


Eigen::Vector2d SafetyModule::transformPointPolar2KarthesianCoordinates(const double range, const double alpha) {
  Eigen::Vector2d laserMeas(cos(alpha) * range, sin(alpha) * range);
  return laserMeas;
}


int SafetyModule::getSignOfDistance(const Eigen::Hyperplane<double, 2>& hyperplane, const Eigen::Vector2d& point) {
  return sign(hyperplane.signedDistance(point));
}

Eigen::Vector2d SafetyModule::computeStoppingPoint(const double transVel, const double rotVel,
                                                   const Eigen::Vector2d& positionStart,
                                                   const Eigen::Vector2d& positionCenter) {
  double dTheta = (computeStoppingDistance(transVel) + _minimumLongitudinalDistance) / abs(transVel/rotVel) * sign(rotVel);
  Eigen::Rotation2D<double> rotationMatrix(dTheta);
  return positionCenter + rotationMatrix * (positionStart - positionCenter);
}

Eigen::Vector2d SafetyModule::computeRotationCenterOfTrajectory(const double radius, const double transVel, const double rotVel,
                                                                const Eigen::Vector2d& positionStart,
                                                                const Eigen::Vector2d& currentHeading) {
  Eigen::Vector2d rotatedHeading90Deg(-currentHeading(1), currentHeading(0));
  return positionStart + sign(rotVel) *sign(transVel) * radius * rotatedHeading90Deg;  // center on the left if positive rotation rate
}

double SafetyModule::computeTrajectoryRadius(const double transVel, const double rotVel) {
  double radius;
  if (std::abs(rotVel)<_rotVelThreshold) {
    radius = 1e4;
  } else {
    radius = std::abs(transVel/rotVel);
  }
  return std::max(0.001, radius);
}

bool SafetyModule::objectTooClose(const Eigen::Vector2d& laserMeas, const Eigen::Vector2d& positionRobot, double& laserPointDistToCenterSquared) {
  laserPointDistToCenterSquared = (laserMeas - positionRobot).squaredNorm();
  bool tooCloseToRobotCenter = (laserPointDistToCenterSquared <= (_robotRadius + _minimumDistanceToRobot) * (_robotRadius + _minimumDistanceToRobot));
  bool tooCloseToSensor(laserMeas.norm()<=_minimumDistanceToSensor);
  ROS_DEBUG_STREAM_COND_NAMED(tooCloseToSensor || tooCloseToRobotCenter, "safety_module", "Distance to robot center is " << sqrt(laserPointDistToCenterSquared));
  ROS_DEBUG_STREAM_COND_NAMED(tooCloseToSensor || tooCloseToRobotCenter, "safety_module","Distance to sensor is " << laserMeas.norm());
  ROS_DEBUG_STREAM_COND_NAMED(tooCloseToSensor || tooCloseToRobotCenter, "safety_module", "Object is too close to sensor: " << tooCloseToSensor << ", too close to robot center: " << tooCloseToRobotCenter);
  return tooCloseToRobotCenter || tooCloseToSensor;
}
