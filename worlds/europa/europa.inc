include "europa/sick.inc"
include "europa/hokuyo.inc"
include "europa/camera.inc"

define europa_base position 
(
  color "red"			# Default color.
  gui_nose 1                  	# Draw a nose on the robot so we can see which way it points
  obstacle_return 1           	# Can hit things.
  ranger_return 0.5            	# reflects sonar beams
  blob_return 1               	# Seen by blobfinders  
  fiducial_return 1           	# Seen as "1" fiducial finders

  localization "gps"         	  
  localization_origin [0 0 0 0] 	# Start odometry at (0, 0, 0).

  # alternative odometric localization with simple error model
  # localization "odom"         		# Change to "gps" to have impossibly perfect, global odometry
  #  odom_error [ 0.05 0.05 0.1 ]   	# Odometry error or slip in X, Y and Theta
                                   	# (Uniform random distribution)   

  # four DOF kinematics limits
  # [ xmin xmax ymin ymax zmin zmax amin amax ]				
  velocity_bounds [-1.0 1.0 0 0 0 0 -90.0 90.0 ]					
  acceleration_bounds [-1.0 1.0 0 0 0 0 -90 90.0 ]

  # actual size
  size [0.80 0.70 0.22]

  # the center of rotation is offset from its center of area
  origin [0 0 0 0]

  # draw a nose on the robot so we can see which way it points
  gui_nose 1

  # estimated mass in KG
  mass 100.0 
  # differential steering model
  drive "diff"
)

define europa_base_sensors europa_base
(
  hokuyolaser( pose [0 0 0 0])
  europa_camera( pose [0 0 0.05 0])
)

define europa europa_base_sensors
(  
  # simplified Body shape:
  block( 
    points 8
    point[0] [0.4 0.15]
    point[1] [0.2 0.35]
    point[2] [-0.2 0.35]
    point[3] [-0.4 0.15]
    point[4] [-0.4 -0.15]
    point[5] [-0.2 -0.35]
    point[6] [0.2 -0.35]
    point[7] [0.4 -0.15]
    z [0 0.22]
  )
)

