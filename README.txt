The Y axis points from the delta center to the side camera
The Z axis points from the delta center to the top camera
The actuator inputs are 3D vectors, and it is an implementation decision to 
decide which index is what actuator.  In my implementation, the actuator closest to the camera
was actuator 0.  The actuator to the right if standing on the delta facing the side camera was 
actuator 1.  

The main entry point is Vision_Actuator_Coordinator.py (VAC)

The job of VAC is to run trajectories on a Delta while filming with both cameras.
The recorded videos are processed and combined to create a 3d trajectory for the Delta.

A trajectory consists of a series of actuator inputs.  The actuators will move to each input 
position and each camera will take a frame.

The important functions to understand are VAC.run_traj() and Computer_Vision.process_video()

--------------------------------------------------

A trajectory is run via VAC.run_traj():
	for each point in trajectory:
		actuators go to specified position and record the time
		both cameras take a picture and record the time

	Each camera processes the videos to identify where the colored dots are in the video,
	outputting the dot locations (in cm) as a function of time (Computer_Vision.process_video)

	The camera trajectories are combined to get a 3d trajectory (vac.combine_cam_trajects())
		The top camera measures (x,y) coordinates
		The side camera measures (x,z) coordinates

		Frames that are close in time are merged with the average x valud being taken

	The 3d Delta trajectory that is measured by the two cameras is associated with the 
	actuator inputs that were commanded at close to the same time

There are comments detailing the process more in the code.  Start with Vac.run_traj() and
look at every method that is called from it.

The code assumes that you can control a Delta to go to a position via VAC.delta.goto_pos()
You have to implement something for delta.goto_pos() to control your actuators

----------------------------------------------------

To process a video, Computer_Vision.process_video() is used:
	I left a lot of comments in the function to make it readable, and I'd just be rewriting them
	here if I wrote pseudocode.

	process_video() takes in a series of frames from a single camera.  Those frames are
	expected to have colored dots in them.  The Side Camera is expected to have two colored dots in view,
	and the top camera is expected to have 3 colored dots in view

	In each frame, the centroids of the dots are found and stored in a list (in pixel coordinates).

	Once this has been done for the whole video, the pixel coordinates are transformed into centimeters
	using the known distance between the dots (you have to measure it).  

	The coordinates are transformed
	so that the mean position of the dots in the first frame lie at (0,0).  It is important to make sure 
	that the first frame is taken while the delta is at whatever you consider to be the origin.

	The mean positions of the dots are returned.

----------------------------------------------------

Debugging:
The cameras detect the dots by searching for anything within a particular threshold of the color stored in the 
config file.  SIDE_A, SIDE_B denote the A and B values of the dots on the side of the Delta when the color is converted
to LAB color space.

To make sure that the cameras are finding the dots correctly, you can process a video with the debug flag,
and it will sequentially show you the frames in the video with the dots highlighted.  You can do a simple test by 
running test_cam(cam_number) in the python interpreter for Computer_Vision.py
i.e. run "python -i Computer_Vision.py" and then type "test_cam(0)" to test the side camera


You can verify the 3D coordinates you are measuring in two ways:
	1: from VAC run x_test() to run a random trajectory.  Afterwards,
	a graph will appear to compare the x values measured by each camera (they should be the same)

	2: from VAC run z_test().  The actuators will be commanded to all move straight up or down
	a few times.  Afterwards, the z value measured by the side camera will be plotted against the
	known Delta height (when all actuators move at the same time, the delta movement is purely z translation)


---------------------------------------------------

VAC.augment_data_with_symmetry():
This is not called anywhere, but I left it in for your reference.
The Delta workspace has multiple symmetries that can significantly augment the data you obtain

1: from an end effector position [x,y,z], you can reach positions [x,y,z+z0] by moving all actuators 
	up or down by z0

2: You can reflect the Delta position over the Y axis by swapping actuators 1 and 2

3: You can rotate the Delta position about z axis by 120 degrees by rotating every actuator position by 120 degreesa

----------------------------------------------------

Getting Started:

1: Read through VAC.run_traj() and all methods that are called within

2: look at bottom of VAC.py to see an example usage

3: set up your Delta with the end effector intersecting the lines of sight of both cameras

3: implement VAC.delta.goto_pos() for your actuators to control the Delta

4: attach colored dots to your delta with different colors on the top and sides
	obtain the LAB values for the colors as seen by your cameras using any color dropper tool.
	On MAC I used Digital Color Meter

5: look through the config file and change:
	TOP_A, TOP_B, SIDE_A, SIDE_B, _DOT_DIST, _CAM_2_DELTA, TOP_CAM, SIDE_CAM


6: run Computer_Vision.test_cam() to make sure you are correctly finding the dots

7: run the lines at the bottom of VAC.py




