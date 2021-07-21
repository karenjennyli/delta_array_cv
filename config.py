#Threshhold values for in L*A*B* color space
#128 is added because macs measure the color space from -128 to 127, but opencv measures it from 0 to 255
TOP_A = -47 + 128 #mean threshhold values for A component of dots on top of Delta
TOP_B = 41 + 128 #mean threshhold values for B component of dots on top of Delta 

SIDE_A = 85 + 128
SIDE_B = -8 + 128

THRESH_SIZE = 30 # 20 #threshhold for detecting dot color
MIN_VOL = 40 # 100 #dots must have this volume in pixels to be considered dots

#Distance in cm between dots on delta
TOP_DOT_DIST = 1.087 # 1.25
SIDE_DOT_DIST = 0.997 # .9

SIDE_CAM_2_DELTA = 30 # 14.8 #cm from side camera to Delta center at rest
TOP_CAM_2_DELTA = 22 # 25.7 #cm from top camera to Delta center

#Webcam number of each camera (depends on where you plugged them in)
#cv2.VideoCapture(cam_number) will pull up the feed from one of your webcams, you can figure out which
#one is what number by trial and error.
TOP_CAM = 1
SIDE_CAM = 3

#Computer vision usually crops image before processing to avoid processing whole image
CAM_BOUND = (0,1080,0,1920) #size of whole image
CAM_FPS = 30
EXTEND_BOUND = 300 # adds to height/width of crop defined by contours of last dots found

#Camera Calibration Values
SIDE_CAM_DIST = [0.02335574,  0.02770726, -0.00443935,  0.00311229, -0.49226571]
SIDE_CAM_MTX = [[1.43626106e+03, 0.00000000e+00, 8.00264420e+02],
 [0.00000000e+00, 1.43188745e+03, 5.43171139e+02],
 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]

TOP_CAM_DIST = [ 7.49561861e-02, -2.84826215e-01, -2.83749938e-03,  1.15894756e-04,
   2.45673600e-01]
TOP_CAM_MTX = [[1.44592804e+03, 0.00000000e+00, 7.72256602e+02],
 [0.00000000e+00, 1.44685590e+03, 5.45889277e+02],
 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]

#the two Deltas on mounting have flipped coordinate frames
#LEFT_DELTA implies the Delta near the camera

INIT_PT = [0,0,4.5398]


