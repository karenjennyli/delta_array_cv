import cv2
import numpy as np 
import config
import time
import copy
from multiprocessing import Pool

class C920_Params():
	def __init__(self,cam):
		if cam == 0:
			self.mtx = np.array(config.TOP_CAM_MTX)
			self.dist = np.array(config.TOP_CAM_DIST)
		else:
			self.mtx = np.array(config.SIDE_CAM_MTX)
			self.dist = np.array(config.SIDE_CAM_DIST)

		h = config.CAM_BOUND[1]
		w = config.CAM_BOUND[3]
		self.newcameramtx,_ = cv2.getOptimalNewCameraMatrix(self.mtx,self.dist,(w,h),0,(w,h))

class Camera_Reader():
	def __init__(self,cam,view):
		self.cap = cv2.VideoCapture(cam)
		self.recording = False
		self.view = view

		if view == 0:
			self.threshA = config.TOP_A 
			self.threshB = config.TOP_B 
			self.num_dots = 3

			self.dot_dist = config.TOP_DOT_DIST
		else:
			self.threshA = config.SIDE_A 
			self.threshB = config.SIDE_B 
			self.num_dots = 2


			self.dot_dist = config.SIDE_DOT_DIST
		self.cam_params = C920_Params(view)

		self.last_vid = []
		self.last_vid_process = None
		self.thresh_size = config.THRESH_SIZE
		self.min_dot_volume = config.MIN_VOL

		self.skipped_frames = []
		self.skipped_frames_idx = set()

	def add_skipped_frame(self,frame, frame_idx=-1):
		#frames are skipped when things went wrong, so I store them to debug
		max_num = 100
		self.skipped_frames.append(frame)
		self.skipped_frames_idx.add(frame_idx)
		if len(self.skipped_frames) > max_num:
			self.skipped_frames = self.skipped_frames[len(self.skipped_frames)-max_num:]


	def record_frame(self,frame_info,t=None):
		ret,frame = self.cap.read() # ret is whether frame was captured (true/false)
		if self.view == 0:
			print(f"Top: {ret}")
		else:
			print(f"Side: {ret}")
		if t == None:
			t = time.time()
		if ret and np.shape(frame) != ():
			frame_info.append([frame,t])

	def undistort_frame(self,img):
		#this doesn't actually seem to make a big difference
		#but computer vision people say you should
		mtx = self.cam_params.mtx
		dist = self.cam_params.dist
		newcameramtx = self.cam_params.newcameramtx
		
		dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

		return dst

	def undistort_frames(self,frames):
		print(f"Number of frames: {len(frames)}")
		imgs,times = list(zip(*frames))
		dsts = [self.undistort_frame(img) for img in imgs]
		return list(zip(dsts,times))

	def record_video_and_process(self,duration):
		#record video for specified duration and return processed position information
		frame_info = []

		start = time.time()
		self.last_vid = []
		while time.time() - start < duration:
			ret,frame = self.cap.read()
			if ret:
				frame_info.append((frame,time.time()))
				self.last_vid.append(frame)
				#time.sleep(fps)

		self.last_vid_process = (self.process_video(frame_info))

	def process_video(self,frames,dbg=False): # ``frames`` = [frame, t]
		#frames is a list of tuples (image,timestamp)

		abs_bound = config.CAM_BOUND # (0,1080,0,1920)

		#in the worst case search the whole image for dots
		bound = list(abs_bound)

		center_poses = []

		keep_times = []
		for frame_idx,frame in enumerate(frames):

			#crop image according to bounds
			img = frame[0][bound[0]:bound[1],bound[2]:bound[3]]

			#convert image to LAB colors
			img_LAB = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

			#create a mask for colors that are within the threshhold of the dot colors
			lower_bound = np.array([0,self.threshA-self.thresh_size,self.threshB-self.thresh_size],dtype="uint8")
			upper_bound = np.array([255,self.threshA+self.thresh_size,self.threshB+self.thresh_size],dtype="uint8")
			mask = cv2.inRange(img_LAB,lower_bound,upper_bound)

			
			#get the border points around each Delta dot	
			contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


			# save images
			# drawn_contours = cv2.drawContours(img, contours, -1, (0,255,0), 3)
			# if self.view == 0:
			# 	cv2.imwrite(f"frames/frame{frame_idx}_top.jpg", drawn_contours)
			# else:
			# 	cv2.imwrite(f"frames/frame{frame_idx}_side.jpg", drawn_contours)
			

			if len(contours) < self.num_dots:
				#didn't find enough dots in cropped image, so check the whole image

				bound = copy.copy(abs_bound)
				img = frame[0][bound[0]:bound[1],bound[2]:bound[3]]

				img_LAB = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

				lower_bound = np.array([0,self.threshA-self.thresh_size,self.threshB-self.thresh_size],dtype="uint8")
				upper_bound = np.array([255,self.threshA+self.thresh_size,self.threshB+self.thresh_size],dtype="uint8")
				mask = cv2.inRange(img_LAB,lower_bound,upper_bound)
					
				contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

				if len(contours) < self.num_dots:
					if dbg:
						print("Not enough dots ....")
						cv2.imshow("image",img[:,:,:])
						cv2.waitKey(0)
						cv2.imshow("image",frame[0])
						cv2.waitKey(0)
						cv2.destroyAllWindows()

					else:
						if self.view == 0:
							print(f"Not enought dots in whole image ... Skipping Frame {frame_idx} (top)")
						else:
							print(f"Not enought dots in whole image ... Skipping Frame {frame_idx} (side)")
					self.add_skipped_frame(frame, frame_idx)
					continue		

			#keep the contours with the most points
			maxes = np.zeros(self.num_dots)
			keep_contours = [None for i in range(self.num_dots)]
			for c in contours:
				if np.min(maxes) < len(c):
					min_idx = np.argmin(maxes)
					maxes[min_idx] = len(c)
					keep_contours[min_idx] = c


			centers = []

			lb = [bound[0],bound[2]] #y,x lower bound
			#rearrange things because camera coordinates are y,x
			bound = [bound[1],bound[0],bound[3],bound[2]]
			skip_frame = False
			for c in keep_contours:
				M = cv2.moments(c)
				if M['m00'] < self.min_dot_volume:
					if self.view == 0:
						print(f"Dots too small ... Skipping Frame {frame_idx} (top)")
					else:
						print(f"Dots too small ... Skipping Frame {frame_idx} (side)")
					if dbg:
						cv2.imshow("image",img[:,:,:])
						cv2.waitKey(0)
						cv2.imshow("image",frame[0])
						cv2.waitKey(0)
						cv2.destroyAllWindows()
					self.add_skipped_frame(frame, frame_idx)
					bound = [bound[1],bound[0],bound[3],bound[2]]
					skip_frame = True
					break

				#get centers of contours
				cx = int(M['m10']/M['m00']) + lb[1]
				cy = int(M['m01']/M['m00']) + lb[0]


				#This commented out code gets the center of the top of the dots
				#from the side cam.  This is useful because the bottom of my dots 
				#was getting occluded, and it was shifting the dot centers

				# if self.num_dots == 2:
				# 	x_cnt_err = np.square(c[:,:,0]-cx)
					
				# 	samples = np.min([10,len(x_cnt_err)-1])
				# 	center_idxs = np.argpartition(x_cnt_err[:,0],samples)[:samples]

				# 	cy = np.amin(c[center_idxs,:,1]) + lb[0]
				# else:
				# 	cy = int(M['m01']/M['m00']) + lb[0]
				centers.append((cx,cy))

				bound[0] = min(bound[0],np.min(c[:,:,1],axis=0)[0]+lb[0])
				bound[1] = max(bound[1],np.max(c[:,:,1],axis=0)[0]+lb[0])
				bound[2] = min(bound[2],np.min(c[:,:,0],axis=0)[0]+lb[1])
				bound[3] = max(bound[3],np.max(c[:,:,0],axis=0)[0]+lb[1])




			bound[0] = max(abs_bound[0],bound[0]-config.EXTEND_BOUND)
			bound[1] = min(abs_bound[1],bound[1]+config.EXTEND_BOUND)
			bound[2] = max(abs_bound[2],bound[2]-config.EXTEND_BOUND)
			bound[3] = min(abs_bound[3],bound[3]+config.EXTEND_BOUND)	

			if skip_frame:
				continue

			center_poses.append(centers)
			keep_times.append(frame[1])

		if len(center_poses) == 0:
			print("No poses")
			return None,None


		center_poses = np.array(center_poses) # stores [[(x,y), (x,y)], [(x,y), (x,y)]]
		times = np.array(keep_times)

		#mean of dots in first frame
		init_center = np.mean(center_poses[0,:,:],axis = 0)

		#subtract out pixel position of the center of the dots
		# in first frame from centers in all other frames so that the first frame is the origin
		sub = np.ones(np.shape(center_poses))
		sub[:,:,0] = init_center[0]
		sub[:,:,1] = init_center[1]
		
		#get the distance between dot centers in pixels
		pix_dot_dist = np.linalg.norm(center_poses[0,0,:]-center_poses[0,1,:])
		#self.dot_dist/pix_dot_dist converts pixel distance to cm.
		adjusted_centers = (center_poses-sub)*self.dot_dist/pix_dot_dist


		mean_poses = np.mean(adjusted_centers,axis=1)
		if dbg:
			test_mean = np.mean(center_poses,axis=1)
			for frame_idx,frame in enumerate(frames):
				if frame_idx >= len(center_poses):
					break
				[cv2.circle(frame[0],(int(cx),int(cy)),20,(0,0,255),10) for (cx,cy) in center_poses[frame_idx,:,:]]
				cam_center = self.cam_params.mtx[:2,2]
				cv2.circle(frame[0],(int(cam_center[0]),int(cam_center[1])),10,(255,255,255),10)
				cv2.circle(frame[0],(int(test_mean[frame_idx,:][0]),int(test_mean[frame_idx,:][1])),5,(255,0,255),10)
				cv2.imshow("image",frame[0])
				cv2.waitKey(0)
				cv2.destroyAllWindows()

		return adjusted_centers,times, self.skipped_frames_idx

	def debug_last_video(self):
		times = np.zeros(len(self.last_vid))
		self.process_video(list(zip(self.last_vid,times)),dbg=True)

	def debug_skipped_frames(self):
		self.process_video(copy.copy((self.skipped_frames)),dbg=True)

	def test(self):
		time.sleep(2)
		frames = []
		for i in range(5): c1.record_frame([])
		for i in range(50):
			self.record_frame(frames)
		self.process_video(frames,dbg=True)


def test_cam(cam_num, view):
	#runs a camera for a few frames and shows the tracked dots and delta center
	# if cam_num == 0:
	# 	c1 = Camera_Reader(config.SIDE_CAM,1)
	# else:
	# 	c1 = Camera_Reader(config.TOP_CAM,0)
	c1 = Camera_Reader(cam_num, view)
	time.sleep(1)
	frames = []
	for i in range(5): c1.record_frame([])
	for i in range(5):
		c1.record_frame(frames)
	frames = c1.undistort_frames(frames)
	c1.process_video(frames,dbg=True)