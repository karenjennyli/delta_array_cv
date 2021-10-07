import matplotlib
from Computer_Vision import Camera_Reader
from Computer_Vision import test_cam
from Control_Arduino import Control_Delta
import config
import threading
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
import time
from scipy.spatial.transform import Rotation as Rot
import math

import pdb


class Vision_Actuator_Coordinator():
	def __init__(self):
		self.side_cam = Camera_Reader(config.SIDE_CAM,1)
		self.top_cam = Camera_Reader(config.TOP_CAM,0)

		self.delta = Control_Delta() #make your own version of Control Delta

		self.init_pt = config.INIT_PT #add offset to Delta so that measured values are relative to the base

		self.side_cam_2_delta = config.SIDE_CAM_2_DELTA
		self.top_cam_2_delta = config.TOP_CAM_2_DELTA

		self.positions = []


	def combine_cam_trajects(self,s_times,s_dot_centers,t_times,t_dot_centers, skipped_frames, plot_x=False):
		#combine footage from side camera and top camera to get 3D view of Delta

		'''
		Args:
			s_times: times when the side camera took a picture
			s_dot_centers: center of dots seen by side camera (in cm)
			etc.
		'''

		pts = np.zeros((len(s_times),3)) #3D position of Delta
		pts = []
		kept_times = []

		s_dot_centers *= -1 #flip side camera x and z axes
		t_dot_centers[:,:,1] *= -1 #flip top camera y axis

		#for plotting x coordinates of delta to make sure the side camera and top
		#camera give consistent values for the x axis
		x_sc_lst = []
		x_tc_lst = []

		for i,s_t in enumerate(s_times):
			#find closest time from t
			#skip point if there is no corresponding image from top camera within .1 seconds
			t_ind = np.argmin(np.abs(t_times-s_t))
			if abs(t_times[t_ind]-s_t) > .1 or i in skipped_frames:
				print("Skipping point")
				continue

			s_cnts = s_dot_centers[i]
			t_cnts = t_dot_centers[t_ind]

			s_cnt = np.mean(s_cnts,axis=0) # center of dots for a given time
			t_cnt = np.mean(t_cnts,axis=0)


			#scale measured values based on how much the delta has moved towards the camera
			#if the delta moved 2x closer to the camera, relative to the camera center,
			#the dots would appear 2x higher.  This combines the information from the top
			# and side camera to account for the scaling
			#It iterates three times because it converges very fast.
			y = t_cnt[1] # y-coordinate of top camera
			for i in range(3):
				side_cam_scale = (self.side_cam_2_delta-y)/self.side_cam_2_delta
				z = s_cnt[1]*side_cam_scale

				top_cam_scale = (self.top_cam_2_delta-z)/self.top_cam_2_delta
				y = t_cnt[1]*top_cam_scale

			x_sc = s_cnt[0]*side_cam_scale
			x_tc = t_cnt[0]*top_cam_scale
			x_sc_lst.append(x_sc)
			x_tc_lst.append(x_tc)

			x = (x_sc + x_tc)/2
			pts.append([x,y,z])
			kept_times.append(s_t)

		pts = np.array(pts)

		if plot_x: self.compare_x(s_times,t_times,x_tc_lst,x_sc_lst)

		return np.array(pts),np.array(kept_times)

	def sync_actuator_with_cam(self,delta_traj,delta_times,cam_traj,cam_times):
		'''
		Args:
			delta_traj: actuator heights
			delta_times: times when delta stopped to have a picture taken

			cam_traj: 3D trajectory found by cameras
			cam_times: times when a picture was taken
		'''

		data = []

		cam_times -= delta_times[0]
		delta_times -= delta_times[0]

		for i,c_t in enumerate(cam_times):
			d_ind = np.argmin(np.abs(delta_times-c_t))
			data.append((delta_traj[d_ind],cam_traj[i]+self.init_pt,c_t))

		return data

	def rotz(self,vec,rad):
		#rotate vec about the z axis by rad.
		rotation_axis = np.array([0, 0, 1])
		rotation_vector = rad * rotation_axis
		rotation = Rot.from_rotvec(rotation_vector)
		rotated_vec = rotation.apply(vec)
		return rotated_vec

	def augment_data_with_symmetry(self,data):
		#triples training data by reassigning actuators and rotating workspace point by 120 degrees
		#Solves problem of regions of workspace being occluded to camera

		ref_data = []

		#add reflection over y axis by swapping actuators 1 and 2

		# for h,pos,t in data:
		# 	ref_data.append((h,pos,t))

		# 	h_ref = [h[0],h[2],h[1]]
		# 	pos_ref = [-pos[0],pos[1],pos[2]]
		# 	ref_data.append((h_ref,pos_ref,t))

		#add rotation by +-120 degrees by rotating all actuators +- 120

		# rot_data = []
		# for h,pos,t in ref_data:
		# 	rot_data.append((h,pos,t))

		# 	h1 = [h[2],h[0],h[1]]
		# 	pos1 = self.rotz(pos,-2*np.pi/3)
		# 	rot_data.append((h1,pos1,t))

		# 	h2 = [h[1],h[2],h[0]]
		# 	pos2 = self.rotz(pos,2*np.pi/3)
		# 	rot_data.append((h2,pos2,t))
		rot_data = data

		#augment data by virtually moving actuators all up or all down to cover entire stroke
		aug_data = []
		step = .1
		for h,pos,t in rot_data:
			low = np.min(h)
			high = np.max(h)

			height_steps = np.tile(np.arange(-low,7-high,step),(3,1)).transpose()
			heights = h + height_steps
			poses = np.tile(pos,(len(heights),1))
			poses[:,2] += height_steps[:,0]
			aug_data.extend(list(zip(heights,poses,np.ones(len(height_steps))*t)))

		return aug_data


	def run_traj_accurate(self,finger_number,traj,pt_space=.5,plot_x=False,plot_z=False,dbg_side_cam=False,dbg_top_cam=False):
		'''
		Args:
			traj: desired heights of actuators # Numpy array

			pt_space: if a traj of [[0,0,0],[1,0,0]] were given, and pt_space=.5,
				the trajectory would be converted to [[0,0,0],[.5,0,0],[1,0,0]]

			plot_x: compare x measurements of both cameras for debugging

			plot_z: compare z measurement by side camera to known heights of actuators
				This is only helpful if all the actuators move up and down at the same time
				so that you know exactly how the delta should have moved.

			dbg_cam: show the frames from a camera with the tracked points and camera center highlighted
		'''

		side_frames = []
		top_frames = []
		delta_poses = []
		delta_times = []

		#record some ununsed camera frames to "warm up" the camera
		#the first few frames are sometimes black when things have just been loaded
		for i in range(3):
			self.side_cam.record_frame([])
			self.top_cam.record_frame([])


		traj = self.pt_space(traj,spacing=pt_space)

		for i,pt in enumerate(traj):
			print(f"Point {i} of {len(traj)}")
			#you have to implement this for your delta
			position =  self.delta.goto_pos(pt, finger_number)
			self.positions.append((i, pt, position))


			delta_poses.append(pt)
			delta_times.append(i)
			self.side_cam.record_frame(side_frames,t=i)
			self.top_cam.record_frame(top_frames,t=i) # ``top_frame`` = [frame, t]


		self.side_cam.undistort_frames(side_frames)
		s_dot_centers,s_times, s_skipped_frames_idx = self.side_cam.process_video(side_frames,dbg=dbg_side_cam)
		self.top_cam.undistort_frames(top_frames)
		t_dot_centers,t_times, t_skipped_frames_idx = self.top_cam.process_video(top_frames,dbg=dbg_top_cam)
		delta_poses,delta_times = np.array(delta_poses),np.array(delta_times)

		skipped_frames = s_skipped_frames_idx.union(t_skipped_frames_idx)

		cam_traj,cam_times = self.combine_cam_trajects(s_times,s_dot_centers,t_times,t_dot_centers, skipped_frames, plot_x=plot_x)

		datapoints = self.sync_actuator_with_cam(delta_poses,delta_times,cam_traj,cam_times)

		if plot_z: self.compare_z(datapoints)

		return datapoints


	def compare_z(self,data):
		#test accuracy of cameras in measuring value of z
		#give data where all actuators have the same height

		plt.figure()
		delta_pos,cam_pos,time = list(zip(*data))
		delta_pos = np.array(delta_pos) + self.init_pt
		time = np.array(time)
		cam_pos = np.array(cam_pos)

		error = np.abs(delta_pos[:,2]-cam_pos[:,2])
		print("Mean z error:",np.mean(error))
		print("Std z error:",np.std(error))
		print("Max z error:",np.max(error))

		plt.plot(time,delta_pos[:,2],"g")
		plt.plot(time,cam_pos[:,2],"b")

		axis = plt.gca()
		axis.set_xlabel("Time (seconds)")
		axis.set_ylabel("Z Position (cm)")
		plt.title("Comparison of Actuator z values with Camera-Measured z values")
		axis.legend(["Actuator Position","Side Camera Measurement"])
		plt.show()

	def compare_x(self,s_times,t_times,x_tc,x_sc):
		try:
			plt.figure()
			plt.plot(s_times-s_times[0],x_sc)
			plt.plot(t_times-t_times[0],x_tc)
			axis = plt.gca()
			axis.set_xlabel("Time (seconds)")
			axis.set_ylabel("X position (cm)")
			plt.title("Comparison of Camera Measurements in X direction")
			axis.legend(["Side Camera","Top Camera"])
			plt.show()
		except:
			pass

	def plot_dot_means(self,times,mean_poses,label1,label2):
		plt.subplot(2,1,1)
		plt.plot(times-times[0],mean_poses[0,:])
		axis = plt.gca()
		axis.set_ylabel(label1)
		axis.set_xlabel("time")
		plt.subplot(2,1,2)
		plt.plot(times-times[0],mean_poses[1,:])
		axis = plt.gca()
		axis.set_ylabel(label2)
		axis.set_xlabel("time")

	def pt_space(self,pts,spacing=.5):
		result = []
		for i in range(len(pts)-1):
			pt = np.array(pts[i])
			next_pt = np.array(pts[i+1])
			dist = np.max(np.abs(next_pt-pt))
			num_pts = max(1,int(dist/spacing))
			result.extend(np.linspace(pt,next_pt,num=num_pts,endpoint=False).round(1))
		result.append(pts[-1])
		return result

	def plot3(self,traj,color="green",title="",ax = None):
		if ax is None:
			ax = plt.axes(projection='3d')

		ax.scatter3D(traj[:,0],traj[:,1],traj[:,2],color)
		ax.set_xlabel("x")
		ax.set_ylabel("y")
		ax.set_zlabel("z")
		if title != "":
			plt.title(title)
		return ax


	def write_traj(self,data,filename="./Measured_Poses/Traj.txt"):
		#write a measured trajectory to file
		file = open(filename,"w+")
		for h,p,t in data:
			file.write(str(t)+":"+str(h[0])+","+str(h[1])+","+
				str(h[2])+":"+str(p[0])+","+str(p[1])+","+str(p[2])+"\n")

	def write_positions(self,positions,filename="./Measured_Poses/Positions.txt"):
		#write a desired and potentiometer positions to file
		file = open(filename,"w+")
		for (t, pt, p) in self.positions: # time, input, potentiometer reading
			file.write(str(t)+":"+str(pt[0])+","+str(pt[1])+","+
				str(pt[2])+":"+str(p[0])+","+str(p[1])+","+str(p[2])+"\n")

	def sample_workspace(self,max_height):
		# a trajectory that samples the entire workspace
		# I didn't end up using this
		traj = []
		a_count = 0
		inc = .25
		for a in range(0,int(max_height/inc)):
			a = a*inc
			a_count += 1
			if a_count%2 == 1:
				b_start = 0
				b_end = int(max_height/inc)
				b_inc = 1
			else:
				b_start = int(max_height/inc)
				b_end = -1
				b_inc = -1
			b_count = 0
			for b in range(b_start,b_end,b_inc):
				b_count += 1
				b = b*inc
				if b_count%2 == 1:
					traj.append([a,b,max_height])
				else:
					traj.append([a,b,0])
		return traj


def x_test(vac,finger_number):
	traj = np.random.rand(5,3)*4 + 1 # ``+ 1`` so never goes to 0
	vac.run_traj_accurate(finger_number,traj,pt_space=.5,plot_x=True)


def z_test(vac,finger_number,pos=[1,1,1],pt_space=.5): # changed default pos = [1,1,1]
	traj = np.array([[1,1,1],[2.5,2.5,2.5],[1.5,1.5,1.5],[2.5,2.5,2.5],[4.5,4.5,4.5],[3.5,3.5,3.5]])
	traj[1:] += np.array(pos)

	data = vac.run_traj_accurate(finger_number,traj,pt_space=pt_space,plot_z=False)
	heights,ee_poses,t = list(zip(*data))
	start_idx = max(1,int(np.max(pos)/pt_space))
	heights = np.array(heights)
	ee_poses = np.array(ee_poses)
	plt.plot(t[start_idx:],heights[start_idx:,0]-heights[start_idx,0])
	plt.plot(t[start_idx:],ee_poses[start_idx:,2]-ee_poses[start_idx,2])
	axis = plt.gca()
	axis.set_ylabel("Z Position (cm)")
	plt.title("Comparison of Actuator z values with Camera-Measured z values")
	axis.legend(["Actuator Position","Side Camera Measurement"])
	plt.show()

def run_path(vac,finger_number,traj,pt_space=.5,plot_x=False):
	#run a path and record

	data = vac.run_traj_accurate(finger_number,traj,pt_space=pt_space,plot_x=plot_x,dbg_side_cam=False)
	heights,ee_poses,_ = list(zip(*data))
	ax = vac.plot3(np.array(ee_poses),color="green")
	plt.show()
	return data

def read_traj_from_file(traj_data_path):
	f = open(traj_data_path)
	data = []
	i = 0
	for line in f.readlines():
		l = line.strip()
		d = l.split(":")
		heights = [float(x) for x in d[1].split(",")]
		data.append(heights)
		i += 1
	return data

def collect_training_data(vac, finger_number):
    '''
    collect training data, change each actuator by an interval of .05cm
    '''
    origin = np.array([[1,1,1]])
    traj = []

    for i in range(8):
        for j in range(8):
            for k in range(8):
                first = 1 + i * 0.5
                second = 1 + j * 0.5
                third = 1 + k * 0.5
                traj.append([first, second, third])

    traj = np.array(traj) # convert to numpy array
    traj = np.concatenate((traj, traj)) # duplicate traj

    datapoints = run_path(vac, finger_number, traj, pt_space=10, plot_x=True) # ``pt_space=10`` so no spacing is added
    vac.write_traj(datapoints, file_name=f"./Measured_Poses/Traj_Training_Data_{finger_number}.txt") # save datapoints to text file

def collect_testing_data(vac, finger_number):
    '''
    collect testing data
    '''
    number_of_points = 100 # number of random datapoints for testing data
    origin = np.array([[1,1,1]])
    traj = np.random.rand(number_of_points,3).round(1)*4 + 1
    traj = np.concatenate((origin, traj))
    traj = np.concatenate((traj, traj))
    datapoints = run_path(vac, finger_number, traj, pt_space=10, plot_x=True) # ``pt_space=10`` so no spacing is added
    vac.write_traj(datapoints, file_name=f"./Measured_Poses/Traj_Testing_Data_{finger_number}.txt") # save datapoints to text file
    # vac.write_positions(vac.positions, file_name=f"./Measured_Poses/Positions_Testing_Data_{finger_number}.txt") # save desired position and potentiometer reading position (not necessary)

def rerun_traj(vac, finger_number):
    '''
    if you ever need to rerun a trajectory, run this code
    '''
    traj = read_traj_from_file("./Measured_Poses/Traj_Training_Data_{finger_number}.txt")
    datapoints = run_path(vac, finger_number, traj, pt_space=10, plot_x=True) # ``pt_space=10`` so no spacing is added
    vac.write_traj(datapoints, file_name=f"./Measured_Poses/Traj_Training_Data_{finger_number}.txt") # save datapoints to text file

def run_tests(vac, finger_number):
    '''
    run x_test and z_test to make sure everything is working
    x_test:
        make sure top_cam and side_cam measurements are similar
    z_test:
        # run a trajectory that moves all the actuators up and down
        # use this to make sure that your z measurements are consistent from different places
        # i.e. no matter where the delta starts from, if all actuators move up 3 cm, you should measure 3 cm with the cams
    '''
    x_test(vac, finger_number) # run a random trajectory to compare x axis measurements from side and top cam
    z_test(vac, finger_number)


vac = Vision_Actuator_Coordinator()
finger_number = 0 # finger number from 1 to 4


traj = [[2.5,2.5,2.5],[1.5,1.5,1.5],[2.5,2.5,2.5],[5.5,5.5,5.5],[3.5,3.5,3.5]] # example traj
datapoints = run_path(vac, finger_number, traj, pt_space=10, plot_x=True)
print(datapoints)

print("Finished.")

