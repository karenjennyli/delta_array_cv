import tensorflow.keras as keras
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Activation,BatchNormalization
from tensorflow.keras.layers import Input
from tensorflow.keras.models import load_model
import numpy as np 
import config
from scipy.stats import norm as norm_dist
import random

class NN():
	def __init__(self,ik_save_file="./models/rigid_ik_model",fk_save_file="./models/rigid_fk_model",
			ik_load_file="./models/rigid_ik_model",fk_load_file="./models/rigid_fk_model",load=True):

		self.fk = self.create_fk_model()
		self.ik = self.create_ik_model()

		self.fk_save_file = fk_save_file
		self.ik_save_file = ik_save_file

		self.fk_load_file = fk_load_file
		self.ik_load_file = ik_load_file

		self.ik_Adam = tf.keras.optimizers.Adam(.0005) # learning_rate=.0005

		if(load):
			self.load_all()

		self.mem_buf = []
		self.mem_size = config.MEM_SIZE 
		self.batch_size = config.BATCH_SIZE 
		self.train_per_ep = config.TRAIN_PER_EP

	def evaluate(self,data):
		#data assumed to be [heights,ee_poses,times]
		heights,ee_poses,_ = list(zip(*data))

		fk_pred = self.predict(self.fk,heights)
		ik_pred = self.predict(self.ik,ee_poses)

		fk_error = np.mean(np.sqrt(np.sum(np.square(ee_poses-fk_pred),axis=1)))
		ik_error = np.mean(np.sqrt(np.sum(np.square(heights-ik_pred),axis=1)))

		fk_diff = np.mean(ee_poses-fk_pred)
		ik_diff = np.mean(heights-ik_pred)

		print(" fk error ",fk_error)
		print(" ik error ",ik_error)

		print(" fk avg difference", fk_diff)
		print(" ik avg difference", ik_diff)

		print("\n-----------------------\n")

		return fk_error,ik_error

	def evaluate_detailed(self, data):
		#data assumed to be [heights,ee_poses,times]
		heights,ee_poses,_ = list(zip(*data))

		fk_pred = self.predict(self.fk,heights)
		ik_pred = self.predict(self.ik,ee_poses)
		
		print(" fk pred ")
		print(fk_pred)
		print()
		print(" ik pred ")
		print(ik_pred)
		print()

		fk_error = np.mean(np.sqrt(np.sum(np.square(ee_poses-fk_pred),axis=1)))
		ik_error = np.mean(np.sqrt(np.sum(np.square(heights-ik_pred),axis=1)))
		# fk_error = ee_poses-fk_pred
		# ik_error = heighti s-ik_pred

		print(" fk error ",fk_error)
		print(" ik error ",ik_error)

		print("\n-----------------------\n")

		return fk_error,ik_error

	def train_on_batch(self,heights,ee_poses):
		#get a random batch from memory and train on it

		'''
		fk is learned such that input actuator heights map to measured
		end effector positions

		ik is learned such that fk(ik(ee_poses)) = ee_poses
		You can also just do self.ik.fit(ee_poses,heights)
		but it seemed better to enforce consistency that fk(ik(x)) = x

		when the forward and inverse kinematics were learned from ground truth
		independently, the errors in the predictions are about the same
		'''

		heights = np.array(heights)
		ee_poses = np.array(ee_poses)

		self.fk.fit(heights,ee_poses,verbose=0) # verbose is wheter or not you want to see progress

		with tf.GradientTape(watch_accessed_variables=False) as g:
			g.watch(self.ik.trainable_weights)
			ik_out = self.ik(ee_poses)
			fk_out = self.fk(ik_out)
			
			ik_loss = tf.reduce_mean(tf.keras.losses.MSE(fk_out,ee_poses))
			ik_grads = g.gradient(ik_loss,self.ik.trainable_weights)
		self.ik_Adam.apply_gradients(zip(ik_grads,self.ik.trainable_weights))

	def train_from_mem(self):
		if len(self.mem_buf) == 0:
			return
		for i in range(self.train_per_ep):
			heights,ee_poses = self.sample_memory()
			self.train_on_batch(heights,ee_poses)


	def predict(self,model,x):
		x_in = np.array(x)

		x_out = model.predict(x_in)
		return x_out

	def remember(self,data):
		self.mem_buf.extend(data)
		if len(self.mem_buf) > self.mem_size:
			self.mem_buf = self.mem_buf[len(self.mem_buf)-self.mem_size:]

	def sample_memory(self,batch_size=None):
		#get a random sample from memory buffer

		if batch_size is None:
			batch_size = self.batch_size
		batch = random.choices(self.mem_buf,k=self.batch_size)
		heights,ee_locs,_ = list(zip(*batch))
		return heights,ee_locs


	def create_fk_model(self,input_dim = 3, output_dim = 3):
		model = Sequential()
		model.add(Dense(256, input_dim=input_dim, activation='relu')) # input layer (relu makes any negative numbers 0)
		model.add(Dense(256, activation='relu')) # two hidden layers
		model.add(Dense(256, activation='relu')) # layer size? / output shape = 256
		model.add(Dense(output_dim, activation='linear')) # output layer

		model.compile(optimizer=keras.optimizers.Adam(),
               loss='MSE') # mean square error
		return model


	def create_ik_model(self,input_dim = 3, output_dim = 3):
		model = Sequential()
		model.add(Dense(256, input_dim=input_dim, activation='relu'))
		model.add(Dense(256, activation='relu'))
		model.add(Dense(256, activation='relu'))
		model.add(Dense(output_dim, activation='linear'))
		# why not compile model too?
		return model

	def save_all(self):
		self.save(self.fk,self.fk_save_file)
		self.save(self.ik,self.ik_save_file)

	def load_all(self):
		self.load_fk()
		self.load_ik()

	def load(self,model_file):
		return keras.models.load_model(model_file)

	def load_fk(self):
		self.fk = self.load(self.fk_load_file)

	def load_ik(self):
		self.ik = self.load(self.ik_load_file)

	def save(self,model,model_file):
		model.save(model_file)

	def sample_workspace(self,max_height,sample_dens):
		#run fk on every possible c-space configuration
		
		num_pts = int(max_height/sample_dens)
		heights = np.empty((num_pts**3,3))
		idx = 0
		for i in range(num_pts):
			for j in range(num_pts):
				for k in range(num_pts):
					heights[idx,:] = sample_dens*np.array([i,j,k])
					idx += 1
		pred = self.predict(self.fk,heights)
		return pred


def read_data_from_file(training_data_path):
	#assume data format [height1,height2,height3:x,y,z]

	f = open(training_data_path)
	data = []
	for line in f.readlines():
		l = line.strip()
		d = l.split(":")
		heights = [float(x) for x in d[1].split(",")]
		ee_poses = [float(x) for x in d[2].split(",")]
		data.append((heights,ee_poses,0))
	return data

def train_from_file(training_data_path,dbg=False):
	model = NN(ik_save_file="./models/rigid_ik_model",fk_save_file="./models/rigid_fk_model",
			ik_load_file="./models/rigid_ik_model",fk_load_file="./models/rigid_fk_model",load=False)
	
	data = read_data_from_file(training_data_path)

	if dbg:
		print("Read in",len(data),"datapoints")

	model.remember(data)

	for i in range(300):
		print(i)
		model.train_from_mem()
		model.evaluate(data)

	model.save_all()

def predict_from_model(training_data_path,dbg=False):
	model = NN(ik_save_file="./models/rigid_ik_model",fk_save_file="./models/rigid_fk_model",
		ik_load_file="./models/rigid_ik_model",fk_load_file="./models/rigid_fk_model",load=True)

	data = read_data_from_file(training_data_path)

	if dbg:
		print("Read in",len(data),"datapoints")

	model.remember(data)

	model.evaluate(data)

def compare_repeated_trials(trial1_data_path, trial2_data_path):
	data1 = read_data_from_file(trial1_data_path)
	data2 = read_data_from_file(trial2_data_path)

	heights1,ee_poses1,_ = list(zip(*data1))
	heights2,ee_poses2,_ = list(zip(*data2))

	# print(ee_poses1)
	# print("-------------------")
	# print(ee_poses2)
	mse_error = np.mean(np.sqrt(np.sum(np.square(ee_poses1-np.asarray(ee_poses2)),axis=1)))
	ee_error = np.mean(ee_poses1-np.asarray(ee_poses2))

	print("avg difference ", ee_error)
	print("mse ", mse_error)
	return ee_error

# train_from_file("./Measured_Poses/training_data_1024.txt", dbg=True)

# predict_from_model("./Measured_Poses/all_data.txt", dbg=True) # training data
# predict_from_model("./Measured_Poses/test_data.txt", dbg=True) # testing data
# predict_from_model("./prev_data/test_data1_first_model.txt", dbg=True) # testing data
# predict_from_model("./Measured_Poses/Traj.txt", dbg=True) # testing data
# compare_repeated_trials("./Measured_Poses/trial1.txt", "./Measured_Poses/trial2.txt")

predict_from_model("./Measured_Poses/Traj.txt", dbg=True) # testing on 3rd finger

print("Finished.")

'''
This code is written with online learning in mind, but you can also load in a data file and use
train_from_file

If learning online, do roughly this:
For Each Episode:
	move Delta to desired positions and collect data
	D <- collected data
	model.remember(D)
	For i = 1:TRAIN_PER_EP:
		model.train_on_batch()

The model has a memory buffer of the collected data, with a maximum buffer size
model.train_on_batch() draws a random sampling from the memory buffer and does one 
pass of training the model

When I was learning on the delta, I defined an episode as commanding the delta to go to 
a few random positions, and then returning to the origin.  I recorded the Delta's position
every .25 cm on its path.  There isn't a great reason to do it this way instead of recording 
all of the data first, unless you are commanding task-space positions
i.e. you command where the end effector should go, and as it learns it will get better and better at
following those commands.  My idea when I structured it this way was to command it to positions
within the workspace where the fk and ik disagree until there aren't any left.
It turned out that just going to random c-space positions a bunch of times was sufficient, so taking all of 
your data first is fine.  It would definitely help to go to every position multiple times though.  
'''


