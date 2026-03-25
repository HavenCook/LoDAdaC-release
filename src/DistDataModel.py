import os
import torch
import numpy as np 
from torch.utils.data import DataLoader as DL
from torch.utils.data.sampler import SubsetRandomSampler as SRS
import models
import torch.nn as nn

from .read_datasets import read_datasets
from .Optimizer import Optimizer
from .Tracker import Tracker
from .system_info import nvlink_check
from .system_info import get_cuda_devices
from .Compressor import *
# from .DaSHCo import * 

import torch.distributed as dist
'''
DistDataModel is a class which deals with simulating data distributed models.
The class inherets from CommNet, which defines a network topology and communication
methods to be used. This child class deals with the following components.
- Defining a model on each MPI rank.
- Defines the train and test loaders.
- Seeds the models.
- Defines distributed training methods.
- Contains evaluators for results.

NOTES:
	- Need to enable setting the seed. Currently the parameter does nothing.
'''
class DistDataModel():

	def __init__(self,model="LeNet5",dataset="FashionMNIST",topology="ring",\
		optimizer=None,comm_set=['x'],batch_size=16,device='cpu',\
		nvlink=False,track=True,seed=1337,compressor=NoneCompressor(), variety="index", lr_decay="none", lr = 0.001, k=1):

		# Define our training duration and the communication parameters.
		self.epoch = 0
		self.epochs = 50
		self.comm_set = comm_set
		self.topology = topology
		self.seed = seed

		# Define our model, and optimizers.
		self.device = device
		self.optimizer_name = optimizer
		self.model_name = model
		self.model = models.__dict__[model]()
		self.lr_decay = lr_decay
		self.lr = lr
		self.k=k
		self.set_optimizer(optimizer,compressor,nvlink=nvlink)
		self.model = self.model.to(self.device)
		self.dataset = dataset
		self.variety = variety

		self.nprocs = self.optim.nprocs
		self.rank = self.optim.rank
		self.train_dataset, self.test_dataset = read_datasets(dataset)#, "../data")
		self.batch_size = batch_size
		self.loss_fcn = nn.CrossEntropyLoss()
		self.form_loaders(variety=self.variety)

		# Set up our tracker if it makes sense.
		self.track = track
		if track == True:
			self.tracker = Tracker(model=self.model,model_name=self.model_name,loss_function=self.loss_fcn,\
			test_loader=self.test_loader,train_loader=self.train_loader,device=self.device)

	'''
	Gets the samples for each rank from the data set and defines our data loaders.
	'''
	def form_loaders(self,variety="index"):

		if variety == "index":

			# First we have to define our set of indices to split the data and get our sampler.
			data_per_node = int(len(self.train_dataset)/self.nprocs)
			idx = [data_per_node*self.rank+i for i in range(data_per_node)]

		elif variety == "label":

			class_num = len(self.train_dataset.classes)
			splits_per_class = self.nprocs // class_num
			class_idx = self.rank % class_num
			split_idx = self.rank // class_num

			class_to_indices = {c: [] for c in range(class_num)}
			for i, (_, label) in enumerate(self.train_dataset):
				class_to_indices[label].append(i)

			min_class_size = min(len(v) for v in class_to_indices.values())

			all_indices = class_to_indices[class_idx]
			chunk_size  = len(all_indices) // splits_per_class
			start       = split_idx * chunk_size
			end         = len(all_indices) if split_idx == splits_per_class - 1 else (start + chunk_size)
			idx         = all_indices[start:end]

		else:
			print("ERROR IN FORM LOADER (RANK: {}). INVALID VARIETY ({}).".format(self.rank,variety))

		# Set our Sampler.
		sampler = SRS(idx)

		# Now we define our loaders based on this.
		self.train_loader = DL(self.train_dataset, batch_size=self.batch_size, sampler=sampler, drop_last=True)
		self.test_loader = DL(self.test_dataset, batch_size=1000, shuffle=False)
		self.train_loader_all = DL(self.train_dataset, batch_size=1000, shuffle=False) #remove that later


	'''
	Sets our optimizer based on an input string.
	'''
	def set_optimizer(self,optimizer_name,compressor,nvlink=False):

		if self.device == "cpu":
				device_list=[]

				self.optim = Optimizer(self.model, compressor,optim_name=optimizer_name,\
					comm_set=self.comm_set, device=self.device, topology=self.topology,\
					devices=device_list, nvlink=nvlink, lr_decay=self.lr_decay, lr=self.lr, k=self.k)
			
		else:

				# Get the names of our cuda enabled devices.
				devices = get_cuda_devices()

				# If our nvlink flag is live, check if our system has an nvlink using nvidia-smi.
				if nvlink:
					nvlink = nvlink_check()

				self.optim =Optimizer(self.model, compressor,optim_name=optimizer_name,\
				comm_set=self.comm_set, device=self.device, topology=self.topology,\
				devices=devices, nvlink=nvlink, lr_decay=self.lr_decay,lr=self.lr, k=self.k)

				rank = self.optim.rank
				self.device = self.optim.devices[rank % len(self.optim.devices)]

	'''
	Performs training for the defined number of epochs.
	'''
	def train(self,output_file="default",verbose=False):

		if self.rank == 0:
			if self.model_name=="nanoGPT":
				print("epoch\tRank\ttest_loss\ttrain_loss\tcons_error\ttest_time\ttrain_time")
			else:
				print("epoch\tRank\ttest_acc\ttest_loss\ttrain_acc\ttrain_loss\tcons_error\ttest_time\ttrain_time")

		if self.track:
			verbose = True

		best_val_loss = 1e9

		while self.epoch < self.epochs:

			# Increment our epoch and set the model to train.
			self.epoch += 1
			self.model.train()

			# If verbose, print our stats.
			if self.track:
				test_loss,test_acc,test_time = self.tracker.evaluate(loader="test")
				cons_error = self.tracker.compute_cons_error(self.comm_set,self.optim)
				train_loss,train_acc,train_time = self.tracker.evaluate(loader="train")
				if verbose:
					# print("EPOCH: {} -- RANK: {} -- TEST_LOSS: {} -- TEST_ACC: {} -- CONS_ERROR: {} -- TEST_TIME: {} -- TRAIN_TIME: {}".format(self.epoch-1,self.rank,test_loss,test_acc,cons_error,test_time,train_time))
					# print("EPOCH: {} -- RANK: {} -- TEST_LOSS: {} -- TEST_ACC: {} -- TEST_TIME: {} -- TRAIN_TIME: {}".format(self.epoch-1,self.rank,test_loss,test_acc,test_time,train_time))
					if self.model_name=="nanoGPT":
						print("{}\t{}\t{}\t{}\t{}\t{}\t{}".format(self.epoch-1,self.rank,test_loss,train_loss,cons_error,test_time,train_time))
					else:
						print("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}".format(self.epoch-1,self.rank,test_acc,test_loss,train_acc,train_loss,cons_error,test_time,train_time))

			# Perform training for our batches.
			for batch_index, (data,target) in enumerate(self.train_loader):

				# Do the standard training stuff.
				data,target = data.to(self.device), target.to(self.device)
				self.model.zero_grad()
				if self.model_name=="nanoGPT":
					output,loss = self.model(data,target)
				else:
					output = self.model(data)
					loss = self.loss_fcn(output, target)
				loss.backward()
				
				#print("UID {} >> loss: {}".format(self.optim.rank,loss) )

				# Perform step.
				# self.optim.step()
				if self.optimizer_name == "DoCoM":
					self.optim.step(self.loss_fcn,data,target)
				else:
					self.optim.step()

			# Set up a barrier at the end of the epoch.
			self.optim.COMM.Barrier()

		# If we are tracking, return our out_dict at the end of training.
		if self.track:
			# self.tracker.save_history(output_file)
			self.tracker.save_history(output_file,self.optim.COMM)
			return(self.tracker.history)
