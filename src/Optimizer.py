import torch
from .CommNet import CommNet
# from .CommNet_new import CommNet
from .Compressor import *
from copy import deepcopy
'''
The Optimizer class implements various different methods for optimization.
'''
class Optimizer(CommNet):

	def __init__(self,model,compressor,optim_name="DistributedAdam",comm_set=['x'],device="cpu",topology="ring",devices=[],nvlink=False,lr_decay="none",lr=0.001, k=1,model_name = ""):
		if optim_name == "DistributedAdam" or optim_name == "DistributedAdaGrad" or optim_name == "CDProxSGT" or optim_name=="SQuARM-SGD":
			# we are going to do compression in the step and then communicate it
			super().__init__(topology=topology,comms=device,devices=devices,nvlink=nvlink,compressor=NoneCompressor())
		else:
			# we will do compression while we communicate
			super().__init__(topology=topology,comms=device,devices=devices,nvlink=nvlink,compressor=compressor)

		self.comm_set = comm_set
		self.optim_name = optim_name
		self.model= model
		self.opt_compressor = compressor
		
		# for cosine decay as in NanoGPT
		self.lr=lr
		self.lr_decay = lr_decay
		self.steps = 0
		self.warmup_iters = 2000 #100 for all but hb and cd, 2000
		self.min_lr = 6e-5 #1e-4,6e-5
		self.lr_decay_iters = 5000 #2000, 5000
		
		self.k = k
		self.model_name = model_name

		if device == "cpu":
			self.device = device
		else:
			self.device = devices[self.rank % len(devices)]
		self.set_data()


	def get_names(self,field):
		return self.data[field].keys()

	'''
	Sets up the data variables in our CommNet parent, based on the comm_set, optim_name, and model.
	'''
	def set_data(self):

		#initial
		'''
		self.recv_data["reduced"] = {}
		self.recv_data["rec_field"] = {}
		self.recv_data["cpu_holder"] = {}
		for field in params:
			self.data[field] = {}


		for i in range(1,len(params)):
			for name,param in self.model.named_parameters():
				self.data[params[0]][name] = param.data.detach().clone().to(self.device)
				self.data[p][name] = torch.zeros_like(self.data[params[0]][name]).to(self.device)

		for name,param in self.model.named_parameters():
			self.recv_data["reduced"][name] = torch.zeros_like(self.data[params[0]][name]).to(self.device)
			self.recv_data["rec_field"][name] = torch.zeros_like(self.data[params[0]][name]).to(self.device)
			if self.device != "cpu" and not self.nvlink:
				self.recv_data["cpu_holder"][name] = torch.zeros_like(self.data[params[0]][name]).to("cpu")

		return 
'''
		# Check what optimization method we are using.
		if self.optim_name == "DistributedAdam":

			# Set up Adam specific scalars.
			#self.lr = 0.001
			self.beta1 = 0.9
			self.beta2 = 0.999
			self.eps = 1e-8
			self.epoch = 0
			self.gamma = 0.5

			# Set up the initial dictionaries for our data.
			param_fields = ['x','g','m','v','u','v_prev','x_bar','x_bar_prev','g_bar','g_bar_prev','delta_x','delta_g','u_prev']
			self.recv_data["reduced"] = {}
			self.recv_data["rec_field"] = {}
			self.recv_data["cpu_holder"] = {}
			for field in param_fields:
				self.data[field] = {}

			# Set up each variable torch tensor.
			for name,param in self.model.named_parameters():
				self.data['x'][name] = param.data.detach().clone().to(self.device)
				self.data['g'][name] = torch.zeros_like(self.data['x'][name]).to(self.device)
				self.data['m'][name] = torch.zeros_like(self.data['x'][name]).to(self.device)
				self.data['v'][name] = torch.zeros_like(self.data['x'][name]).to(self.device)
				self.data['u'][name] = torch.zeros_like(self.data['x'][name]).to(self.device)
				self.data['u_prev'][name] = torch.zeros_like(self.data['x'][name]).to(self.device)
				self.data['v_prev'][name] = torch.zeros_like(self.data['x'][name]).to(self.device)
				self.data['x_bar'][name] = torch.zeros_like(self.data['x'][name]).to(self.device)
				self.data['x_bar_prev'][name] = torch.zeros_like(self.data['x'][name]).to(self.device)
				self.data['g_bar'][name] = torch.zeros_like(self.data['x'][name]).to(self.device)
				self.data['g_bar_prev'][name] = torch.zeros_like(self.data['x'][name]).to(self.device)
				self.data['delta_x'][name] = torch.zeros_like(self.data['x'][name]).to(self.device)
				self.data['delta_g'][name] = torch.zeros_like(self.data['x'][name]).to(self.device)
				self.recv_data["reduced"][name] = torch.zeros_like(self.data['x'][name]).to(self.device)
				self.recv_data["rec_field"][name] = torch.zeros_like(self.data['x'][name]).to(self.device)
				if self.device != "cpu" and not self.nvlink:
					self.recv_data["cpu_holder"][name] = torch.zeros_like(self.data['x'][name]).to("cpu")
					
		elif self.optim_name == "DistributedAMSGrad":

			# Set up Adam specific scalars.
			#self.lr = 0.001
			self.beta1 = 0.9
			self.beta2 = 0.999
			self.eps = 1e-8
			self.epoch = 0

			# Set up the initial dictionaries for our data.
			param_fields = ['x','g','m','v','u','v_prev','v_max']
			self.recv_data["reduced"] = {}
			self.recv_data["rec_field"] = {}
			self.recv_data["cpu_holder"] = {}
			for field in param_fields:
				self.data[field] = {}

			# Set up each variable torch tensor.
			for name,param in self.model.named_parameters():
				self.data['x'][name] = param.data.detach().clone().to(self.device)
				self.data['g'][name] = torch.zeros_like(self.data['x'][name]).to(self.device)
				self.data['m'][name] = torch.zeros_like(self.data['x'][name]).to(self.device)
				self.data['v'][name] = torch.zeros_like(self.data['x'][name]).to(self.device)
				self.data['u'][name] = torch.zeros_like(self.data['x'][name]).to(self.device)
				self.data['v_prev'][name] = torch.zeros_like(self.data['x'][name]).to(self.device)
				self.data['v_max'][name] = torch.zeros_like(self.data['x'][name]).to(self.device)
				self.recv_data["reduced"][name] = torch.zeros_like(self.data['x'][name]).to(self.device)
				self.recv_data["rec_field"][name] = torch.zeros_like(self.data['x'][name]).to(self.device)
				if self.device != "cpu" and not self.nvlink:
					self.recv_data["cpu_holder"][name] = torch.zeros_like(self.data['x'][name]).to("cpu")

		elif self.optim_name == "DistributedAdaGrad":

			# Set up AdaGrad specific scalars.
			#self.lr = 0.001
			self.beta1 = 0.9
			self.beta2 = 0.999
			self.eps = 1e-8
			self.epoch = 0
			self.gamma = 0.5

			# Set up the initial dictionaries for our data.
			param_fields = ['x','g','m','v','u','v_prev','g_sum','g_bar','g_bar_prev','x_bar','x_bar_prev','delta_x','delta_g','u_prev']
			self.recv_data["reduced"] = {}
			self.recv_data["rec_field"] = {}
			self.recv_data["cpu_holder"] = {}
			for field in param_fields:
				self.data[field] = {}

			# Set up each variable torch tensor.
			for name,param in self.model.named_parameters():
				self.data['x'][name] = param.data.detach().clone().to(self.device)
				self.data['g'][name] = torch.zeros_like(self.data['x'][name]).to(self.device)
				self.data['g_sum'][name] = torch.zeros_like(self.data['x'][name]).to(self.device)
				self.data['m'][name] = torch.zeros_like(self.data['x'][name]).to(self.device)
				self.data['v'][name] = torch.zeros_like(self.data['x'][name]).to(self.device)
				self.data['v_prev'][name] = torch.zeros_like(self.data['x'][name]).to(self.device)
				self.data['u'][name] = torch.zeros_like(self.data['x'][name]).to(self.device)
				self.data['u_prev'][name] = torch.zeros_like(self.data['x'][name]).to(self.device)
				self.data['g_bar'][name] = torch.zeros_like(self.data['x'][name]).to(self.device)
				self.data['g_bar_prev'][name] = torch.zeros_like(self.data['x'][name]).to(self.device)
				self.data['x_bar_prev'][name] = torch.zeros_like(self.data['x'][name]).to(self.device)
				self.data['x_bar'][name] = torch.zeros_like(self.data['x'][name]).to(self.device)

				self.data['delta_x'][name] = torch.zeros_like(self.data['x'][name]).to(self.device)
				self.data['delta_g'][name] = torch.zeros_like(self.data['x'][name]).to(self.device)
				self.recv_data["reduced"][name] = torch.zeros_like(self.data['x'][name]).to(self.device)
				self.recv_data["rec_field"][name] = torch.zeros_like(self.data['x'][name]).to(self.device)
				if self.device != "cpu" and not self.nvlink:
					self.recv_data["cpu_holder"][name] = torch.zeros_like(self.data['x'][name]).to("cpu")
		
		elif self.optim_name == "AdamW":

			# Set up Adam specific scalars.
			#self.lr = 0.001
			self.beta1 = 0.9
			self.beta2 = 0.999
			self.eps = 1e-8
			self.labda = 0.01
			self.epoch = 0

			# Set up the initial dictionaries for our data.
			param_fields = ['x','g','m','v','m_hat','v_hat']
			self.recv_data["reduced"] = {}
			self.recv_data["rec_field"] = {}
			self.recv_data["cpu_holder"] = {}
			for field in param_fields:
				self.data[field] = {}

			# Set up each variable torch tensor.
			for name,param in self.model.named_parameters():
				self.data['x'][name] = param.data.detach().clone().to(self.device)
				self.data['g'][name] = torch.zeros_like(self.data['x'][name]).to(self.device)
				self.data['m'][name] = torch.zeros_like(self.data['x'][name]).to(self.device)
				self.data['v'][name] = torch.zeros_like(self.data['x'][name]).to(self.device)
				self.data['m_hat'][name] = torch.zeros_like(self.data['x'][name]).to(self.device)
				self.data['v_hat'][name] = torch.zeros_like(self.data['x'][name]).to(self.device)
				self.recv_data["reduced"][name] = torch.zeros_like(self.data['x'][name]).to(self.device)
				self.recv_data["rec_field"][name] = torch.zeros_like(self.data['x'][name]).to(self.device)
				if self.device != "cpu" and not self.nvlink:
					self.recv_data["cpu_holder"][name] = torch.zeros_like(self.data['x'][name]).to("cpu")
		# If we have nothing, return nothing.

		elif self.optim_name == "CDProxSGT":
			# Set up Adam specific scalars.
			#self.lr = 0.02
			self.beta1 = 0.9
			self.beta2 = 0.999
			self.eps = 1e-8
			self.mu = 1e-4
			self.epoch = 0
			self.gamma_g = 0.5
			self.gamma_x = 0.5

			# Set up the initial dictionaries for our data.
			param_fields = ['x','g','g_tilde','g_tilde_prev','m','x_bar','x_bar_prev','g_bar','g_bar_prev','delta_x','delta_g']
			self.recv_data["reduced"] = {}
			self.recv_data["rec_field"] = {}
			self.recv_data["cpu_holder"] = {}
			for field in param_fields:
				self.data[field] = {}

			# Set up each variable torch tensor.
			for name,param in self.model.named_parameters():
				self.data['x'][name] = param.data.detach().clone().to(self.device)
				self.data['g'][name] = torch.zeros_like(self.data['x'][name]).to(self.device)
				self.data['g_tilde'][name] = torch.zeros_like(self.data['x'][name]).to(self.device)
				self.data['g_tilde_prev'][name] = torch.zeros_like(self.data['x'][name]).to(self.device)
				self.data['x_bar'][name] = torch.zeros_like(self.data['x'][name]).to(self.device)
				self.data['x_bar_prev'][name] = torch.zeros_like(self.data['x'][name]).to(self.device)
				self.data['g_bar'][name] = torch.zeros_like(self.data['x'][name]).to(self.device)
				self.data['g_bar_prev'][name] = torch.zeros_like(self.data['x'][name]).to(self.device)
				self.data['delta_x'][name] = torch.zeros_like(self.data['x'][name]).to(self.device)
				self.data['delta_g'][name] = torch.zeros_like(self.data['x'][name]).to(self.device)
				
				self.data['m'][name] = torch.zeros_like(self.data['x'][name]).to(self.device)
				self.recv_data["reduced"][name] = torch.zeros_like(self.data['x'][name]).to(self.device)
				self.recv_data["rec_field"][name] = torch.zeros_like(self.data['x'][name]).to(self.device)
				if self.device != "cpu" and not self.nvlink:
					self.recv_data["cpu_holder"][name] = torch.zeros_like(self.data['x'][name]).to("cpu")

		elif self.optim_name == "SQuARM-SGD":
			# Set up Adam specific scalars.
			#self.lr = 0.02
			self.beta1 = 0.9
			self.eps = 1e-8
			self.mu = 1e-4
			self.epoch = 0
			self.gamma_g = 0.5
			self.gamma_x = 0.5

			# Set up the initial dictionaries for our data.
			param_fields = ['x','g','g_tilde','m','x_bar','x_bar_prev','delta_x','m_prev']
			self.recv_data["reduced"] = {}
			self.recv_data["rec_field"] = {}
			self.recv_data["cpu_holder"] = {}
			for field in param_fields:
				self.data[field] = {}

			# Set up each variable torch tensor.
			for name,param in self.model.named_parameters():
				self.data['x'][name] = param.data.detach().clone().to(self.device)
				self.data['g'][name] = torch.zeros_like(self.data['x'][name]).to(self.device)
				self.data['g_tilde'][name] = torch.zeros_like(self.data['x'][name]).to(self.device)
				self.data['x_bar'][name] = torch.zeros_like(self.data['x'][name]).to(self.device)
				self.data['x_bar_prev'][name] = torch.zeros_like(self.data['x'][name]).to(self.device)
				self.data['delta_x'][name] = torch.zeros_like(self.data['x'][name]).to(self.device)
				self.data['m_prev'][name] = torch.zeros_like(self.data['x'][name]).to(self.device)
				self.data['m'][name] = torch.zeros_like(self.data['x'][name]).to(self.device)

				self.recv_data["reduced"][name] = torch.zeros_like(self.data['x'][name]).to(self.device)
				self.recv_data["rec_field"][name] = torch.zeros_like(self.data['x'][name]).to(self.device)
				if self.device != "cpu" and not self.nvlink:
					self.recv_data["cpu_holder"][name] = torch.zeros_like(self.data['x'][name]).to("cpu")
		else:

			return


	def get_lr(self, it):
		# 1) linear warmup for warmup_iters steps
		# if it%500 == 0 and it<=2000:
		# 	print("Step increased")
		if it < self.warmup_iters:
			return self.lr * it / self.warmup_iters
		# 2) if it > lr_decay_iters, return min learning rate
		if it > self.lr_decay_iters:
			return self.min_lr
		# 3) in between, use cosine decay down to min learning rate
		decay_ratio = (it - self.warmup_iters) / \
				(self.lr_decay_iters - self.warmup_iters)
		assert 0 <= decay_ratio <= 1
		coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
		return self.min_lr + coeff * (self.lr - self.min_lr)
	
	
	'''
	The following is a general step function. It takes in fields to be communicated at each step and
	performs the optimization according to those.
	NOTES:
		- Defaults to ONLY communicating the 'x' vector
	'''
	def step(self, loss_fcn=None,data=None,target=None):

		'''
		No implenmentation in base class....
		'''

		# If we are using Adam, proceed accordingly.
		if self.optim_name == "DistributedAdam":

			for name,param in self.model.named_parameters():
				with torch.no_grad():
					learning_rate = self.lr
					if self.lr_decay == "cosine":
						learning_rate = self.get_lr(self.steps)
					
					self.data['g'][name] = param.grad.data.detach().clone()
					if self.opt_compressor.get_name() != "none":
						compress_in = self.opt_compressor.compress(self.data['g'][name]-self.data['g_bar'][name])
						self.data['g_bar'][name] += self.opt_compressor.decompress(compress_in[1:])
						self.data['g_bar_prev'][name] = self.data['g_bar'][name].clone()

						compress_final = self.opt_compressor.compress(self.data['g'][name]-self.data['g_bar'][name])
						self.data['delta_g'][name] = self.opt_compressor.decompress(compress_final[1:])

						self.neighbor_reduce_cond(field='g_bar',name=name,comm_set=self.comm_set) 
						self.data['g'][name] = self.data['g'][name]+self.gamma*(self.data['g_bar'][name]-self.data['g_bar_prev'][name])
					else:
						self.neighbor_reduce_cond(field='g', name=name, comm_set=self.comm_set)
					
					self.data['m'][name] = (self.beta1 * self.data['m'][name]) +\
					( (1 - self.beta1) * self.data['g'][name])
					self.neighbor_reduce_cond(field='m', name=name, comm_set=self.comm_set)

					self.data['v_prev'][name] = self.data['v'][name].clone()
					self.data['v'][name] = (self.beta2 * self.data['v'][name]) +\
					( (1 - self.beta2) * (self.data['g'][name] * self.data['g'][name]) )
					
					self.neighbor_reduce_cond(field='v', name=name, comm_set=self.comm_set)

					self.data['u_prev'][name] = self.data['u'][name].clone()

					self.data['u'][name] = self.data['u'][name] - self.data['v_prev'][name] +\
					self.data['v'][name]
					self.neighbor_reduce_cond(field='u', name=name, comm_set=self.comm_set)

					# self.data['x'][name] -= learning_rate*(self.data['m'][name]/(torch.sqrt(self.data['u'][name]+self.eps)))
					self.data['x'][name] -= learning_rate*(self.data['m'][name]/(torch.sqrt(self.data['u_prev'][name]+self.eps)))

					if self.opt_compressor.get_name() != "none":
						compress_in = self.opt_compressor.compress(self.data['x'][name]-self.data['x_bar'][name])
						self.data['x_bar'][name] += self.opt_compressor.decompress(compress_in[1:])
						self.data['x_bar_prev'][name] = self.data['x_bar'][name].clone()

						#here we try doing sparse doing this little tweak:
						compress_final = self.opt_compressor.compress(self.data['x'][name]-self.data['x_bar'][name])
						self.data['delta_x'][name] = self.opt_compressor.decompress(compress_final[1:])

						self.neighbor_reduce_cond(field='x_bar',name=name,comm_set=self.comm_set) 
						self.data['x'][name] = self.data['x'][name]+self.gamma*(self.data['x_bar'][name]-self.data['x_bar_prev'][name])
					else:
						self.neighbor_reduce_cond(field='x', name=name, comm_set=self.comm_set)


					param.copy_(self.data['x'][name])
					param.grad.copy_(self.data['g'][name])
					self.steps += 1
		
		elif self.optim_name == "DistributedAMSGrad":

			for name,param in self.model.named_parameters():
				with torch.no_grad():
					learning_rate = self.lr
					if self.lr_decay == "cosine":
						learning_rate = self.get_lr(self.steps)


					g = param.grad.detach()
					if not torch.isfinite(g).all():
						g = torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)

					self.data['g'][name] = g
					self.neighbor_reduce_cond(field='g', name=name, comm_set=self.comm_set)

					self.data['m'][name] = (self.beta1 * self.data['m'][name] + (1 - self.beta1) * self.data['g'][name])
					self.neighbor_reduce_cond(field='m', name=name, comm_set=self.comm_set)

					self.data['v_prev'][name] = self.data['v'][name].clone()
					self.data['v'][name] = (self.beta2 * self.data['v'][name] +(1 - self.beta2) * (self.data['g'][name] * self.data['g'][name]))
					self.neighbor_reduce_cond(field='v', name=name, comm_set=self.comm_set)

					torch.maximum(self.data['v_max'][name],self.data['v'][name],out=self.data['v_max'][name])

					self.data['u'][name].add_(self.data['v_max'][name]).sub_(self.data['v_prev'][name])
					self.data['u'][name].clamp_(min=0)
					denom = torch.sqrt(self.data['u'][name] + self.eps)

					self.data['x'][name].addcdiv_(self.data['m'][name], denom, value=-learning_rate)
					self.neighbor_reduce_cond(field='x', name=name, comm_set=self.comm_set)

					param.copy_(self.data['x'][name])
					param.grad.copy_(self.data['g'][name])
					self.steps += 1

		elif self.optim_name == "DistributedAdaGrad":
			
			self.epoch += 1
			for name,param in self.model.named_parameters():
				with torch.no_grad():
					learning_rate = self.lr
					if self.lr_decay == "cosine":
						learning_rate = self.get_lr(self.steps)
					
					self.data['g'][name] = param.grad.data.detach().clone()
					# self.neighbor_reduce_cond(field='g', name=name, comm_set=self.comm_set)
					if self.opt_compressor.get_name() != "none":
						compress_in = self.opt_compressor.compress(self.data['g'][name]-self.data['g_bar'][name])
						self.data['g_bar'][name] += self.opt_compressor.decompress(compress_in[1:])
						self.data['g_bar_prev'][name] = self.data['g_bar'][name].clone()

						compress_final = self.opt_compressor.compress(self.data['g'][name]-self.data['g_bar'][name])
						self.data['delta_g'][name] = self.opt_compressor.decompress(compress_final[1:])

						#\sum WjiQ(xj-xj_)
						self.neighbor_reduce_cond(field='g_bar',name=name,comm_set=self.comm_set)
						self.data['g'][name] = self.data['g'][name]+0.5*(self.data['g_bar'][name]-self.data['g_bar_prev'][name])
					else:
						self.neighbor_reduce_cond(field='g', name=name, comm_set=self.comm_set)

					self.data['m'][name] = (self.beta1 * self.data['m'][name]) +\
					( (1 - self.beta1) * self.data['g'][name])
					self.neighbor_reduce_cond(field='m', name=name, comm_set=self.comm_set)

					self.data['g_sum'][name] = self.data['g_sum'][name] + \
							(self.data['g'][name] * self.data['g'][name])
					self.data['v_prev'][name] = self.data['v'][name].clone()
					self.data['v'][name] = self.data['g_sum'][name] / self.epoch
					self.neighbor_reduce_cond(field='v', name=name, comm_set=self.comm_set)

					self.data['u_prev'][name] = self.data['u'][name].clone()
					self.data['u'][name] = self.data['u'][name] - self.data['v_prev'][name] +\
					self.data['v'][name]
					self.neighbor_reduce_cond(field='u', name=name, comm_set=self.comm_set)

					# self.data['x'][name] -= learning_rate*(self.data['m'][name]/(torch.sqrt(self.data['u'][name]+self.eps)))
					self.data['x'][name] -= learning_rate*(self.data['m'][name]/(torch.sqrt(self.data['u_prev'][name]+self.eps)))
					if self.opt_compressor.get_name() != "none":

						compress_in = self.opt_compressor.compress(self.data['x'][name]-self.data['x_bar'][name])
						self.data['x_bar'][name] += self.opt_compressor.decompress(compress_in[1:])
						self.data['x_bar_prev'][name] = self.data['x_bar'][name].clone()

						#here we try doing sparse doing this little tweak:
						compress_final = self.opt_compressor.compress(self.data['x'][name]-self.data['x_bar'][name])
						self.data['delta_x'][name] = self.opt_compressor.decompress(compress_final[1:])

						self.neighbor_reduce_cond(field='x_bar',name=name,comm_set=self.comm_set)
						self.data['x'][name] = self.data['x'][name]+self.gamma*(self.data['x_bar'][name]-self.data['x_bar_prev'][name])

					else:
						self.neighbor_reduce_cond(field='x',name=name,comm_set=self.comm_set)

					param.copy_(self.data['x'][name])
					param.grad.copy_(self.data['g'][name])
					self.steps += 1

		elif self.optim_name == "AdamW":
			
			self.epoch += 1
			for name,param in self.model.named_parameters():
				with torch.no_grad():
					learning_rate = self.lr
					if self.lr_decay == "cosine":
						learning_rate = self.get_lr(self.steps)
					# https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html
					
					self.data['g'][name] = param.grad.data.detach().clone()
					self.neighbor_reduce_cond(field='g', name=name, comm_set=self.comm_set)
					
					self.data['x'][name] -= learning_rate*self.labda*self.data['x'][name]
						
					self.data['m'][name] = (self.beta1 * self.data['m'][name]) +\
							((1 - self.beta1) * self.data['g'][name])
					
					self.data['v'][name] = (self.beta2 * self.data['v'][name]) +\
							((1 - self.beta2) * (self.data['g'][name] * self.data['g'][name]))
					
					self.data['m_hat'][name] = self.data['m'][name] / (1 - self.beta1)
					self.data['v_hat'][name] = self.data['v'][name] / (1 - self.beta2)
					
					self.data['x'][name] -= learning_rate*(self.data['m_hat'][name] /\
							(torch.sqrt(self.data['v_hat'][name])+self.eps))
					self.neighbor_reduce_cond(field='x', name=name, comm_set=self.comm_set)
					
					param.copy_(self.data['x'][name])
					param.grad.copy_(self.data['g'][name])
					self.steps += 1

		elif self.optim_name == "CDProxSGT":
			
			self.epoch += 1
			for name,param in self.model.named_parameters():
				with torch.no_grad():
					learning_rate = self.lr
					if self.lr_decay == "cosine":
						learning_rate = self.get_lr(self.steps)
					
					self.data['g_tilde'][name] = param.grad.data.detach().clone()
					self.data['g'][name] = self.data['g'][name] - \
							self.data['g_tilde_prev'][name] + self.data['g_tilde'][name]
					self.data['g_tilde_prev'][name] = self.data['g_tilde'][name].clone()

					if self.opt_compressor.get_name() !="none":

						compress_in = self.opt_compressor.compress(self.data['g'][name]-self.data['g_bar'][name])
						self.data['g_bar'][name] += self.opt_compressor.decompress(compress_in[1:])
						self.data['g_bar_prev'][name] = self.data['g_bar'][name].clone()

						#here we try doing sparse doing this little tweak:
						compress_final = self.opt_compressor.compress(self.data['g'][name]-self.data['g_bar'][name])
						self.data['delta_g'][name] = self.opt_compressor.decompress(compress_final[1:])

						#\sum WjiQ(xj-xj_)
						self.neighbor_reduce_cond(field='g_bar',name=name,comm_set=self.comm_set)

						self.data['g'][name] = self.data['g'][name]+self.gamma_g*(self.data['g_bar'][name]-self.data['g_bar_prev'][name])

					else:
						self.neighbor_reduce_cond(field='g',name=name,comm_set=self.comm_set)

					self.data['x'][name] = torch.max(torch.abs(self.data['x'][name] - learning_rate*(self.data['g'][name])) \
									  ,torch.zeros_like(self.data['x'][name])) * torch.sign(self.data['x'][name] - learning_rate*(self.data['g'][name]))
				
					if self.opt_compressor.get_name() != "none":

						#Q(x-x_)
						compress_in = self.opt_compressor.compress(self.data['x'][name]-self.data['x_bar'][name])

						#x_ = x_{t} + Q(x-x_)
	
						self.data['x_bar'][name] += self.opt_compressor.decompress(compress_in[1:])
						self.data['x_bar_prev'][name] = self.data['x_bar'][name].clone()

						#here we try doing sparse doing this little tweak:
						compress_final = self.opt_compressor.compress(self.data['x'][name]-self.data['x_bar'][name])
						self.data['delta_x'][name] = self.opt_compressor.decompress(compress_final[1:])

						#\sum WjiQ(xj-xj_)
						self.neighbor_reduce_cond(field='x_bar',name=name,comm_set=self.comm_set)
						self.data['x'][name] = (self.data['x'][name]+self.gamma_x*(self.data['x_bar'][name]-self.data['x_bar_prev'][name]))
				
					else:
						self.neighbor_reduce_cond(field='x',name=name,comm_set=self.comm_set)
					
					param.copy_(self.data['x'][name])
					param.grad.copy_(self.data['g'][name])
					self.steps += 1
		
		elif self.optim_name == "SQuARM-SGD":
			
			self.epoch += 1
			
			for name,param in self.model.named_parameters():
				with torch.no_grad():
					learning_rate = self.lr
					if self.lr_decay == "cosine":
						learning_rate = self.get_lr(self.steps)
					
					self.data['g_tilde'][name] = param.grad.data.detach().clone()

					self.data['g'][name] = self.data['g_tilde'][name]

					self.data['m'][name] = self.beta1*self.data['m'][name]+self.data['g'][name]
					self.data['m_prev'][name] = self.data['m'][name].clone()

					
					self.data['x'][name] = (self.data['x'][name] - learning_rate*(self.beta1*self.data['m'][name]+self.data['g'][name]))
				
					if self.opt_compressor.get_name() != "none":
						
						#Q(x-x_)
						compress_in = self.opt_compressor.compress(self.data['x'][name]-self.data['x_bar'][name])

						#x_ = x_{t} + Q(x-x_)
						self.data['x_bar'][name] += self.opt_compressor.decompress(compress_in[1:])
						self.data['x_bar_prev'][name] = self.data['x_bar'][name].clone()

						#here we try doing sparse doing this little tweak:
						compress_final = self.opt_compressor.compress(self.data['x'][name]-self.data['x_bar'][name])
						self.data['delta_x'][name] = self.opt_compressor.decompress(compress_final[1:])
						

						#\sum WjiQ(xj-xj_)
						self.neighbor_reduce_cond(field='x_bar',name=name,comm_set=self.comm_set) 
						self.data['x'][name] = (self.data['x'][name]+self.gamma_x*(self.data['x_bar'][name]-self.data['x_bar_prev'][name]))
				
					else:
						self.neighbor_reduce_cond(field='x',name=name,comm_set=self.comm_set)

					param.copy_(self.data['x'][name])
					param.grad.copy_(self.data['g'][name])
					self.steps += 1


					
	'''
    Performs a conditional neighbor_reduce() from CommNet for the given variable.
    If the given variable is not in the "comm_set" list, then nothing happens. 
    NOTE:
    	- This only works on one subfield / "name" at a time. This allows for it to be done
    	  within each loop of "step()".
    '''
	def neighbor_reduce_cond(self,field,name,comm_set=[]):
		if self.nprocs == 1:
			return
		
		# If we are communicating the current field, perform a neighbor_reduce and set the associated values.
		if field in comm_set and self.epoch%self.k == 0:
			super().neighbor_reduce(field,name,unique=False)

			self.data[field][name] = self.recv_data["reduced"][name].clone()
		
