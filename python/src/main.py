import autograd.numpy as np
import sys
from param_dist_funcs import param_est
from state_dist_funcs import state_est
from importlib import reload 
		

class analyze_FRET_data():
	"""
	Runs all esimation routines for a batch of data.
	"""
	
	def __init__(self, data, exp_dir):
	
		self.data_dict = data
		self.exp_dir = exp_dir
		if len(self.data_dict['IDD'].shape) == 1:
			self.num_datasets = 1
			self.num_IDD = len(self.data_dict['IDD'])
			self.num_IDA = len(self.data_dict['IDA'])
			self.num_IAA = len(self.data_dict['IAA'])
		else:
			self.num_datasets = self.data_dict['IDD'].shape[0]
			self.num_IDD = self.data_dict['IDD'].shape[1]
			self.num_IDA = self.data_dict['IDA'].shape[1]
			self.num_IAA = self.data_dict['IAA'].shape[1]
			
		# Import user-defined functions and analysis params
		# This is a bit hacky, but need to append to front, so the new
		# module location is checked first, and then reload modules.
		# This allows one to run a batch of estimations from 1 wrapper.
		sys.path = [self.exp_dir] + sys.path
		import define_model_functions
		import define_analysis_params
		reload(define_model_functions)
		reload(define_analysis_params)
		
		self.m = define_model_functions.model()
		self.anl_params = define_analysis_params.anl_params()
		self.m.f_D = self.m.f_D()
		self.m.f_A = self.m.f_A()
		self.m.prior_dict = self.m.prior_dict()
		
		if self.anl_params.process_noise == 'Non-Gaussian':
			self.m.Q_tilde = self.m.Q_tilde()
		
		# Order: DT, AT, [f_D params], [f_A params], [process noise params], G
		self.param_names = ['DT', 'AT']
		self.param_names += self.m.f_D.p_name
		self.param_names += self.m.f_A.p_name
		if self.anl_params.process_noise == 'Gaussian':
			self.param_names += ['sigma_chi']
		elif self.anl_params.process_noise == 'Non-Gaussian':
			self.param_names += self.m.Q_tilde.p_name
		
		# Convenience variables to keep track of which variables is which
		self.num_params = len(self.param_names)
		self.num_params_f_D = len(self.m.f_D.p_name)
		self.num_params_f_A = len(self.m.f_A.p_name)
		if self.anl_params.process_noise == 'Non-Gaussian':
			self.num_params_Q_tilde = len(self.m.Q_tilde.p_name)
		
		
	def run(self):
		
		for i in range(self.num_datasets): 
			self.res_dir = self.exp_dir + '/data%d' % i
			if self.num_datasets > 1:
				self.IDD = self.data_dict['IDD'][i]
				self.IAA = self.data_dict['IAA'][i]
				self.IDA = self.data_dict['IDA'][i]
				self.IDD_noise_sd = self.data_dict['IDD_noise_sd'][i]
				self.IAA_noise_sd = self.data_dict['IAA_noise_sd'][i]
				self.IDA_noise_sd = self.data_dict['IDA_noise_sd'][i]
				if 'E' in self.data_dict.keys():
					self.E_true = self.data_dict['E'][i]
			else:
				self.IDD = self.data_dict['IDD']
				self.IAA = self.data_dict['IAA']
				self.IDA = self.data_dict['IDA']
				self.IDD_noise_sd = self.data_dict['IDD_noise_sd']
				self.IAA_noise_sd = self.data_dict['IAA_noise_sd']
				self.IDA_noise_sd = self.data_dict['IDA_noise_sd']
				if 'E' in self.data_dict.keys():
					self.E_true = self.data_dict['E']
			self.tDD = self.data_dict['tDD']
			self.tAA = self.data_dict['tAA']
			
			a = param_est(self)
			a.gen_prior_dist()
			a.sample_from_post_dist()
			
			b = state_est(self, a)
			b.sample_from_post_dist()
			