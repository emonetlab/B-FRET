import autograd.numpy as np
from utils import norm_pdf, lognorm_pdf, standard_uniform_pdf
from autograd.scipy.special import gamma
from scipy.stats import t

class model():
	
	class f_D():

		def __init__(self):
		
			self.p_name = ['tau_D1', 'tau_D2', 'delta_D']
			self.p_init = [100, 100, 0.5]
			self.p_lo_bnd = [0, 0, 0]
			self.p_hi_bnd = [np.inf, np.inf, 1]
		
		def func(self, t, params):
	
			return params[2]*np.exp(-t/params[0]) \
				   + (1 - params[2])*np.exp(-t/params[1])
	
	class f_A():

		def __init__(self):
		
			self.p_name = ['tau_A1', 'tau_A2', 'delta_A']
			self.p_init = [100, 100, 0.5]
			self.p_lo_bnd = [0, 0, 0]
			self.p_hi_bnd = [np.inf, np.inf, 1]
			
		def func(self, t, params):
			
			return params[2]*np.exp(-t/params[0]) \
				   + (1 - params[2])*np.exp(-t/params[1])
		
	class prior_dict(object):
		"""
		Define functional form for prior distributions for D_T, A_T, but these
		still have a hyperparameter (location) to be fixed.
		"""
	
		def func(self, anl_params):
		
			pdfs_dict = dict()
			
			# Note that the ordering here is irrelevant
			# DT, AT
			pdfs_dict['DT'] = lognorm_pdf(sigma=np.log(2)).pdf
			pdfs_dict['AT'] = lognorm_pdf(sigma=np.log(1.1)).pdf
		
			# f_D params
			pdfs_dict['tau_D1'] =  lognorm_pdf(sigma=np.log(2)).pdf
			pdfs_dict['tau_D2'] = lognorm_pdf(sigma=np.log(2)).pdf
			pdfs_dict['delta_D'] = standard_uniform_pdf().pdf
			
			# f_A params
			pdfs_dict['tau_A1'] =  lognorm_pdf(sigma=np.log(1.1)).pdf
			pdfs_dict['tau_A2'] =  lognorm_pdf(sigma=np.log(1.1)).pdf
			pdfs_dict['delta_A'] =  standard_uniform_pdf().pdf
			
			# prior on process noise
			FC = anl_params.sigma_chi_prior_FC
			mode = anl_params.sigma_chi_prior_mode
			lg = lognorm_pdf(sigma=np.log(FC), fixed_mode=mode)
			pdfs_dict['sigma_chi'] = lg.pdf
			
			return pdfs_dict