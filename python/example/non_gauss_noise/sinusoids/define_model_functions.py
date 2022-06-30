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
		
	class Q_tilde():
		
		def __init__(self):
		
			self.p_name = ['sigma_chi', 'nu']

		def _f(self, x):
			
			val = 1/self.sd*gamma(self.nu/2 + 0.5)/np.sqrt(np.pi*self.nu)\
				  /gamma(self.nu/2)*(1 + (x/self.sd)**2/self.nu)\
				  **(-self.nu/2 - 0.5)
			
			return val

		def sample(self, params, size=1):
			
			sd, nu = params
			
			return t.rvs(df=nu, scale=sd, size=size)

		def pdf(self, params):
			
			self.sd, self.nu = params
			
			return self._f
		
		
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
			
			# This iterates through all hyperparameters of Q_tilde, using 
			# parameter names defined in its constructor
			for i in range(len(anl_params.Q_tilde_prior_modes)):
				FC = anl_params.Q_tilde_prior_FC[i]
				mode = anl_params.Q_tilde_prior_modes[i]
				lg = lognorm_pdf(sigma=np.log(FC), fixed_mode=mode)
				pdfs_dict[model().Q_tilde().p_name[i]] = lg.pdf
			
			return pdfs_dict