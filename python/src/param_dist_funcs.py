import time, os, warnings, pickle
from scipy.optimize import curve_fit, minimize
from autograd import grad, hessian
import autograd.numpy as np
from utils import kalman_filter, zeus_mcmc, lognorm_pdf, \
				  non_Gauss_1D_filtering, diag_gaussian_2d_pdf, \
				  nearest_PD
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class state_space_model():
	"""
	Defines the state space model matrices and vectors for FRET estimation
	"""
	
	def __init__(self, analyze_FRET_data_inst):
		"""
		Constructor for state space model.
		"""
		
		self.a = analyze_FRET_data_inst
		
		self._R = np.zeros((self.a.num_IDD, 2, 2))
		self._R[:, 0, 0] = self.a.IDD_noise_sd**2
		self._R[:, 1, 1] = self.a.IDA_noise_sd**2
		
		self.crstlk_a = self.a.data_dict['crstlk']['a']
		self.crstlk_d = self.a.data_dict['crstlk']['d']
		
		self._y = np.vstack((self.a.IDD, self.a.IDA)).T
		
		# Initial statistics of Gaussian distribution
		self.P_0 = np.array([[self.a.anl_params.P0_x1, 0], 
							 [0, self.a.anl_params.P0_chi]])
		self.m_0 = np.array([1, self.a.anl_params.m0_chi])
		
	def Q(self, params):
		"""
		Returns process noise matrix assuming Gaussian process noise on chi.
		"""
		
		sigma_chi_idx = 2 + self.a.num_params_f_D + self.a.num_params_f_A
		_Q = np.array([[0, 0], [0, params[sigma_chi_idx]**2.0]]) 
		return _Q
				
	def H(self, params):
		"""
		Returns measurement matrix at all timepoints, for given parameters
		"""
		
		# Crosstalk parameters
		G = self.a.data_dict['crstlk']['G']
		
		# Evaluate functions at ALL timepoints at once
		f_D_params = params[2: 2 + self.a.num_params_f_D]
		f_A_params = params[2 + self.a.num_params_f_D: 
							2 + self.a.num_params_f_D + self.a.num_params_f_A]
		f_D = self.a.m.f_D.func(self.a.tDD, f_D_params)
		f_A = self.a.m.f_A.func(self.a.tDD, f_A_params)
		
		DT = params[0]
		AT = params[1]
		_H00 = DT*f_D
		_H01 = -f_D*f_A
		_H10 = self.crstlk_d*DT*f_D + self.crstlk_a*AT*f_A
		_H11 = (G - self.crstlk_d)*f_D*f_A
		_H = np.array([[_H00, _H01], [_H10, _H11]])
		_H = np.moveaxis(_H, [0, 1, 2], [1, 2, 0])
		
		return _H
	
	def R(self):
		"""
		Returns measurement covariance at all times, as a function of params
		"""
		
		return self._R
		
	def y(self):
		"""
		Returns data vectors at all times
		"""
		
		return self._y
		

class param_est():
	"""
	Class to run all parameter estimation routines
	"""
	
	def __init__(self, analyze_FRET_data_inst):
		"""
		Constructor takes `analyze_FRET_data' instance and inherits its attrs
		"""
	
		self.a = analyze_FRET_data_inst
		out_dir = self.a.res_dir + '/param_dists'
		if not os.path.exists(out_dir):
			os.makedirs(out_dir)
		
	def plot_initial_est_bleach(self, f_D_prior_loc, f_A_prior_loc):
		"""
		Plotting routines for initial bleaching curve estimation
		"""
	
		fig = plt.figure(figsize=(8, 3))
		ax1 = plt.subplot(121)
		ax1.set_title(r'$I_{DD}$', fontsize=15)
		ax1.plot(self.a.tDD, self.a.IDD, color='0.5', lw=1, 
				 label=r'Measured $I_{DD}$')
		ax1.plot(self.a.tDD, f_D_prior_loc[-1]*self.a.m.f_D.func(self.a.tDD, 
				 f_D_prior_loc[:-1]), color='r', lw=2, label='Fit')
		ax1.tick_params(which='both', labelsize=10)
		ax1.legend()
		ax1.set_xlabel('Time (s)')
		ax1.set_ylabel('Intensity')
		
		ax2 = plt.subplot(122)
		ax2.set_title(r'$I_{AA}$', fontsize=15)
		ax2.plot(self.a.tAA, self.a.IAA, color='0.5', lw=1, 
				 label=r'Measured $I_{AA}$')
		ax2.plot(self.a.tAA, f_A_prior_loc[-1]*self.a.m.f_A.func(self.a.tAA, 
				 f_A_prior_loc[:-1]), color='r', lw=2, label='Fit')
		ax2.tick_params(which='both', labelsize=10)
		ax2.legend()
		ax2.set_xlabel('Time (s)')
		
		plt.tight_layout()
		out_dir = self.a.res_dir + '/param_dists'
		plt.savefig('%s/initial_est_bleach_trend.png' % out_dir, dpi=300)
		plt.close('all')
		
	def calc_prior_modes(self):
		"""
		Finds the modes of the prior distributions by naive curve fitting of
		the bleaching curves. This fixes the priors for the paramters 
		defining f_A and f_D
		"""
	
		# Fit donor bleaching curve to get mode of prior dist. 
		# Add the initial IDD value to be estimated (gives scaling constant)
		f_D_params_init = self.a.m.f_D.p_init + [self.a.IDD[0]]
		f_A_params_init = self.a.m.f_A.p_init + [self.a.IAA[0]]
		
		# Bounds for f_D and f_A parameters
		f_D_lo_bnd = self.a.m.f_D.p_lo_bnd + [0]
		f_D_hi_bnd = self.a.m.f_D.p_hi_bnd + [np.inf]
		f_A_lo_bnd = self.a.m.f_A.p_lo_bnd + [0]
		f_A_hi_bnd = self.a.m.f_A.p_hi_bnd + [np.inf]
		
		def f_D_bleach(t, *params):
			return params[-1]*self.a.m.f_D.func(t, params[:-1])
		def f_A_bleach(t, *params):
			return params[-1]*self.a.m.f_A.func(t, params[:-1])
		f_D_prior_loc = curve_fit(f_D_bleach, self.a.tDD, self.a.IDD, 
								  p0=f_D_params_init, 
								  bounds=(f_D_lo_bnd, f_D_hi_bnd))[0]
		f_A_prior_loc = curve_fit(f_A_bleach, self.a.tAA, self.a.IAA, 
								  p0=f_A_params_init, 
								  bounds=(f_A_lo_bnd, f_A_hi_bnd))[0]
		
		# Just check that the fit lines up to the data trend
		self.plot_initial_est_bleach(f_D_prior_loc, f_A_prior_loc)
			
		# Aggregate modes of all param priors. Order is: DT, AT, [f_D params], 
		# [f_A params], [process noise params].
		# Note that DT is defined by fitting (IDD[0] and IDA[0]), rather
		# than taking the coefficient of the curve fit
		prior_modes = []
		DT = (self.a.IDA[0] + 
		  (self.a.data_dict['crstlk']['G'] - self.a.data_dict['crstlk']['d'])
		  *f_D_prior_loc[-1] - self.a.data_dict['crstlk']['a']
		  *f_A_prior_loc[-1])/self.a.data_dict['crstlk']['G']
		AT = f_A_prior_loc[-1]
		prior_modes.append(DT)
		prior_modes.append(AT)
		prior_modes.extend(f_D_prior_loc[:-1])
		prior_modes.extend(f_A_prior_loc[:-1])
		if self.a.anl_params.process_noise == 'Gaussian':
			prior_modes.append(self.a.anl_params.sigma_chi_prior_mode) 
		elif self.a.anl_params.process_noise == 'Non-Gaussian':
			for mode in self.a.anl_params.Q_tilde_prior_modes:
				prior_modes.append(mode)
		assert len(prior_modes) == self.a.num_params, "Length of prior modes "\
		  "(%s) must be number of total parameters = %s"\
		  % (len(prior_modes), self.a.num_params)
		
		return np.array(prior_modes)
		
	def gen_prior_dist(self):
		"""
		Fixes prior distributions by finding the mode of the hyperparameter
		distributions, then setting these hyperparameters as fixed.
		"""
		
		# Sets just the functional forms of the prior distributions. pdfs
		# are a function of the hypeparameters still, though.
		prior_dist_funcs = self.a.m.prior_dict.func(self.a.anl_params)
		
		# Maximize prior distributions over their hyperparameters. 
		self.prior_modes = self.calc_prior_modes()
		
		# Now, set hyperparameter to optimum. Prior distributions are now fixed.
		# Order is same as usd in self.calc_prior_modes: DT, AT, [f_D params], 
		# [f_A params], [process noise params]
		self.priors = [prior_dist_funcs[self.a.param_names[i]]
						(self.prior_modes[i]) for i in range(self.a.num_params)]
		
	def calc_obs_likelihoods(self, p, chis):
		"""
		Calculate an observation likelihood at each timepoint, for an array
		of chis. Used for non-Gaussian proccess noise in Bayes filtering. 
		"""
		
		int_N = len(chis)
		H = self.ssm.H(p)
		R = self.ssm.R()
		y = self.ssm.y()
		LHs = []
		for i in range(self.a.num_IDD):
			mean1 = H[i, 0, 0]*np.ones(int_N) + H[i, 0, 1]*chis
			mean2 = H[i, 1, 0]*np.ones(int_N) + H[i, 1, 1]*chis
			means = np.vstack((mean1, mean2))
			covs = np.array([R[i] for j in range(int_N)])
			covs = np.moveaxis(covs, [0, 1, 2], [2, 0, 1])
			
			# Copy observations to all chi values to vectorize computation
			ys = np.array([y[i] for j in range(int_N)]).T
			LHs.append(diag_gaussian_2d_pdf(ys, means, covs))
		LHs = np.array(LHs)
		
		return LHs

	def calc_non_gauss_chi_range(self):
		""" 
		Adaptively calculate the range of chis for the integration for
		Non-Gaussian filtering
		"""
	
		a = self.a.data_dict['crstlk']['a']
		d = self.a.data_dict['crstlk']['d']
		G = self.a.data_dict['crstlk']['G']
		DT = self.prior_modes[0]
		AT = self.prior_modes[1]
		idx_lo = 2 + self.a.num_params_f_D
		idx_hi = idx_lo + self.a.num_params_f_A
		f_A_params = self.prior_modes[idx_lo:idx_hi]

		# Calculate estimated chi values for whole time series
		IAA_est = AT*self.a.m.f_A.func(self.a.tDD, f_A_params)
		Fc = np.max(self.a.IDA - d*self.a.IDD - a*IAA_est, 0)
		R = Fc/self.a.IDD
		E_corr = R/(R + G)*(IAA_est[0]/IAA_est)
		chi_est =  DT*E_corr

		chi_med = np.median(chi_est)
		chi_range =  np.percentile(chi_est, 99) - np.percentile(chi_est, 1)
		max = chi_med + (self.a.anl_params.chi_interval_width/2)*chi_range
		min = chi_med - (self.a.anl_params.chi_interval_width/2)*chi_range
		
		chis = np.linspace(min, max, self.a.anl_params.chi_num_of_subintervals)
		
		return chis

	def neg_log_posterior(self, log_p):
		""" 
		Calculates the negative log of the posterior of the parameters given 
		the data.
		"""
		
		p = np.exp(log_p)
		
		# log[prob(p)]
		neg_log_prior = -np.sum([np.log(f(p)) for f, p in zip(self.priors, p)])
		
		AT = p[1]
		f_A_params = p[2 + self.a.num_params_f_D: 
					   2 + self.a.num_params_f_D + self.a.num_params_f_A]
		
		# Technically the commented term term is in the posterior, but does 
		# not affect the optimal estimate since it's a scalar
		#neg_log_IAA = 0.5*np.sum(np.log(self.a.IAA_noise_sd**2))
		neg_log_IAA = 0.5*np.sum((self.a.IAA - AT*self.a.m.f_A.func(
								  self.a.tAA, f_A_params))**2/
								  self.a.IAA_noise_sd**2)
		
		# log[prob(y == IDD, IDA | p)]
		if self.a.anl_params.process_noise == 'Gaussian':
			_, _, _, _, S, v = kalman_filter(self.ssm.Q(p), self.ssm.H(p),
											 self.ssm.R(), self.ssm.y(), 
											 self.ssm.m_0, self.ssm.P_0, 
											 self.a.tDD)
		
			# Perhaps there is a cleaner way, but at least it's vectorized
			S_inv = np.linalg.inv(S)
			vS1 = np.sum(S_inv[:, 0, :]*v, axis=-1)
			vS2 = np.sum(S_inv[:, 1, :]*v, axis=-1)
			vS = np.vstack((vS1, vS2)).T
			vSv = np.sum(vS*v, axis=-1)
			neg_log_y = 0.5*(np.sum(vSv + np.log(np.linalg.det(2*np.pi*S))))
			
		elif self.a.anl_params.process_noise == 'Non-Gaussian':
			
			# Define state transition pdf; here we assume deterministic
			# dynamics are trivial, x --> x + q; q is defined by q_tilde
			idx_lo = 2 + self.a.num_params_f_D + self.a.num_params_f_A
			idx_hi = idx_lo + self.a.num_params_Q_tilde
			Q = self.a.m.Q_tilde.pdf(p[idx_lo:idx_hi])
			
			m_0 = self.a.anl_params.m0_chi
			P_0 = self.a.anl_params.P0_chi
			
			chis = self.calc_non_gauss_chi_range()
			
			# Observation likelihoods
			LHs = self.calc_obs_likelihoods(p, chis)
			
			# Filter and smooth by integrating over chi at each timestep
			dist = non_Gauss_1D_filtering(chis, Q, LHs, m_0, P_0, self.a.tDD)
			
			dx, p1 = dist['dx'], dist['pred_probs']
			neg_log_y = -np.sum(np.log(np.sum(dx*p1*LHs, axis=-1)), axis=0)
		
		log_post = neg_log_prior + neg_log_IAA + neg_log_y
		print (log_post)
		return  log_post
		
	def log_posterior(self, log_p):
		"""
		Used to maximize the posterior for MCMC
		"""
		return -self.neg_log_posterior(log_p)
		
	def gen_post_dist_laplace_approx(self):
		"""
		Use Laplace approximation to generate the posterior by maximimizing 
		the log likelihood, then getting the Hessian to approximate as a 
		normal distribution.
		"""
		
		print ('Optimizing log posterior to get mode for Laplace approx..')
		timenow = time.time()
		with warnings.catch_warnings():
			warnings.simplefilter('ignore')
			
			# Impose bounds for the bleaching parameters as provided by user
			# Let's keep everything between 1e-8 and 1e8 to avoid overflow
			nmax = 1e8
			nmin = 1e-8
			bounds = np.array([[np.log(nmin), np.log(nmax)]]*self.a.num_params)
			for i in range(self.a.num_params_f_D):
				_i = i + 2
				bounds[_i][0] = np.log(max(self.a.m.f_D.p_lo_bnd[i], nmin))
				bounds[_i][1] = np.log(min(self.a.m.f_D.p_hi_bnd[i], nmax))
			for i in range(self.a.num_params_f_A):
				_i = i + self.a.num_params_f_D + 2
				bounds[_i][0] = np.log(max(self.a.m.f_A.p_lo_bnd[i], nmin))
				bounds[_i][1] = np.log(min(self.a.m.f_A.p_hi_bnd[i], nmax))
			
			# Minimize using sequential least squares programming
			res = minimize(self.neg_log_posterior, np.log(self.prior_modes),
						   bounds=bounds, method='SLSQP',
						   options=self.a.anl_params.MAP_opt_options)
			
		# Hessian of log posterior is with respect to *log* of parameters.
		self.MAP_log_hess = hessian(self.neg_log_posterior)(res.x)
		self.MAP_inv_log_hess = np.linalg.inv(self.MAP_log_hess)
		eigs = np.linalg.eig(self.MAP_inv_log_hess)[0]
		if np.any(eigs < 0):
			self.MAP_inv_log_hess = nearest_PD(self.MAP_inv_log_hess)
			eigs = np.linalg.eig(self.MAP_inv_log_hess)[0]
		self.MAP_log_p = res.x
		self.BIC = 2*res.fun + self.a.num_params*np.log(self.a.num_IDD 
														+ self.a.num_IDA 
														+ self.a.num_IAA)
		print ('Function value: ', res.fun, 'BIC:', self.BIC,  
			   'Opt success:', res.success, '\n', 'Optimum:', np.exp(res.x))
		print ('Covariance matrix eigs', eigs)
		print ('Time elapsed:', time.time() - timenow)
		
	def plot_prior_and_post_dists(self):
		"""
		Plots prior and posterior distributions in both linear and log.
		"""
		
		for iP in range(self.a.num_params):
			
			fig = plt.figure(figsize=(8, 3))
			ax1 = plt.subplot(121)
			ax2 = plt.subplot(122)
			
			# Plot priors on log scale
			log_mid = np.log(self.prior_modes[iP])/np.log(10)
			x = np.logspace(log_mid - 4, log_mid + 4, 100000)
			y = self.priors[iP](x)
			idx_rng = y > 0.1*max(y)
			ax1.plot(x[idx_rng], y[idx_rng], color='dodgerblue', 
					 lw=2, label='Prior')
			ax1.set_xlim(min(x[idx_rng]), max(x[idx_rng]))
			ax1.set_ylim(0, 2*max(y))
			
			# Plot posterior on log scale
			if self.a.anl_params.do_MCMC:
				mode = np.median(self.post_p_samples[:, iP])
				ax1.hist(self.post_p_samples[:, iP], bins=100, density=True, 
						 color='r', label='Posterior')
			else:
				std = self.MAP_inv_log_hess[iP, iP]**0.5
				mode = np.exp(self.MAP_log_p[iP])
				y = lognorm_pdf(std).pdf(mode)(x)
				ax1.plot(x, y, color='r', label='Posterior')
			ax1.axvline(mode, ls='--', color='r', label='MAP')
			
			# Repeat for linear scale
			mid = self.prior_modes[iP]
			x = np.linspace(mid/1e3, mid*1e3, 100000)
			y = self.priors[iP](x)
			idx_rng = y > 0.1*max(y)
			ax2.plot(x[idx_rng], y[idx_rng], color='dodgerblue', 
					 lw=2, label='Prior')
			ax2.set_xlim(min(x[idx_rng]), max(x[idx_rng]))
			ax2.set_ylim(0, 2*max(y))
			if self.a.anl_params.do_MCMC:
				mode = np.median(self.post_p_samples[:, iP])
				ax2.hist(self.post_p_samples[:, iP], bins=100, density=True, 
						 color='r', label='Posterior')
			else:			
				y = lognorm_pdf(std).pdf(mode)(x)
				ax2.plot(x, y, color='r', label='Posterior')
			ax2.axvline(mode, ls='--', color='r', label='MAP')
			
			name = self.a.param_names[iP]
			ax1.set_title(name)
			ax1.tick_params(which='both', labelsize=10)
			ax1.set_xscale('log')
			ax1.set_ylabel('pdf')
			ax1.legend()
			ax2.set_title(name)
			ax2.tick_params(which='both', labelsize=10)
			ax2.set_xlabel(name)
			ax2.legend()
			plt.tight_layout()
			out_dir = self.a.res_dir + '/param_dists'
			plt.savefig('%s/%s.png' % (out_dir, name), dpi=300)
		
	def sample_from_post_dist(self):
		"""
		Gets samples from posterior of parameters by doing either Laplace 
		approximation or MCMC. These are then used to estimate the states.
		"""
		
		self.ssm = state_space_model(self.a)
		
		vars = ['post_p_samples']
		if self.a.anl_params.do_MCMC:
			log_samples = zeus_mcmc(self.log_posterior, 
									np.log(self.prior_modes), 
									self.a.anl_params.MCMC_num_workers, 
									self.a.anl_params.MCMC_num_samples, 
									burnin=self.a.anl_params.MCMC_burnin, 
									all_cores=self.a.anl_params.MCMC_all_cores,
									d_init=self.a.anl_params.MCMC_d_init)
			
		else:
			self.gen_post_dist_laplace_approx()
			log_samples = np.random.multivariate_normal(
						    self.MAP_log_p, 
							self.MAP_inv_log_hess, 
							self.a.anl_params.MAP_num_samples)
			vars += ['MAP_log_p', 'MAP_log_hess', 'MAP_inv_log_hess', 'BIC']
		self.post_p_samples = np.exp(log_samples)
		
		# Save results and plot the parameter distributions
		data = dict()
		for var in vars:
			exec("data['%s'] = self.%s" % (var, var))
		out_dir = self.a.res_dir + '/param_dists'
		if self.a.anl_params.do_MCMC:
			filename = '%s/posterior_data_MCMC.pkl' % out_dir
		else:
			filename = '%s/posterior_data.pkl' % out_dir
		pickle.dump(data, open(filename, 'wb'))
		self.plot_prior_and_post_dists()	
		