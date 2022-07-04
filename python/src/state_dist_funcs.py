import time, os, pickle
import autograd.numpy as np
from pathos.helpers import cpu_count
from pathos.multiprocessing import ProcessingPool
from utils import kalman_filter, RTS_smoother, non_Gauss_1D_filtering, \
				  non_Gauss_1D_smoothing, diag_gaussian_2d_pdf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
		

class state_est():

	def __init__(self, analyze_FRET_data_inst, param_est_inst):
		"""
		Constructor takes `analyze_FRET_data' instance and 'param_est' 
		instance and inherits their attributes. Parameter estimation
		is run prior to this to generate a sample of parameters
		from the parameter posterior distribution.
		"""
		
		self.a = analyze_FRET_data_inst
		self.p = param_est_inst
		self.ssm = self.p.ssm
		self.ps = self.p.post_p_samples
		
		out_dir = self.a.res_dir + '/state_dists'
		if not os.path.exists(out_dir):
			os.makedirs(out_dir)
		
	def single_p_integrate_Gaussian(self, p):
		"""
		Get the state posterior distribution for Gaussian process noise, which
		is solved in closed form using the Kalman filter and RTS smoother.
		This is done for a fixed parameter vector `p'.
		"""
		
		# Forward pass / Kalman filter
		m_pred, P_pred, m, P, _, _ = kalman_filter(self.ssm.Q(p), 
												   self.ssm.H(p), 
												   self.ssm.R(), 
												   self.ssm.y(), 
												   self.ssm.m_0, 
												   self.ssm.P_0, 
												   self.a.tDD)
	
		# Backwards pass / Rauch-Tung-Streibel smoother
		ms, Ps = RTS_smoother(m_pred, P_pred, m, P)
		
		return [ms, Ps]
	
	def single_p_integrate_non_Gaussian(self, p):
		"""
		Get the state posterior distribution for non-Gaussian process noise.
		This is done by directly integrating the Bayesian filtering and 
		smoothing equations, which is possible since the state (chi) is only
		1-dimensional. This is done for a fixed parameter vector `p'.
		"""
		
		# Define state transition pdf; here we assume deterministic
		# dynamics are trivial, x --> x + q; q is defined by q_tilde
		idx_lo = 2 + self.a.num_params_f_D + self.a.num_params_f_A
		idx_hi = idx_lo + self.a.num_params_Q_tilde
		Q = self.a.m.Q_tilde.pdf(p[idx_lo:idx_hi])
		
		m_0 = self.a.anl_params.m0_chi
		P_0 = self.a.anl_params.P0_chi
		
		# This is defined in the param_est object
		chis = self.p.calc_non_gauss_chi_range()
	
		# Observation likelihoods
		LHs = self.p.calc_obs_likelihoods(p, chis)
		
		# Filter and smooth by integrating over chi at each timestep
		dist = non_Gauss_1D_filtering(chis, Q, LHs, m_0, P_0, self.a.tDD)
		dist = non_Gauss_1D_smoothing(dist)
								  
		return dist
		
	def single_p_SMC_forward(self, p):
		"""
		Get the state posterior distribution for non-Gaussian process noise.
		This is done by using a particle filter. This only gets the filtering
		distribution by propagaint particles from t=0 to t=end. This is done
		for a fixed parameter vector `p'.
		"""
		
		N = self.a.anl_params.num_SMC_samples
		
		# Pick out parameters for Q_tilde distribution
		idx_lo = 2 + self.a.num_params_f_D + self.a.num_params_f_A
		idx_hi = idx_lo + self.a.num_params_Q_tilde
		Q_chi = self.a.m.Q_tilde.sample(p[idx_lo:idx_hi],
										size=(self.a.num_IDD, N))
		
		# Initialize particles from chi prior and give uniform weights
		xs = np.zeros((self.a.num_IDD, 2, N), dtype=np.float16)
		xs[0] = np.random.multivariate_normal(self.ssm.m_0, self.ssm.P_0,
											  size=N).T
		var1 = self.ssm.R()[:, 0, 0]
		var2 = self.ssm.R()[:, 1, 1]
		y = self.ssm.y()
		H = self.ssm.H(p)
		
		for i in range(self.a.num_IDD - 1):
			
			# Particle loctions
			const_pred = xs[i, 0]
			chi_pred = xs[i, 1] + Q_chi[i]
			x_pred = np.vstack((const_pred.T, chi_pred.T))
			
			# Bootstrap filter; don't need normalized prob w/ resampling
			res1 = y[i, 0] - (H[i, 0, 0]*const_pred + H[i, 0, 1]*chi_pred)
			res2 = y[i, 1] - (H[i, 1, 0]*const_pred + H[i, 1, 1]*chi_pred)
			prob  = np.exp(-res1**2/2/var1[i] - res2**2/2/var2[i])
			ws = prob/np.sum(prob)
			idxs = np.random.choice(range(N), N, p=ws)
			xs[i + 1] = x_pred[:, idxs]
			
		return xs
	
	def single_p_SMC_backward(self, p, xs):
		"""
		Get the state posterior smoothing distribution for non-Gaussian 
		process noise, using the xs and weights from the forward-pass 
		particle filter. 
		"""
		
		N = self.a.anl_params.num_SMC_samples
		
		# ws shape  is timesteps, num_particles
		ws = np.zeros((self.a.num_IDD, N), dtype=np.float16)
		ws[-1] = np.ones(N)/N
		
		# Pick out parameters for Q_tilde distribution
		idx_lo = 2 + self.a.num_params_f_D + self.a.num_params_f_A
		idx_hi = idx_lo + self.a.num_params_Q_tilde
		
		for i in range(self.a.num_IDD - 2, -1, -1):
			
			# Row is j = t + 1 index, column is t index
			mat_diffs = xs[i + 1, 1][:, None] - xs[i, 1][None, :]
			probs = self.a.m.Q_tilde.pdf(p[idx_lo:idx_hi])(mat_diffs)
			
			# Sum_k p(x_t+1(j) | x_t(k))
			den = np.sum(probs, axis=-1)
			
			# Sum_j ws_t+1(j) p(x_t+1(j) | x_t(i))
			# Take transponse of probs since sum is over 2nd index
			w = np.sum(probs.T/den*ws[i + 1], axis=-1)
			
			# If paucity of samples; just use the uniform filter weights
			if np.prod(np.isfinite(w)):
				ws[i] = w
				ws[i] /= np.sum(ws[i])
			else:
				ws[i] = np.ones(N)/N
		
		return ws
	
	def calc_pred(self):
		"""
		Calculate the predicted observable and E-FRET trace, using the 
		calculate posterior distributions. These are essentially point
		estimates of the distributions, along with confidence intervals.
		"""
		
		# Extracted E-FRET Signal for BFRET method with confidence intervals
		E = self.pred['E_samples']
		self.pred['E_med'] = np.median(E, axis=0)
		self.pred['E-1SD'] = np.percentile(E, 15.87, axis=0)
		self.pred['E-2SD'] = np.percentile(E, 2.27, axis=0)
		self.pred['E-3SD'] = np.percentile(E, 0.13, axis=0)
		self.pred['E+1SD'] = np.percentile(E, 84.13, axis=0)
		self.pred['E+2SD'] = np.percentile(E, 97.73, axis=0)
		self.pred['E+3SD'] = np.percentile(E, 99.87, axis=0)
		
		# I_DD, IDA, and I_AA observables to compare
		if self.a.anl_params.do_MCMC:
			p = np.median(self.p.post_p_samples, axis=0)
		else:
			p = np.exp(self.p.MAP_log_p)
		chi = np.median(self.pred['chi_samples'], axis=0)
		x = np.vstack((np.ones(self.a.num_IDD), chi)).T
		y_pred = np.einsum('kij,kj->ki', self.ssm.H(p), x)
		AT = p[1]
		f_A_params = p[2 + self.a.num_params_f_D: 
				   2 + self.a.num_params_f_D + self.a.num_params_f_A]
		IAA_pred = AT*self.a.m.f_A.func(self.a.tDD, f_A_params)
		self.pred['IDD'] = y_pred[:, 0]
		self.pred['IDA'] = y_pred[:, 1]
		self.pred['IAA'] = IAA_pred
		
		# Calculate algebraic E-FRET as a comparison
		G = self.a.data_dict['crstlk']['G']
		a = self.a.data_dict['crstlk']['a']
		d = self.a.data_dict['crstlk']['d']
		R = (self.a.IDA - d*self.a.IDD - a*IAA_pred)/self.a.IDD
		self.pred['E_FRET_corr'] = R/(R + G)*self.a.IAA[0]/IAA_pred
		
	def plot_pred(self):
		"""
		Save the predictions to file.
		"""
		
		out_dir = self.a.res_dir + '/state_dists'
		
		fig = plt.figure(figsize=(8, 4))
		plt.plot(self.a.tDD, self.a.IDD, color='b', label=r'$I_{DD}$ Meas', 
				 lw=0.5) 
		plt.plot(self.a.tDD, self.a.IDA, color='r', label=r'$I_{DA}$ Meas', 
				 lw=0.5) 
		plt.plot(self.a.tAA, self.a.IAA, color='g', label=r'$I_{AA}$ Meas', 
				 lw=0.5)
		plt.fill_between(self.a.tDD, self.pred['IDD'] - 2*self.a.IDD_noise_sd,
						 self.pred['IDD'] + 2*self.a.IDD_noise_sd, alpha=0.3,
						 color='b', label=r'$I_{DD}$ pred')
		plt.fill_between(self.a.tDD, self.pred['IDA'] - 2*self.a.IDA_noise_sd,
						 self.pred['IDA'] + 2*self.a.IDA_noise_sd, alpha=0.3,
						 color='r', label=r'$I_{DA}$ pred')
		IAA_interp = np.interp(self.a.tAA, self.a.tDD, self.pred['IAA'])
		plt.fill_between(self.a.tAA, IAA_interp - 2*self.a.IAA_noise_sd,
						 IAA_interp + 2*self.a.IAA_noise_sd, alpha=0.3,
						 color='g', label=r'$I_{AA}$ pred') 
		plt.tick_params(which='both', labelsize=13)
		plt.xlabel(r'Time (s)', fontsize=13)
		plt.ylabel(r'Intensities', fontsize=13)
		plt.xlim(left=0, right=None)
		plt.ylim(bottom=0, top=None)
		plt.legend()
		plt.tight_layout()
		filename = '%s/Observables.png' % out_dir
		plt.savefig(filename, dpi=600)
		
		fig = plt.figure(figsize=(8, 4))
		plt.plot(self.a.tDD, self.pred['E_med'], color='b', label='Pred')
		plt.plot(self.a.tDD, self.pred['E-1SD'], color='0.75', lw=0.5)
		plt.plot(self.a.tDD, self.pred['E-2SD'], color='0.75', lw=0.5)
		plt.plot(self.a.tDD, self.pred['E+1SD'], color='0.75', lw=0.5)
		plt.plot(self.a.tDD, self.pred['E+2SD'], color='0.75', lw=0.5)
		if hasattr(self.a, 'E_true'):
			plt.plot(self.a.tDD, self.a.E_true, color='r', label='True')
		plt.tick_params(which='both', labelsize=13)
		plt.xlabel(r'Time (s)', fontsize=13)
		plt.ylabel(r'$E = \chi/D_T$', fontsize=13)
		plt.xlim(left=0, right=None)
		plt.ylim(bottom=0, top=None)
		plt.legend()
		plt.tight_layout()
		filename = '%s/E.png' % out_dir
		plt.savefig(filename, dpi=600)
		plt.close('all')
		
	def sample_from_post_dist(self):
		"""
		Generate samples from the posterior distribution of the states, 
		for each parameter of the parameter distribution, using usual
		Bayesian inference. 
		"""
		
		# Number of parameter samples (not dimension of parameter space)
		num_ps = self.ps.shape[0]
		num_steps = self.a.num_IDD
		DTs = self.ps[:, 0]
		if self.a.anl_params.state_est_all_cores:
			num_cpus = cpu_count()
		else:
			num_cpus = 1
		
		# To hold the samples from the generated posteriors
		E_samples = np.zeros((num_ps, num_steps))
		chi_samples = np.zeros((num_ps, num_steps))
		
		timenow = time.time()
		
		# Solution is analytical for Gaussian noise; (RTS smoother)
		if self.a.anl_params.process_noise == 'Gaussian':
		
			print ('Running Kalman filter and RTS smoother...')
			pool = ProcessingPool(nodes=num_cpus)
			states = pool.map(self.single_p_integrate_Gaussian, self.ps)
			post_ms = np.array([states[i][0] for i in range(num_ps)])
			post_Ps = np.array([states[i][1] for i in range(num_ps)])
			
			for j in range(num_steps):
				chi_avgs = post_ms[:, j, 1]
				chi_stds = np.sqrt(post_Ps[:, j, 1, 1])
				chi_samples[:, j] = np.random.normal(chi_avgs, chi_stds)
				E_samples[:, j] = chi_samples[:, j]/DTs
			
		elif self.a.anl_params.process_noise == 'Non-Gaussian':
		
			# Particle filter for non-Gaussian transition model
			if self.a.anl_params.state_est_method == 'SMC':
			
				print ('Forward particle filter for Non-Gaussian model...')
				pool = ProcessingPool(nodes=num_cpus)
				xs = pool.map(self.single_p_SMC_forward, self.ps)
				xs = np.array(xs)
				print ('Backward particle filter for Non-Gaussian model...')
				pool = ProcessingPool(nodes=num_cpus)
				ws = pool.map(self.single_p_SMC_backward, self.ps, xs)
				ws = np.array([ws[i] for i in range(num_ps)])
				
				# Choose one particle for each time, each p, from SMC dist.
				# Shape of chi and w: (# param samples, time, # particles)
				chis = xs[:, :, 1]
				for i in range(num_ps):
					for j in range(num_steps):
						samp = np.random.choice(chis[i, j], p=ws[i, j])
						chi_samples[i, j] = samp
						E_samples[i, j] = samp/DTs[i]
				
			# Just directly integrate the Bayesian update equations
			elif self.a.anl_params.state_est_method == 'integrate':
			
				print ('Integrating Bayesian smoothing equations directly...')
				pool = ProcessingPool(nodes=num_cpus)
				dists = pool.map(self.single_p_integrate_non_Gaussian, self.ps)
				
				# Get sample using calculated cdf of posterior distributions
				chis = dists[0]['xs']
				rand_vals = np.random.uniform(0, 1, (len(self.ps), num_steps))
				for i in range(len(self.ps)):
					cdfs = dists[i]['cdfs']
					for j in range(num_steps):
						chis_interp = np.linspace(chis[0], chis[-1], 10000)
						cdfs_interp = np.interp(chis_interp, chis, cdfs[j])
						idx = np.where(rand_vals[i, j] <= cdfs_interp)[0][0]
						chi_samples[i, j] = chis_interp[idx]
						E_samples[i, j] = chis_interp[idx]/DTs[i]
								
		print ('Time elapsed: ', time.time() - timenow)	
		
		# Calculate predictions and CIs
		self.pred = dict()
		self.pred['E_samples'] = E_samples
		self.pred['chi_samples'] = chi_samples
		self.calc_pred()
		
		# Save results and plot the parameter distributions
		out_dir = self.a.res_dir + '/state_dists'
		filename = '%s/posterior_data.pkl' % out_dir
		pickle.dump(self.pred, open(filename, 'wb'))
		self.plot_pred()