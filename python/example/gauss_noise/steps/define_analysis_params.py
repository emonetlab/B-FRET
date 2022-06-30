class anl_params():

	def __init__(a):
	
		a.process_noise = 'Gaussian'
		a.state_est_all_cores = True
		
		if a.process_noise == 'Non-Gaussian':
			
			# Distr of hyperparameter nu for Student's t (lognormal prior)
			# There are 2 parameters here, each with the mode and fold-change 
			# defining the lognormal distribution of their prior distribution
			# sigma and nu, respectively
			a.Q_tilde_prior_modes = [100, 1]
			a.Q_tilde_prior_FC = [30, 10]
			
			# Set to 'SMC' or 'integrate'. 
			a.state_est_method = 'integrate'
			
			# Parameters to do manual integration for chi pdf
			a.chi_interval_width = 8
			a.chi_num_of_subintervals = 400
			
			# Parameters for sequential Monte Carlo (paticle filter)
			a.num_SMC_samples = 300
			
		else:
			
			# Params defining the prior for the sigma of chi's process noise
			# i.e. Q in the SSM is [[0, 0], [0, sigma_chi^2]], where sigma_chi
			# is Bayesian updated from a lognormal prior defined by:
			a.sigma_chi_prior_FC = 30
			a.sigma_chi_prior_mode = 100

		# Use MCMC to sample from posterior? If not, use Laplace approximation
		# Workers needs to be minimum 4 (and must be even) or it will fail.
		# For ensemble slice sampling to work, it is suggested to use 
		# 2*num_params for the number of walkers. However, this is not required.
		# d_init is the size of the ball, in logspace, over which the 
		# workers are initialized. 0.1 ~ 10% of the value seems appropriate.
		# This value must be positive and greater than 0 or sampling will fail.
		a.do_MCMC = False
		if a.do_MCMC:
			a.MCMC_all_cores = True
			a.MCMC_d_init = 0.1
			a.MCMC_num_workers = 14
			a.MCMC_num_samples = 3000
			a.MCMC_burnin = 300
		else:
			a.MAP_num_samples = 1000
			a.MAP_opt_options = {'maxiter': 10000, 'ftol': 1e-8}
			
		# Options for the optimization for curve fitting of bleaching functions
		a.curve_fit_options = dict()
		a.curve_fit_options['ftol'] = 1e-6
		a.curve_fit_options['maxfun'] = 1e5
		a.curve_fit_options['maxiter'] = 4000
		
		# Priors on the state variables for filtering
		# m = [mean of constant variable, mean of chi]
		# == [1, m0_chi]
		# P = [[var of constant variable, 0], [0, variance of chi]]
		# == [[P0_x1, 0], [0, P0_chi]]
		# First state variable is just =1, up to this level of process noise
		a.P0_x1 = 1e-10

		# chi statistics at time 0 -- mean and std of chi estimate at time 0
		a.m0_chi = 0 
		a.P0_chi = 1e6