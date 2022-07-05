import time
import autograd.numpy as np
from numpy import linalg as la
from scipy.stats import norm, lognorm, uniform
from scipy.integrate import quad, quadrature
from pathos.multiprocessing import ProcessingPool
from pathos.helpers import cpu_count
		
		
def diag_gaussian_2d_pdf(x, mean, cov):
	"""
	Returns the pdf of a 2d Gaussian where each dimension is separable.
	i.e. the covariance matrix must be diagonal. Should be faster
	than using matrix multiplication for a general 2D covariance.
	"""
	
	assert np.prod(cov[0, 1]*cov[1, 0]) == 0, 'Need diagonal covariance'
	
	exp_arg = -0.5*(1/cov[0, 0]*(x[0] - mean[0])**2 \
		  + 1/cov[1, 1]*(x[1] - mean[1])**2)
	
	return np.exp(exp_arg)/2*np.pi/np.sqrt(cov[0, 0]*cov[1, 1])
	
class norm_pdf():		
	"""
	Class for pdf of a normal distribution given fixed SD and mean.
	"""
	
	def __init__(self, sigma, fixed_mode=None):
		self.sigma = sigma
		self.fixed_mode = fixed_mode
		
	def _f(self, x):
		return 1/np.sqrt(2*np.pi*self.sigma**2)*\
			   np.exp(-(x - self.mean)**2/2/self.sigma**2)
		
	def pdf(self, p_init):
	
		if self.fixed_mode is None:
			self.mean = p_init
		else:
			self.mean = self.fixed_mode
		
		return self._f
		
class lognorm_pdf():
	"""
	Class for pdf of a lognormal distribution given fixed SD and mode.
	"""
	
	def __init__(self, sigma, fixed_mode=None):
		self.sigma = sigma
		self.fixed_mode = fixed_mode
		
	def _f(self, x):
		return 1/(x*self.sigma*np.sqrt(2*np.pi))*\
			   np.exp(-(np.log(x) - self.mean)**2/(2*self.sigma**2))
		
	def pdf(self, p_init):
	
		# Assume p_init is mode of lognormal -- convert to mean
		if self.fixed_mode is None:
			self.mean = np.log(p_init) + self.sigma**2.0
		else:
			self.mean = np.log(self.fixed_mode) + self.sigma**2.0
		
		return self._f
		
class standard_uniform_pdf():
	"""
	Class for pdf of standard uniform distribution
	"""
	
	def __init__(self):
		pass
		
	def _f(self, x):
		return 1.*(x > 0.)*(x <= 1.)
	
	def pdf(self, p_init):
		
		return self._f
		
def kalman_filter(Q, H, R, y, m_0, P_0, t_vec):
	"""
	Calculates posterior distribution at points in t_vec, for Gaussian 
	measurements and a linear-Gaussian model dynamicsl, using the Kalman
	filter update equations.
	"""
	
	N = len(t_vec)
	median_delta_t = np.median(np.diff(t_vec))
	d_t_vec = 0
	
	# Indexed assignment is not allowed by autograd so must append to lists
	m_pred = []
	P_pred = []
	m = []
	P = []
	S = []
	v = []
	
	m_k = m_0
	P_k = P_0
		
	for k in range(N):
		
		m_pred_k = m_k

		# Q is constant
		if len(Q.shape) == 2:
			P_pred_k = P_k + Q
		
		# If Q is time-dependent
		else:
		
			# If sampling interval large, don't use recent information
			if (k == 0) or (d_t_vec > 2*median_delta_t):
				P_pred_k = P_k + Q[0]
			else:
				P_pred_k = P_k + Q[k - 1]
		
		v_k = y[k] - np.dot(H[k], m_pred_k)
		S_k = np.dot(np.dot(H[k], P_pred_k), H[k].T) + R[k]
		K_k = np.dot(np.dot(P_pred_k, H[k].T), np.linalg.inv(S_k))
		m_k = m_pred_k + np.dot(K_k, v_k)
		P_k = P_pred_k - np.dot(np.dot(K_k, H[k]), P_pred_k)
		
		m_pred.append(m_pred_k)
		P_pred.append(P_pred_k)
		m.append(m_k)
		P.append(P_k)
		S.append(S_k)
		v.append(v_k)
		t_vec_increase = t_vec[min(k + 1, N - 1)] - t_vec[k]
	
	return np.array(m_pred), np.array(P_pred), np.array(m), \
			 np.array(P), np.array(S), np.array(v)

def RTS_smoother(m_pred, P_pred, m, P):
	"""
	Calculates smoothed posterior at points in t_vec, for Gaussian 
	measurements and a linear-Gaussian model dynamicsl, using the 
	predicted distributions of a Kalman filter through the dataset.
	"""
	
	N = m.shape[0]
	
	ms = [m[-1]]
	Ps = [P[-1]]
	
	ms_k = ms[0]
	Ps_k = Ps[0]
	
	# Iterate from penultimate index to 0
	for k in np.arange(N - 2, -1, -1):
		G_k = np.dot(P[k], np.linalg.inv(P_pred[k + 1]))
		ms_k = m[k] + np.dot(G_k, ms_k - m_pred[k + 1])
		Ps_k = P[k] + np.dot(G_k, np.dot(Ps_k - P_pred[k + 1], G_k.T))
		ms.append(ms_k)
		Ps.append(Ps_k)
	
	# Arrays are defined backward
	ms = ms[::-1]
	Ps = Ps[::-1]
	
	return np.array(ms), np.array(Ps)

def zeus_mcmc(log_prob, init, num_wkrs, num_samples, burnin=0, all_cores=True,
			  d_init=0.1):
	"""
	Slice sampling to generate MCMC samples of the log of an unnormalized
	probability distribution. Can run on parallel cores of a node.
	"""
	
	print ('MCMC sampling of posterior distribution ...')
	timenow = time.time()
	num_dim = len(init)
	init_tiled = np.zeros((num_wkrs, num_dim))
	
	import zeus	
	def run_parallel(seed):
		
		np.random.seed(seed)
		for i in range(num_wkrs):
			inits = np.random.normal(init, d_init, num_dim)
			while not np.isfinite(log_prob(inits)):
				inits = np.random.normal(init, d_init, num_dim)
			init_tiled[i] = inits
		samplers = zeus.EnsembleSampler(num_wkrs, num_dim, log_prob,
					 light_mode=True, check_walkers=False) 
		samplers.run_mcmc(init_tiled, num_samples)
		samples = samplers.get_chain(flat=True, discard=burnin)
		
		return samples
	
	if all_cores:
		pool = ProcessingPool()
		chains = pool.map(run_parallel, range(cpu_count()))
		all_samples = np.concatenate(chains, axis=0)
	else:
		all_samples = run_parallel(0)
	
	print ('Time elapsed:', time.time() - timenow)
	return all_samples
	
def non_Gauss_1D_filtering(xs, Q_pdf, LHs, m_0, P_0, t_vec):
	"""
	Generates the posterior distribution given i) a Gaussian prior, ii) the
	observation likelihoods at all points, as a funcition of discrete points 
	x, and iii) a function that returns pdf of the process noise from x_{t} 
	to x_{t + 1}. The pdf is a function of (x_{t+1} - x_t}, which can be 
	non-Gaussian. This is for a 1D state space.
	"""
	
	dx = xs[1] - xs[0]
	N = len(t_vec)
	int_N = len(xs)
	assert LHs.shape[0] == N, 'Likelihood arr must be (# timesteps, # xs)'
	assert LHs.shape[1] == int_N, 'Likelihood arr must be (# timesteps, # xs)'
	
	# Probability of transition from x_n (row) to x_{n + 1} (column). 
	Q_mat = Q_pdf(xs[:, None] - xs[None, :])
	
	prior = norm_pdf(P_0**0.5).pdf(m_0)(xs)
	pred_probs = []
	post_probs = []
	
	for i in range(N):
		
		# Predictive step; yes this is the right order; since rows are x_n
		p_tilde = np.dot(prior, Q_mat)
		p_tilde /= np.sum(dx*p_tilde)
		
		# Update step 
		f_tilde = LHs[i]*p_tilde
		f_tilde /= np.sum(dx*f_tilde)
		
		# Prior for next timestep is current posterior
		prior = f_tilde
		pred_probs.append(p_tilde)
		post_probs.append(f_tilde)
	
	dist = dict()
	dist['pred_probs'] = np.array(pred_probs)
	dist['post_probs'] = np.array(post_probs)
	dist['Q_mat'] = Q_mat
	dist['xs'] = xs
	dist['dx'] = dx
	
	return dist
	
def non_Gauss_1D_smoothing(dist):
	
	Q_mat = dist['Q_mat']
	p_tildes = dist['pred_probs']
	f_tildes = dist['post_probs']
	N = len(p_tildes)
	dx = dist['dx']
	
	# To hold backwards smoothed posteriors at each timepoint
	smth_post_probs = []
	cdfs = []
	s_tilde = f_tildes[-1]
	smth_post_probs.append(s_tilde)
	cdfs.append(np.cumsum(dx*s_tilde))
	
	for i in range(N - 2, -1, -1):
		
		f_tilde = f_tildes[i]
		p_tilde = p_tildes[i + 1]
		p_tilde[p_tilde < 1e-10] = 1e-10
		s_tilde = f_tilde*np.dot(s_tilde/p_tilde, Q_mat)
		s_tilde /= np.sum(s_tilde*dx)
		smth_post_probs.append(s_tilde)
		cdfs.append(np.cumsum(dx*s_tilde))
	
	dist['smth_post_probs'] = smth_post_probs[::-1]
	dist['cdfs'] = cdfs[::-1]
	
	return dist
	

def nearest_PD(A):
	"""
	Find the nearest positive-definite matrix to input
	"""

	B = (A + A.T)/2
	_, s, V = la.svd(B)
	H = np.dot(V.T, np.dot(np.diag(s), V))
	A2 = (B + H)/2
	A3 = (A2 + A2.T)/2

	if is_PD(A3):
		return A3

	spacing = np.spacing(la.norm(A))
	I = np.eye(A.shape[0])
	k = 1
	
	while not is_PD(A3):
		mineig = np.min(np.real(la.eigvals(A3)))
		A3 += I * (-mineig * k**2 + spacing)
		k += 1

	return A3


def is_PD(B):
	"""
	Returns true when input is positive-definite, via Cholesky
	"""
	
	try:
		la.cholesky(B)
		return True
		
	except la.LinAlgError:
		return False