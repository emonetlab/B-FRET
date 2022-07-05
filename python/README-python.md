# B-FRET (python)

This package uses Bayesian inference to generate posterior distributions of FRET signals from noisy measured FRET data. 



## Installation

`bayesian-efret-py` is a standalone package in Python 3. The code has been tested on Python 3.9.0. The following packages and the versions tested on the code are needed, all available on PIP:

1. [NumPy](https://numpy.org/) 1.20.2
2. [SciPy](https://scipy.org/) 1.6.2
3. [Autograd](https://pypi.org/project/autograd/) 1.3 needed for automatic differentiation of the cost function
4. [matplotlib](https://matplotlib.org/) 3.3.4
5. [pathos](https://pypi.org/project/pathos/) 0.2.8 needed for multicore processing (even if not used; install using pip)
6. [zeus-mcmc](https://github.com/minaskar/zeus) 2.3.0 needed for MCMC sampling (optional)



## Main run script

In the `scripts/` folder of the repository, copy `sample_run_script.py` to a `.py` file of any name (this will not be tracked by git).  The run file contains the following simple code:

```python
import sys
sys.path.append('../src')
from load_save import load_data
from main import analyze_FRET_data

DATA_DIR = r'../example'
EXP_DIR = r'../example/gauss_noise/sinusoids'
	
def run():

	data = load_data(DATA_DIR, 'data_sinusoids')
	a = analyze_FRET_data(data, EXP_DIR)
	a.run()

if __name__ == '__main__':
	run()
```



For a given estimation, the general structure will remain as such. In particular, for the multiprocessing aspects of the code to function properly, the script guard  `if __name == '__main':run()` must be present at the end of the script, and all code to be executed must be in the function `run()` . 

For most estimations, the user will only need to change `DATA_DIR` to the directory where  the measured input data (see **Input data** section below) is stored and `EXP_DIR` to the directory where the estimation specifications and models (see **Analysis parameters** and **Model functions** sections below) and output data is to be stored.



##  Input data

Input (measured) data should be a dictionary saved in `.pkl` format. The dictionary must have the following keys:

1. 'IDD': representing the quantity I<sub>DD</sub>,  which is the time series of the fluorescence intensity from donor emission due to donor excitation. This quantity is an array of shape either (N<sub>T,DD</sub>) if there is 1 measured dataset, or (N<sub>data</sub>, N<sub>T,DD</sub>) if there are multiple datasets, where N<sub>T,DD</sub> is  the number of measured timepoints and N<sub>data</sub> is the number of datasets. 
2. 'IDA': representing the quantity I<sub>DA</sub>,  which is the time series of the fluorescence intensity from acceptor emission due to donor excitation. This quantity is an also array of shape either (N<sub>T,DD</sub>) or (N<sub>data</sub>, N<sub>T,DD</sub>). 
3. 'IAA':  representing the quantity I<sub>AA</sub>,  which is the time series of the fluorescence intensity from acceptor emission due to acceptor excitation. This quantity is an array of shape either (N<sub>T,AA</sub>) or (N<sub>data</sub>, N<sub>T,AA</sub>). Note that N<sub>T,AA</sub> does is not necessarily the same as N<sub>T,DD</sub>. 

4. 'tDD': the times at which I<sub>DD</sub> and N<sub>DA</sub> are measured; a vector of length N<sub>T,DD</sub>. 
5. 'tAA': the times at which I<sub>AA</sub> is measured; a vector of length N<sub>T,AA</sub>.  
6. 'IDD_noise_sd': The standard deviation of the measurement noise of  I<sub>DD</sub> at each timepoint;  a vector of length N<sub>T,DD</sub>.  
7. 'IDA_noise_sd': The standard deviation of the measurement noise of  I<sub>DA</sub> at each timepoint;  a vector of length N<sub>T,DD</sub>.  
8. 'IAA_noise_sd': The standard deviation of the measurement noise of  I<sub>AA</sub> at each timepoint;  a vector of length N<sub>T,AA</sub>.  
9. 'crstlk', a dictionary which contains keys 'a' and 'd', whose values (each floats) represent cross-excitation and bleedthrough, respectively, as well as an 'G', representing the optical parameter. 

The user can also provide an optional 'E' key which is an array of the same shape as 'IDD', and contains the ground-truth FRET index, to be compared against the estimated FRET index. 



## Analysis parameters

These are the hyperparameters of the inference procedure, and are all defined within the `anl_params` class in the `define_analysis_params.py` module. The hyperparameters of the inference are all set as attributes of the `anl_params` class, for example:

```python
class anl_params():

    def __init__(a):
	
        a.process_noise = 'Gaussian'
        a.state_est_all_cores = True
	
    if a.process_noise == 'Non-Gaussian':
        a.Q_tilde_prior_modes = [100, 1]
        a.Q_tilde_prior_FC = [30, 10]
        a.state_est_method = 'integrate'
        a.chi_interval_width = 8
        a.chi_num_of_subintervals = 400
    else:
        a.sigma_chi_prior_FC = 30
        a.sigma_chi_prior_mode = 100
		
    # list further attributes below...
```

This module is placed in the `EXP_DIR` folder defined above in the main run script.  See the example in the `example/gauss_noise/steps` directory for the general structure. 

The full list of attributes that one can set are:

1. `process_noise`: string, setting the statistics of the process noise; 'Gaussian' or 'Non-Gaussian'. In general, for situations in which the FRET index has small, nearly continuous changes in time, Gaussian noise should be chosen. 

   **If Gaussian process noise is used**, the process noise is parameterized only by the standard deviation SD. This SD is unknown, and is inferred during the estimation procedure using Bayesian inference with a prior, which is assumed lognormal. The prior then has its own hyperparameters -- the mode and fold change of the lognormal -- which must be set here. Here, `sigma_chi_prior_mode` is a float setting the mode and `sigma_chi_prior_FC` is a positive float setting fold change  (or geometric standard deviation, essentially the 'spread' in log space) of the lognormal prior of the SD.

   ```python
   a.sigma_chi_prior_FC = 30
   a.sigma_chi_prior_mode = 100
   ```



   **If non-Gaussian process noise is used**, then the following procedures should be taken:

   A. The distribution of the noise, which we call *Q_tilde*, is defined as a subclass in the model class (see the **Model functions** section for how to do this). 

   B. The parameters of *Q_tilde* must themselves be estimated, and each is assumed lognormal. For example, if *Q_tilde* is a mean-zero generalized Student's t-distribution, then it is parameterized by hyperparameters *sigma* and *nu*. We must define the lognormals for each of these, by their mode and fold-change:

   ```python
   a.Q_tilde_prior_modes = [100, 1]
   a.Q_tilde_prior_FC = [30, 10]
   ```

   Thus, *sigma* is chosen from a lognormal with mode 100 and fold-change 30, while nu is chosen from a lognormal with mode 1 and fold change 10. The ordering of sigma and nu is defined where *Q_tilde* is defined in **Model functions** (see below).

   C. Non-gaussian process noise requires either using a particle filter (sequential Monte Carlo) or direct integration for the Bayesian integrals, specifically, integrating over the variable *chi*. The preferred method is set with`state_est_method`, which can be set to 'SMC' or 'integrate'. If the evaluation method is 'integrate', then one must set:

   ```python
   a.state_est_method = 'integrate'
   a.chi_interval_width = 8
   a.chi_num_of_subintervals = 400
   ```

   This chooses an integration range whose width is 8x the width of 95% of the chi values, and which is partitioned into 400 intervals to approximate the Reimann integral. Typically, 8 is a conservative width and one can choose a smaller multiplier (say 2 or 4) without much loss of precision. If the evaluation method is 'SMC', one must set the number of particles for the particle filter using `num_SMC_samples` -- typically 300 is fine.  

2. `state_est_all_cores`: boolean. If True, use all available CPU cores in the state estimation, otherwise only run on a single core.

3. `do_MCMC`: boolean. If True, use MCMC sampling to get the posterior distributions of the dynamical states; otherwise they are estimated using the Laplace approximation. If True, set the following attributes:

   A. `MCMC_all_cores`: boolean. If True, use all available CPU cores.

   B. `MCMC_d_init`: float, setting the size of the ball, in logspace, over which the workers are initialized. ~0.1 seems appropriate.

   C. `MCMC_num_workers`: integer. Number of parallel walkers.

   D. `MCMC_num_samples`: integer. Number of samples per walker.

   E. `MCMC_burnin` : integer. Burn-in length.

4. `curve_fit_options`: dictionary, which sets the parameters of the optimization for the initial curve fitting to get an initial estimate on the parameters. Its keys are 'ftol', 'maxfun', and 'maxiter', which are the tolerate, maximum number of function evaluations, and maximum number of iterations for the optimization. Reasonable values are <1e-6, >1e4, and >1e3, respectively. 

5. `P0_x1`: float, setting the variance of the Gaussian prior on the constant dynamical state. It should be very small ~1e-10.

7. `m0_chi` and `P0_chi` are the mean and variance of the Gaussian prior on the *chi* dynamical state. One should generally chose a rather uniformed prior, suggested values are  `m0_chi` = 0 and `P0_chi` = 1e6. 



## Model functions

This module contains the `model` class used to define the functions pertaining to the bleaching curve. There is a good deal of flexibility here in that these functions can contain any functional forms and an arbitrary number of unknown parameters. There are 4 subclasses in `model`.  The example in `example/gauss_noise_estimated_G` should be followed closely, in the following manner:

1. subclass `f_D()` defines the bleaching curve for the donor channel.

   ```python
   class f_D():
   
       def __init__(self):
   
           self.p_name = ['tau_D1', 'tau_D2', 'delta_D']
           self.p_init = [100, 100, 0.5]
           self.p_lo_bnd = [0, 0, 0]
           self.p_hi_bnd = [np.inf, np.inf, 1]
   
           def func(self, t, params):
   
               return params[2]*np.exp(-t/params[0]) + (1 - params[2])*np.exp(-t/params[1])
   ```

    This curve is parameterized by *p* parameters -- each of the lists in the constructor should be *p*-element lists (in the example, *p* = 3). The `p_init` is the values of the initial guesses for these, while `p_lo_bnd` and `p_hi_bnd` are the hard bounds for each parameter. This function should have 1 method, `func(self, t, params)`, which returns the bleaching function as a function of the time *t* and the parameters *params*, which is the *p*-element list as in the constructor. Obviously, the ordering of the 4 lists defined in `__init__` must be consistent. 

2. subclass `f_A()` is analogous for the bleaching curve for the acceptor channel.

3. subclass `Q_tilde()` defines the statistics of the process noise, *if the process noise is non-Gaussian* (see **Analysis parameters** above). 

   ```python
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
   ```

   

   The general structure of this class can be inferred from the example. Here, there are two hyperparameters defining the *Q_tilde* distribution (in this case, the student's t-distribution), *sigma_chi* and *nu*. The class has 2 methods: `pdf` and `sample`. The method `pdf(self, params)` returns the pdf of the process noise with hyperparameters set to *params*. Thus, `pdf(self, params)`  *returns a function* defining the pdf of the data. In the example, we used the auxiliary method `_f(self, x)` to define this pdf. The method `sample(self, params, size=1)` returns *N = size* independence samples from the process noise pdf (in the example, we use a built-in function).

4. subclass `prior_dict(object)`. The example should be closely followed, and only a few portions should be modified.  This object defines the pdfs for the priors on *all* *p* unknown parameters. The class has one method, `func(self, anl_params)`, which returns a dictionary with *p* entries. 

   ```python
   class prior_dict(object):
       def func(self, anl_params):
   
           pdfs_dict = dict()
   
           # DT, AT
           pdfs_dict['DT'] = lognorm_pdf(sigma=np.log(2)).pdf
           pdfs_dict['AT'] = lognorm_pdf(sigma=np.log(1.1)).pdf
   
           # Add further pdfs in their respective keys below ...
   ```

   `pdfs_dict` is a dictionary, with one key for each parameter being estimated.  The necessary keys are: DT, AT, all of the parameters of f_D and f_A (the entries of `f_D.p_name` and `f_A.p_name`), and either

   ​	 i) sigma_chi (if model noise is Gaussian), or 

   ​	ii) the entries of`Q_tilde.p_name` if the noise is non-Gaussian.

   *The value of each entry of `pdfs_dict` is a single-argument function returning the pdf whose mode is the argument*. In the first part of the estimation, the mode for all of these priors will be determined by simple curve fitting. This will then define the prior distribution once and for all (all hyperparameters are now fixed). There are useful helper functions in `src/utils.py` for common distributions -- normal, lognormal, and standard uniform. 

