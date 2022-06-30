function anl_params = define_analysis_params()
% DEFINE_ANALYSIS_PARAMS defines analysis-related parameters. 
% Edit and put this file in the 'root_dir' folder in which analysis results are saved. 

% Process noise 
anl_params.process_noise = 'Gaussian'; %'Gaussian' or 'Non-Gaussian'
if strcmp(anl_params.process_noise,'Non-Gaussian')% 
    anl_params.chi_num_of_subintervals = 400;%The interval is divided into subintervals
    anl_params.chi_interval_width = 8; %Width of the integral interval w.r.t chi relative to the estimated range of chi (used only in define_chi_interval.m).   
    %The degrees of freedome for the Student's t-distribution
    anl_params.nu_FC = 10;%Width of the prior in fold-change ; log(FC) = sigma gives a parameter of the LogNormal distribuiton
    anl_params.nu_ini = 1;
end

%sigma_chi prior (For both Gaussian and Non-Gaussian process noise)
anl_params.sigma_chi_FC = 30;%Width of the prior in fold-change ; log(FC) = sigma gives a parameter of the LogNormal distribuiton
anl_params.sigma_chi_ini = 100;%Initial condition for the optimizaito process and the mode of the prior distribution


% Parameters for the Laplace's method
anl_params.Laplace_nsamples = 1000;

% MCMC sampling parameters
anl_params.do_MCMC = 0;
anl_params.num_of_workers = 4; %# of core CPU for parallel computing. 
anl_params.nsamples = 2000; %# of MCMC sampling 
anl_params.thin_n = 14;% %thin parameter for slice sampling
anl_params.burnin = 300;


%% Fixed parameters 


% options for the curve fitting when determining measurement noise and
% initial rough estimation of the model parameters
anl_params.options1 = optimoptions('lsqcurvefit','FunctionTolerance',1e-6,...
    'MaxFunctionEvaluations',1e5,'MaxIterations',4000,...
    'UseParallel',true);

% options for the optimization function for MAP estimate
anl_params.options2 = optimoptions('fminunc','MaxFunctionEvaluations',1E6,...
    'MaxIteration',4000,'StepTolerance',1e-10,'UseParallel',true,...
    'OptimalityTolerance',1e-8);

%For the initializattion of the covariance matirix P_0(\theta)
anl_params.epsilon = 1E-10; 

% Parameters for the prior distribution of the inital state x0
anl_params.sigma_chi_0 = 1E6;% Put a sufficiently large number for the variance (puttting too big a value instabilize the determinant)
anl_params.chi_0 = 0; 


end

