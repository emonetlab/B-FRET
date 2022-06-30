function mdl_fun = define_model_functions(anl_params)
% DEFINE_MODEL_FUNCTIONS define the functional forms of the model and priors. 

%% Defining the functions for the intact fractions of the donor (f_D) and acceptor (f_A)
%f_D
mdl_fun.f_D.fun = @(params,t) params(3)*exp(-t/params(1)) + (1-params(3))*exp(-t/params(2));
mdl_fun.f_D.p_name = {'tau_D1','tau_D2','delta_D'};
mdl_fun.f_D.p_ini = [1000 100 0.5]; % initial condition for least-square fitting
mdl_fun.f_D.p_lb = [0 0 0]; % lower bound of the parameters
mdl_fun.f_D.p_ub = [inf inf 1]; % upper bound 

%f_A
mdl_fun.f_A.fun = @(params,t) params(3)*exp(-t/params(1)) + (1-params(3))*exp(-t/params(2));
mdl_fun.f_A.p_name = {'tau_A1','tau_A2','delta_A'};
mdl_fun.f_A.p_ini = [1000 100 0.5];
mdl_fun.f_A.p_lb = [0 0 0]; 
mdl_fun.f_A.p_ub = [inf inf 1];  

%% Define the process noise for Non-Gaussina case
if strcmp(anl_params.process_noise,'Non-Gaussian')
    %nu > 0: nu = 1 gives Caucy, nu >> 5 gives Gaussian
    mdl_fun.Q_tilde.fun = ...
        @(params,x) (1/params(1))*...
        (gamma(params(2)/2 + 0.5)/(sqrt(pi*params(2))*gamma(params(2)/2))).*(1 + x.^2./(params(2)*params(1)^2)).^(-params(2)/2 -0.5) ;%The student t distributions
    mdl_fun.Q_tilde.p_name = {'sigma_chi','nu'};
end

%% Define the functional forms of the priors 
% The exact functions are dependent on the inital estimation of the priors
% (p_ini)

FC = 2;% Fold change that corresponds to the std of log(x)
sigma = log(FC);
mu = @(x) log(x) + sigma^2; %mu gives the log(median) of the log-normal distribution
mdl_fun.prior0.tau_D1 = @(x,p_ini) lognpdf(x, mu(p_ini),sigma);
mdl_fun.prior0.tau_D2 = @(x,p_ini) lognpdf(x, mu(p_ini),sigma);
mdl_fun.prior0.DT = @(x,p_ini) lognpdf(x, mu(p_ini),sigma);

FC = 1.1;sigma = log(FC);mu = @(x) log(x) + sigma^2;
mdl_fun.prior0.tau_A1 = @(x,p_ini) lognpdf(x, mu(p_ini),sigma);
mdl_fun.prior0.tau_A2 = @(x,p_ini) lognpdf(x, mu(p_ini),sigma);
mdl_fun.prior0.AT = @(x,p_ini) lognpdf(x, mu(p_ini),sigma);

mdl_fun.prior0.delta_D = @(x,p_ini) unifpdf(x,0,1);
mdl_fun.prior0.delta_A = @(x,p_ini) unifpdf(x,0,1);

% prior for sigma_chi
FC = anl_params.sigma_chi_FC;sigma = log(FC);mu = @(x) log(x) + sigma^2;
mdl_fun.prior0.sigma_chi = @(x,p_ini) lognpdf(x, mu(anl_params.sigma_chi_ini),sigma);  
%There's no p_ini dependence, but kept p_ini as an argument anyway.

if ~strcmp(anl_params.process_noise,'Gaussian')% for Non-Gaussian process noise 
    FC = anl_params.nu_FC;sigma = log(FC);mu = @(x) log(x) + sigma^2;
    mdl_fun.prior0.nu = @(x,p_ini) lognpdf(x, mu(anl_params.nu_ini),sigma); 
end


end

