function param_posterior = sample_from_param_posterior(data,inputs)
% SAMPLE_PARAM_POSTERIOR obtains the posterior distribution of the model
% parameters, p(\theta|data).

%Input variables:
%data: A structure containing 3 background-subtructed fluorescent intensity timeseries and their time
%crstlk: A structure containing the values of crosstalk coefficients (a, d, and G)
%inputs: A structure containing information about the model, priors, etc

%Output variables:
%param_posterior.samples: Samples from the approximated posterior distribution of the parameters.

%For only Laplace approximation (i.e., anl_params.do_MCMC ~= 1)
%param_posterior.p_MAP_vec: MAP of the parameters
%param_posterior.mvnsigma: Variance-covariance matrix of the distribution of log-parameters.
%param_posterior.log_mu: Mode of the distribution of log-parameters
%param_posterior.nearestSPD: This is 1, if the estimated variance-covariance matrix of the Gaussian distribution that approximates the posterior distribution of model parameters are not positive semi definite, and it is replaced by a nearest symmetric positive definite.
%%
%% parameters and functions
anl_params = inputs.anl_params;
do_MCMC = anl_params.do_MCMC;
p_ini_vec = inputs.p_ini.p_ini_vec;

if do_MCMC ~= 1 %Laplace approximation 
    %%
    options2 = anl_params.options2;
    %%  Compute MAP (and Hessian matrix) of the parameters
    % Defining the negative log posterior as a function of log of the parameters
    energy_fun = @(log_p_vec) neg_log_posterior(log_p_vec,data,inputs);
    
    
    % Finding the minimum of the energy function and computing the Hessian matrix
    disp('Trying to find MAP estimate of the parameters...')
    tic
    [log_p_MAP_vec,fval,~,~,~,log_p_hessian]= fminunc(energy_fun,log(p_ini_vec),options2);
    p_MAP_vec = exp(log_p_MAP_vec);
    toc
    
    %% Computing BIC
    n_of_data_points = length(data.IDD) + length(data.IDA) + length(data.IAA);%# of data points
    BIC = length(p_MAP_vec)*log(n_of_data_points) + 2*fval;
    fprintf('BIC: %1.3f\n',BIC);
    %% Laplace's method: Sampling from a log-normal distribution that approximete the posterior distribution
    disp('Sampling from the approximated posterior distributions of the parameters...')
    param_posterior = Laplace_sampling(log_p_MAP_vec,log_p_hessian,inputs);
    %%  storing the results
    param_posterior.p_MAP_vec = p_MAP_vec;%MAP of the prameters
    param_posterior.BIC = BIC;
    %% Fully Bayesian analysis based on MCMC sampling
elseif do_MCMC == 1
    % parameters
    nsamples = anl_params.nsamples;
    thin_n = anl_params.thin_n;
    burnin = anl_params.burnin;
    num_of_workers = anl_params.num_of_workers;
    
    % define log posterior function
    log_posterior = @(log_p_vec) -neg_log_posterior(log_p_vec,data,inputs);
    
    % MCMC sampling
    p_sample_cell = cell(1,4);
    parfor i = 1:num_of_workers
        log_p_sample_MCMC = slicesample(log(p_ini_vec),nsamples,'logpdf',log_posterior,'thin',thin_n,'burnin',burnin);
        p_sample_cell{i} = exp(log_p_sample_MCMC);
    end
    samples = [];
    for i = 1:num_of_workers
        samples = [samples; p_sample_cell{i}];
    end
    
    param_posterior.samples = samples;
end


end

