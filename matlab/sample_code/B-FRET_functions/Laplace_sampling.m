function param_posterior = Laplace_sampling(log_mu,log_p_hessian,inputs)
% LAPLACE_SAMPLING samples from a log-normal distribution that approximate the 
% posterior distribution of the model parameters. First, log of parameters 
% are sampled form a multi-dimenstional Gaussian distribution with LOG_MU and 
% inverse of LOG_P_HESSIAN being the parameters of the Gaussina distribution. 
% Then, the exponential of the log of parameters is computed to obtain the parameters. 

%Input variables:
%log_mu: A vector containing the log of the MAP of the parameters
%log_p_hessian: Hessian matrix at the minimum of the energy function
%inputs: A structure containing information about the model, priors, etc

%Output variables:
%param_posterior.samples: Samples from the approximated posterior distribution of the parameters.
%param_posterior.mvnsigma: Variance-covariance matrix of the distribution of log-parameters.
%param_posterior.log_mu: Mode of the distribution of log-parameters

%% Analysis parameters
Laplace_nsamples = inputs.anl_params.Laplace_nsamples;

%% Obtaining an variance-covariance matrix 
inv_hessian = inv(log_p_hessian);% Computing the inverse of the Hessian matrix
mvnsigma = (inv_hessian + inv_hessian.')/2;% making sure the matrix is symmetric
e = eig(mvnsigma);% Eigen values of the variance-covariance matrix
if ~all(e >= 0) % if the matrix is NOT positive semidefinite
    mvnsigma = nearestSPD(mvnsigma);% Finding nearest symmetric positive definite (SPD)
    warning('The initial variance-covariance matrix was not positive semi definite; ''nearestSPD'' was used. ');
    param_posterior.nearestSPD = 1;
else
    param_posterior.nearestSPD = 0;
end

%% Sampling from the Normal distribution (log of the parameters)
log_p_sample_Laplace = mvnrnd(log_mu,mvnsigma,Laplace_nsamples);

%% Outputs
param_posterior.samples = exp(log_p_sample_Laplace);%samples of the parameters
param_posterior.mvnsigma = mvnsigma;%Variance-covariance matrix of the distribution of log-parameters
param_posterior.log_mu = log_mu;%Mode of the distribution of log-parameters

end

