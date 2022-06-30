function neg_llh_IAA = compute_neg_llh_IAA(p,data,inputs)
% NEG_LLH_IAA computes nagative log likelihood function for IAA
%
% Inputs:
% p: A vector containing model parameters
% data: A structure containing 3 background-subtructed fluorescent intensity timeseries and their time
% inputs: A structure containing information about the model, priors etc.

% Outputs:
% neg_llh_IAA: negative log likelihood

%% Expanding inputs
mdl_fun = inputs.mdl_fun;
prior0 = inputs.mdl_fun.prior0;

%% Free parameters
% list of fields
fields = fieldnames(prior0);
% getting total concentrations of the acceptor from p
AT = p(ismember(fields,'AT'));
% parameters for f_A_fun
theta_A = p(ismember(fields,mdl_fun.f_A.p_name));
%defining bleaching function (fluorescent intensities)
f_A_fun = @(t) mdl_fun.f_A.fun(theta_A,t);

%% Measurement noise parameters
% IAA data
IAA = data.IAA;
tAA = data.tAA;
IAA_noise_sd = data.IAA_noise_sd;%Std of measurement noise
IAA_noise_var = IAA_noise_sd.^2;%Var of measuremement noise

% computing negative log likelihood (neglecting constant)
neg_llh_IAA = sum(((IAA - AT.*f_A_fun(tAA)).^2)./(2*IAA_noise_var));
end

