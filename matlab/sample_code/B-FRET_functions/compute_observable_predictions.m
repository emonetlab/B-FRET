function obs_pred = compute_observable_predictions(data,inputs,param_posterior,state_posterior)
% COMPUTE_OBSERVABLE_PREDICTIONS compute the predctions of the observables.

% Input variables:
%data: A structure containing 3 background-subtructed fluorescent intensity timeseries and their time
%inputs: A structure containing information about the model, priors, etc
%SSM: A structure that defines the state-space model
%param_posterior: A structure containing the posterior distribuiton of the parameters
%state_posterior: A structure containing the posterior distribuiton of the states

% Output variables:
% obs_pred.IDD: Prediction of IDD
% obs_pred.IDA: Predcition of IDA
% obs_pred.IAA: Prediction of IAA
%% Expanding input variables
mdl_fun = inputs.mdl_fun;
SSM = inputs.SSM;
prior = inputs.prior;
anl_params = inputs.anl_params;
do_MCMC = anl_params.do_MCMC;
tDD = data.tDD;
%% 

% list of fields
fields = fieldnames(prior);

% Obtaining estimation of most probable parameters 
if do_MCMC ~= 1
    p = param_posterior.p_MAP_vec;
else
    p = median(param_posterior.samples);
end

%%  prediction of IAA
theta_A = p(ismember(fields,mdl_fun.f_A.p_name));
f_A_fun = @(t) mdl_fun.f_A.fun(theta_A,t);
AT = p(ismember(fields,'AT'));
IAA_pred = AT.*f_A_fun(tDD);

%%  prediction of IDD and IDA

% Composing the 'most probable' state vecotors
chi_sample = state_posterior.chi;
x = cell(1,length(tDD));
for i = 1:length(tDD)
    x{i} = [1; median(chi_sample(:,i))];
end

% Computing the prediciton of y vector
y_pred = zeros(2,length(tDD));
H = SSM.H_fun(p);
for i = 1:length(tDD)
    y_pred(:,i) = H{i}*x{i};
end
IDD_pred = y_pred(1,:);
IDA_pred = y_pred(2,:);

%% Organizing the output
obs_pred.IDD = IDD_pred;
obs_pred.IDA = IDA_pred;
obs_pred.IAA = IAA_pred;

