function [RTS_state] = RTS_smoothed_state(p,data,priors,inputs,anl_params)
%RTS_SMOOTHED_STATE computes the smoothed states

% input variables
%p: A vector containing the values of the parameters
%data: A structure containing the intensity data.
%prior: A structure containing the priors
%inputs: A structure containing information about the model, priors, etc
%anl_params: A structure containing analysis-related parameters
%
% ouput variables
%RTS_state.E.MAP: A vector of MAP estimate of E=chi/DT
%RTS_state.E.std: A vector of std of the posterior distribution of E
%RTS_state.chi.MAP: Similar to E.MAP
%RTS_state.chi.std: Similar ot E.std
%RTS_state.ms: A cell containing the average of the posterior distribution of the state
%RTS_state.Ps: A cell containing the var.-cov. matrices of the posterior distributiojn of the state.

%%


%% Computing model/data dependent terms, Q, R, H and y
Q = inputs.SSM.Q_fun(p);
H = inputs.SSM.H_fun(p);
R = inputs.SSM.R;
y = inputs.SSM.y;

%% Computing m_pred_k (m_k^{-}), P_pred_k (P_k^{-1}), m_k, and P_k by Kalman filter

% initialization
fields = fieldnames(priors);
chi_0 = anl_params.chi_0;
sigma_chi_0 = anl_params.sigma_chi_0;
m_0 = [1; chi_0];%m_{k-1} = m_0
P_0 = [anl_params.epsilon 0; 0 sigma_chi_0^2];%P_{k-1} = P_0
ts_length = length(data.IDD);

% kalman filter
[m_pred, P_pred, m, P, ~, ~] = kalman_filter(Q,R,H,y,ts_length,m_0,P_0,data);

%% Computing the parameters for the smoothing distribution ms_k and Ps_k by RTS smoother

%RTS smoother
[ms, Ps] = RTS_smoother(m_pred, P_pred, m, P,ts_length);

DT = p(ismember(fields,'DT'));

% saving data
RTS_state.ms = ms;
RTS_state.Ps = Ps;
RTS_state.E.MAP = cellfun(@(x) x(2),ms)/DT;
RTS_state.E.std = cellfun(@(x) sqrt(x(4)),Ps)/DT;
RTS_state.chi.MAP = cellfun(@(x) x(2),ms);
RTS_state.chi.std = cellfun(@(x) sqrt(x(4)),Ps);

end