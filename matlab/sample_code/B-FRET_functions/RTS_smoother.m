function [ms, Ps] = RTS_smoother(m_pred, P_pred, m, P,ts_length)
%RTS_SMOOTHER computes RTS smoother

%Input variables
%m_pred: A cell that stores the mean of predictive distributions (Output of Kalman filter)
%P_pred: A cell that stores the var.-cov. matrix of predictive distribution (Output of Kalman filter)
%m: A cell that stores the mean of filtered/posterior distributions(Output of Kalman filter)
%P: A cell that stores the var.-cov. matrix of filtered/posterior distribution (Output of Kalman filter)
%
%Output variables
%ms: A cell that stores the mean of smoothed distributions
%Ps: A cell that stores the var.-cov. matrix of smoothed distributions

%% 
% initialization
ms_kp1 = m{end};%m^s_{k+1} = m_T
Ps_kp1 = P{end};%P^s_{k+1} = P_T

% cells to store the results
ms = cell(1,ts_length);
Ps = cell(1,ts_length);

% for last time point
ms{end} = m{end};
Ps{end} = P{end};

for l = 1:ts_length-1
    
    % computing ms_k and Ps_k
    k = ts_length - l;
    G_k = P{k}/P_pred{k+1};
    ms_k = m{k} + G_k*(ms_kp1 - m_pred{k+1});
    Ps_k = P{k} + G_k*(Ps_kp1 - P_pred{k+1})*(G_k.');
    
    % saving the data
    ms{k} = ms_k;
    Ps{k} = Ps_k;
    
    % for the next cycle
    ms_kp1 = ms_k;%m^s_{k+1}
    Ps_kp1 = Ps_k;%P^s_{k+1}
    
end

end