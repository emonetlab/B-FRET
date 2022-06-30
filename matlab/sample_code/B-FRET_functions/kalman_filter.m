function [m_pred, P_pred, m, P, S, v] = kalman_filter(Q,R,H,y,N_D,m_0,P_0,data)
%KALMAN_FILTER computes kalman filter

% input variables
%Q, R, H, y: Vectors and matrices defining the state-space model
%N_D: The length of the timeseries
%m_0: The mean of the initial filtering/posterior distribution
%P_0: The var.-cov. matrix of the initial filtering/posterior distribution
%data: A structure containing 3 background-subtructed fluorescent intensity timeseries and their time
%
% output variables
%m_pred: A cell that stores the mean of predictive distributions 
%P_pred: A cell that stores the var.-cov. matrix of predictive distribution 
%m: A cell that stores the mean of filtered/posterior distributions
%P: A cell that stores the var.-cov. matrix of filtered/posterior distribution 
%S: A cell that stores a by-product of kalman filter
%v: A cell that stores a by-product of kalman filter

%%
% initial parameters for the filtering/posterior distribution
m_km1 = m_0;
P_km1 = P_0;

% cells to store the results
m_pred = cell(1,N_D);
P_pred = cell(1,N_D);
m = cell(1,N_D);
P = cell(1,N_D);
S = cell(1,N_D);
v = cell(1,N_D);

tDD = data.tDD;
delta_tDD = median(tDD(2:end) - tDD(1:end-1));
tDD_increase = 0;
for k = 1:N_D
    
    % Prediction
    if k == 1 || tDD_increase > 2*delta_tDD %If the sampling time interval is large, do not use the information from the preceding measurement
        m_pred_k = m_km1;
        P_pred_k = P_km1 + Q{1};
    else
        m_pred_k = m_km1;
        P_pred_k = P_km1 + Q{k-1};
    end
    
    % Update
    v_k = y{k} - H{k}*m_pred_k;
    S_k = H{k}*P_pred_k*(H{k}.') + R{k};
    K_k = P_pred_k*(H{k}.')/S_k;
    m_k = m_pred_k + K_k*v_k;
    P_k = P_pred_k - K_k*S_k*(K_k.');
    
    % Saving the data
    m_pred{k} = m_pred_k;
    P_pred{k} = P_pred_k;
    m{k} = m_k;
    P{k} = P_k;
    S{k} = S_k;
    v{k} = v_k;
    
    % For next cycle
    m_km1 = m_k;
    P_km1 = P_k;
    tDD_increase = tDD(min(k+1,N_D)) - tDD(k);%For data sampled inhomogeneously in time
end




end

