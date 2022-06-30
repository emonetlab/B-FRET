function SSM = define_state_space_model(data,crstlk,inputs)
% DEFINE_STATE_SPACE_MODEL defnes the state space model given the data including 
% their measurement noise levels and cross-talk coefficients (crstlk);
%
% Input variables:
% data: A structure containing 3 background-subtructed fluorescent intensity timeseries, their measurement noise and their time
% crstlk: A structure containing cross-talk coefficients (a, d, and G);
% inputs: A structure containing analysis parameters and model functions etc. 

% Output variable:
% SSM: A structure that defines the state-space model

%% Expanding inputs
mdl_fun = inputs.mdl_fun;
%% Defining state-space model as functions of parameters, p
% Q and H are functions of the parameters p
SSM.Q_fun = @(p) compose_Q(p,data,mdl_fun);%Q_fun is used only for the case of Gaussian process noise
SSM.H_fun = @(p) compose_H(p,data,crstlk,mdl_fun);

% R and y are determined by data
SSM.R = compose_R(data);
SSM.y = compose_y(data);


end

