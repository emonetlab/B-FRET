function R = compose_R(data)
% COMPOSE_R computes the the values of the matrix R that specifies the measurement-noise variance of IDD and IDA
%
% Inputs:
% data: A structure containing 3 background-subtructed fluorescent intensity
%
% Outputs: 
% R: A cell array that stores the matrix R at each time point.


%%

L = length(data.IDD);% length of the timeseries
R = cell(1,L);

% standard deviation of measurement noise
IDD_noise_sd = data.IDD_noise_sd;
IDA_noise_sd = data.IDA_noise_sd;

% composing R
for k = 1:L
    R{k} = [IDD_noise_sd(k)^2, 0; 0, IDA_noise_sd(k)^2];
end
end

