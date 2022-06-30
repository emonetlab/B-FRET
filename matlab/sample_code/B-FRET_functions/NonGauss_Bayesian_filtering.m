function [distr, Q_tilde_mat]= NonGauss_Bayesian_filtering(ini,p,y,H,R,inputs,data)
%NONGAUSS_BAYESIAN_FILTERING computes the predictive and filtering distribuitons
% as well as the likelihood funciton of chi, given model parameters
% (theta), the data (D) and the model (M) WITHOUT assming Gaussian process noise. 
% The integrals involved in the calcularion are approximted by representing
% the integrands by step functions (see Kitagawa, 2010, Chap. 14)

% Input variables
% ini: A structure that contains some initialization variables
% p: A vector containing the model parameters
% y,H,R: Cells that define the state-space model
% inputs: A structure containing information about the model, priors, etc
% data: A structure containing 3 background-subtructed fluorescent intensity timeseries and their time
%
% Output variables
%distr.p_tilde_mat: N_D (# of time points) by d (# of subinterval of chi space) matrix containign the predictive distributions.
%Note the k-th raw in this matrix is p(chi_k|y_{1:k-1},theta). 
%distr.f_tilde_mat: N_D by d matrix containign the filtering distributions.
%Note the k-th raw in this matrix is p(chi_k|y_{1:k},theta).
%distr.LH_of_chi_mat: N_D by d matrix containign the likelihood function of chi
%Q_tilde_mat: A matrix consisting of Q_tilde used for efficient computation
%% expanding ini that contains some necesary variables
d = ini.d;
dq = ini.dq;
process_noise_param_ind = ini.process_noise_param_ind;
x = ini.x;
xq = ini.xq;
delta_x = ini.delta_x;
f_tilde0 = ini.f_tilde;
N_D = ini.N_D;


%% Preparing Q_tilde matrix for efficient computation of the predictive distribution
process_noise_params = p(logical(process_noise_param_ind));
Q_tilde = inputs.mdl_fun.Q_tilde.fun(process_noise_params,xq);%Discritized process noise
Q_tilde_mat = zeros(d);%Memory allocation
center_ind = (dq - 1)/2 + 1;%Centeral index for Q_tilde;
for ii = 1:d
    Q_tilde_mat(ii,:) = Q_tilde(center_ind+1-ii:center_ind +d-ii);
end

%%  Recursive computation of predictive (p) and filtering distributions
p_tilde_mat = zeros(N_D,d);%Creating a matrix to store predictive distributions;
f_tilde_mat = zeros(N_D,d);%Creating a matrix to store filtering distributions;
LH_of_chi_mat = zeros(N_D,d);%Creating a matrix to store likelihood function of chi;

tDD = data.tDD;
delta_tDD = median(tDD(2:end)-tDD(1:end-1));%Typical sampling interval
tDD_increase = 0;

%FH1 = figure;hold on;
%FH2 = figure;hold on;
for k = 1:N_D
    
    % Prediction
    if k == 1 || tDD_increase > 2*delta_tDD %If the sampling time interval is large, do not use the information from the preceding measurement
        p_tilde = f_tilde0*Q_tilde_mat;%Constant delta_x is omitted because it's normalized out anyway.
    else
        p_tilde = f_tilde*Q_tilde_mat;%Constant delta_x is omitted because it's normalized out anyway.
    end
    Z = delta_x*sum(p_tilde);%Normalization constnat
    p_tilde = p_tilde./Z;
    p_tilde_mat(k,:) = p_tilde;
    
    % Updating filtering ditribution
    LH_of_chi = mvnpdf((H{k}*[ones(1,d);x])',y{k}',R{k})';
    %Below is equivalent to the above, but slower
%     for ii = 1:d
%         LH_of_chi(ii) = mvnpdf(y{k},H{k}*[1;x(ii)],R{k});
%     end
    LH_of_chi_mat(k,:) = LH_of_chi;
    f_tilde = LH_of_chi.*p_tilde;
    Z = delta_x*sum(f_tilde);%Normalization constant
    f_tilde = f_tilde./Z;%Normalization
    f_tilde_mat(k,:) = f_tilde;
    tDD_increase = tDD(min(k+1,N_D)) - tDD(k);%For data sampled inhomogeneously in time
    %figure(FH1);plot(x,p_tilde);plot(x,f_tilde);
    %figure(FH2);plot(x,LH_of_chi);
    %pause
end
%% Organizing the output variables
distr.p_tilde_mat = p_tilde_mat;
distr.f_tilde_mat = f_tilde_mat;
distr.LH_of_chi_mat = LH_of_chi_mat;

end

