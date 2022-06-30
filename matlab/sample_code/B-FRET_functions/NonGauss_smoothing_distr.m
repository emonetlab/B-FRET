function distr = NonGauss_smoothing_distr(ini,distr,Q_tilde_mat)
%NONGAUSS_SMOOTHING_DISTR computes smoothing distrubtion , given model parameters
% (theta), the data (D) and the model (M) without assuming Gaussina process noise. 
% The integral involved in the calcularion is approximted by representing
% the integrands by step functions (see Kitagawa, 2010, Chap. 14).

% Input variables
% ini: A structure that contains some initialization variables
% distr: A structure containing predictive and filtering distributions at each time point (Outputs of NonGauss_Bayesian_filtering)
% Q_tilde_mat: A matrix for effcient computation (An output of NonGauss_Bayesian_filtering)

% Output variables
% distr.s_tilde_mat: N_D (# of time points) by d (# of subinterval of chi space) matrix containign the smoothing distributions
% distr.s_tilde_CDF_mat: CDF version of s_tilde_mat
%% expanding ini that contains some necesary variables
d = ini.d;
delta_x = ini.delta_x;
N_D = ini.N_D;

%% predictive and filtering distributions
p_tilde_mat = distr.p_tilde_mat;
f_tilde_mat = distr.f_tilde_mat;

%%
s_tilde_mat = zeros(N_D,d);%Creating a matrix to store smoothed distributions;
s_tilde = f_tilde_mat(N_D,:);%Initialization of s_tilde
s_tilde_mat(N_D,:) = s_tilde;%Smoothed distribution at the final time point is equal to the filtered distribution
for i = N_D-1:-1:1
    
    %Updating filtering and predictive distributions
    f_tilde = f_tilde_mat(i,:);%p(chi_i|y_{1:i},theta)
    p_tilde = p_tilde_mat(i+1,:);%p(chi_{i+1}|y_{1:i},theta)
    S_tilde = s_tilde./p_tilde;%Here, s_tilde is p(chi_{i+1}|y_{1:N_D},theta)
    S_tilde(isnan(S_tilde)|isinf(S_tilde)) = 0;% if 0/0 (= NaN) or x/0 (= Inf) happnes, replace by zero.
    
    %Updating smoothed distribution (delta_x is omitted because it's normalized out anyway)
    s_tilde = f_tilde.*(S_tilde*Q_tilde_mat);
    Z = delta_x*sum(s_tilde);%Normalization constant
    s_tilde = s_tilde./Z;%Normalization
    s_tilde_mat(i,:) = s_tilde;
end

%% Organizing the outputs
distr.s_tilde_mat = s_tilde_mat;
distr.s_tilde_CDF_mat = cumsum(s_tilde_mat.*delta_x,2);%Converting PDF to CDF;


end

