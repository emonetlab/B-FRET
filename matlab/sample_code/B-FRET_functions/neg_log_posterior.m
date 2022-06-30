function phi = neg_log_posterior(log_p,data,inputs)
%NEG_LOG_FRET_POSTERIOR compute negative log posetrior of the parameters

%Input variables
%log_p: A vector containing log of model parameters
%data: A structure containing 3 background-subtructed fluorescent intensity timeseries and their time
%inputs: A structure containing information about the model, priors, etc

%Output variable
%phi: Negative log posterior (energy function) of the parameters (If LLH is 0) or nagative log-likelihood (if LLH is 1).

%% expanding the input variables
p = exp(log_p);
SSM = inputs.SSM;
prior = inputs.prior;
anl_params = inputs.anl_params;

%% Computing nagative log prior
neg_log_prior = sum(cellfun(@(x,y) -log(x(y)),struct2cell(prior)', num2cell(p)));
%below is equivalent, but slower
% fields = fieldnames(prior);
% p_L = length(p);%length of parameter vector
% neg_log_prior_vec = zeros(1,p_L);
% for i = 1:p_L
%     neg_log_prior_vec(i) = - log(prior.([fields{i}])(p(i)));
% end
% neg_log_prior = sum(neg_log_prior_vec);

%% Computing negative log likelihood for IAA: -log[p({IAA}|theta)]
neg_llh_IAA = compute_neg_llh_IAA(p,data,inputs);

%% Computing model/data dependent terms,R, H and y (except Q which is only used in Gaussian case)
H = SSM.H_fun(p);
R = SSM.R;
y = SSM.y;

%% Computing negative log likelihood for y=({IDD},{IDA}): -log[p(y|theta)]
% Common variables
chi_0 = anl_params.chi_0;
sigma_chi_0 = anl_params.sigma_chi_0;
N_D = length(data.IDD);

if strcmp(anl_params.process_noise,'Gaussian')%If the process noise is Gaussian
    %% Gaussian case
    % Initialization
    Q = SSM.Q_fun(p);
    m_0 = [1; chi_0]; %m_{k-1} = m_0
    P_0 = [anl_params.epsilon 0; 0 sigma_chi_0^2]; % P_{k-1} = P_0;
    
    % Computing S and v, which are dependent on predictive distributions
    [~, ~, ~, ~, S, v] = kalman_filter(Q,R,H,y,N_D,m_0,P_0,data);
    
    % Computing the negative log-likelihood
    neg_llh_y = sum(cellfun(@(S) 0.5*log(det(2*pi*S)),S) + cellfun(@(v,S) 0.5*((v.')/S)*v,v, S));
    %Below is equivalent, but slower
    %         neg_llh_y_vec = zeros(1,N_D);
    %         for k = 1:N_D
    %             neg_llh_y_vec(k) = 0.5*log(det(2*pi*S{k})) + 0.5*((v{k}.')/S{k})*v{k};
    %         end
    %         neg_llh_y = sum(neg_llh_y_vec);
    
else % If process noise is Non-Gaussian
    %% Non-Gaussian case
    %  Initialization
    fields = fieldnames(prior);
    max_x = anl_params.chi_intergral_interval(2);
    min_x = anl_params.chi_intergral_interval(1);
    ini.d = anl_params.chi_num_of_subintervals;
    if mod(ini.d,2)~=1 %if d is even
        ini.d = ini.d+1;%d is assumed to be odd
    end
    ini.x = linspace(min_x, max_x, ini.d);%Discritizing chi space
    ini.delta_x = ini.x(2) - ini.x(1);
    ini.xq = linspace(-(max_x-min_x),max_x-min_x,2*(ini.d-1)+1);%Discritizing chi for the process noise term
    ini.dq = length(ini.xq);
    f_tilde =  normpdf(ini.x,anl_params.chi_0,anl_params.sigma_chi_0);%Filtered distribuiton at time zero
    Z = ini.delta_x*sum(f_tilde);%Normalization constant
    ini.f_tilde = f_tilde./Z;%Normalization
    ini.N_D = N_D;
    %Finding indices in 'p' that stores informaiton about the process-noise parameters
    p_name = inputs.mdl_fun.Q_tilde.p_name;%Name of the parameters for the process noise
    ini.process_noise_param_ind = ismember(fields,p_name);
    [distr, ~]= NonGauss_Bayesian_filtering(ini,p,y,H,R,inputs,data);
    
    % Computing the negative log-likelihood
    neg_llh_y = sum(-log(ini.delta_x.*sum(distr.LH_of_chi_mat.*distr.p_tilde_mat,2)));
    %Below is equivalent, but slower
    %         neg_llh_y_vec = zeros(1,N_D);
    %         for k = 1:N_D
    %             neg_llh_y_vec(k) = ...
    %                 -log(ini.delta_x.*sum(distr.LH_of_chi_mat(k,:).*distr.p_tilde_mat(k,:)));
    %         end
    %         neg_llh_y = sum(neg_llh_y_vec);

end

%% Computing negative log posterior
phi = neg_llh_y + neg_llh_IAA + neg_log_prior;


end

