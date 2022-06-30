function analyze_FRET_data(all_data,dir_info)
%% ANALYZE_FRET_DATA analyzes FRET data contained in 'all_data' using the B-FRET algorithm.
%The detail of this anlysis is specified by DEFINE_ANALYSIS_PARAMS.M and DEFINE_MODEL_FUNCTIONS.M that have to be properly edited and stored in the folder specified by 'dir_info.root_dir'.

% Inupt variables:
%all_data: A structure containing the data to be analyzed. The structure must contain the following fields:
%   all_data.IDD: M by N_D matrix (double) for IDD where M is the number of timeseries and N_D is the number of time points
%   all_data.IDA: M by N_D matrix (double)for IDA
%   all_data.IAA: M by N_A matrix (double)for IAA
%   all_data.IDD_noise_sd: M by N_D matrix (double) that specify the standard deviation of the measurement noise of IDD at each time point
%   all_data.IDA_noise_sd: M by N_D matrix (double) that specify the standard deviation of the measurement noise of IDA at each time point
%   all_data.IAA_noise_sd: M by N_D matrix (double) that specify the standard deviation of the measurement noise of IAA at each time point
%   all_data.tDD: A vector (1 by N_D; double) containing the measurement time for IDD and IDA.
%   all_data.tAA: A vector (1 by N_A; double) containing the measurement time for IAA.
%   all_data.crstlk: A structure that contains three fields, a, d and G, which are scalars representing imaging-system-dependent parameters
% Note that it is assumed that for IDD, IDA, IAA, IDD, IDA and IAA i-th row (i = 1,..., M) is from the same sample.  

%dir_info: A structure containing directory information
%   dir_info.root_dir: The directory in which DEFINE_ANALYSIS_PARAMS.M and DEFINE_MODEL_FUNCTIONS.M are placed.
%   dir_info.sub_dir_base_name: The base name of the folder created for each data (i.e., for each row of IDD, IDA, IAA etc.).

% Output variables:
%Bayes_FRET_result.mat: A MAT_File (.mat) is created in each folder for each data. This contains a structure array 'analyses_results' with the following fields:
%   analyses_results.data: A structure for the data analyzed by B-FRET. These are transfered form the the structure array, 'all_data' (see above).
%   analyses_results.crstlk:  A structure for a, d and G,transfered from the structure 'all_data.crstlk' (see above).
%   analyses_results.inputs: A structure array for the inputs for the algorithm with the following fields:
%       inputs.anl_params: A structure for the parameters that regulate the behavior of the algorithm. These are defined in DEFINE_ANALYSIS_PARAMS.M.
%       inputs.mdl_fun: A structure for the model functions used for the analysis. These are defined in DEFINE_MODEL_FUNCTIONS.M. 
%       inputs.prior: A structure for the functions of all priors
%       inputs.p_ini: A structure for the model-parameter values obtained by initial rough estimation.  
%            p_ini.p_ini_str: a structure for each parameter's initial-estimation value.
%            p_ini.p_ini_vec: a vector containing all parameters's initial estimations (stored in the same order as fields(inputs.prior)) 
%       inputs.SSM: A structure for the definition of the state-space model.
%   analyses_results.param_posterior: A structure containing informaiton related to the posterior distributions of model parameters with the following fields:
%       param_posterior.samples: X (number of samples) by Y (number of parameters) matrix containing parameters sampled form the posteriors (Y is organized in the same order as fields(inputs.prior)).
%       param_posterior.BIC: A scaler for the Bayesian information criterion.
%       param_posterior.p_MAP_vec: MAP of the parameters (only if anl_params.do_MCMC ~= 1, i.e., if the posterior is approximated by a Gaussian without doing MCMC sampling)
%       param_posterior.mvnsigma: Variance-covariance matrix of the distribution of log-parameters (only if anl_params.do_MCMC ~= 1)
%       param_posterior.log_mu: Mode of the distribution of log-parameters (only if anl_params.do_MCMC ~= 1)
%       param_posterior.nearestSPD: This is 1, only if the direct numerical estimation of variance-covariance matrix of the Gaussian disribution is replaced with the nearest positive definite matrix.  
%    analyses_results.state_posterior: A structure containing information about the posterior of the latent variable  witht the following fields:
%        state_posterior.E: X (number of samples) by Y (length of time series) matrix for E; each row is a realization of E sampled from the posterior.
%        state_posterior.chi: X (number of samples) by Y (length of time series) matrix for \chi; each row is a realization of \chi sampled from the posterior.
%    analyses_results.obs_pred: A structure containing predictions of the observables, IAA, IDA, and IDD, based on the estimated model parameters and hidden variable. 
%    analyses_results.FRET_sgnl: A structure containing some summary statistics about FRET signals with the following fields:
%        FRET_sgnl.E_0p13: A vector containing 0.13 percentile of E at each time point
%        FRET_sgnl.E_2p27: A vector containing 2.27 percentile of E at each time point
%        FRET_sgnl.E_15p87: A vector containing 15.87 percentile of E at each time point
%        FRET_sgnl.E_med: A vector containing 50 percentile of E at each time point
%        FRET_sgnl.E_84p13: A vector containing 84.13 percentile of E at each time point
%        FRET_sgnl.E_99p87: A vector containing 99.87 percentile of E at each time point
%        FRET_sgnl.Ecorr: A vector containing Ecorr (FRET index computed by the E-FRET method)

%% Defining analysis methods and model functions
cd(dir_info.root_dir)
anl_params = define_analysis_params();
mdl_fun = define_model_functions(anl_params);
mdl_fun = sort_mdl_fun_fields(mdl_fun,anl_params);
addpath(dir_info.root_dir);%This seems to be necessary to avoid warnings



%% The following is executed for each data
for i = 1:size(all_data.IDD,1)
    tic
    try
        %% Creating a folder in which results are saved
        cd(dir_info.root_dir)
        if ~exist([dir_info.sub_dir_base_name num2str(i)], 'dir')
            mkdir([dir_info.sub_dir_base_name num2str(i)])
            cd([dir_info.sub_dir_base_name num2str(i)]);
            save_dir = pwd;
            
            %% Organizing single data
            data.IDD = all_data.IDD(i,:);
            data.IAA = all_data.IAA(i,:);
            data.IDA = all_data.IDA(i,:);
            data.IDD_noise_sd = all_data.IDD_noise_sd(i,:);
            data.IDA_noise_sd = all_data.IDA_noise_sd(i,:);
            data.IAA_noise_sd = all_data.IAA_noise_sd(i,:);
            data.tDD = all_data.tDD;
            data.tAA = all_data.tAA;
            if isfield(all_data,'E')%If the data is synthetic data and contains true E
                data.E = all_data.E(min(i,size(all_data.E,1)),:);
            end
            
            %% Preparing cross-talk coefficients
            crstlk.a = all_data.crstlk.a; 
            crstlk.d = all_data.crstlk.d;
            crstlk.G = all_data.crstlk.G;
            
            %% Defining the model, measurement noise and priors
            inputs = [];%initialization
            inputs.anl_params = anl_params; % analysis-related parameters
            inputs.mdl_fun = mdl_fun; % model functions
            inputs.SSM = define_state_space_model(data,crstlk,inputs); % State space model
            [inputs.p_ini, inputs.prior] = define_prior(data,crstlk,inputs,save_dir); % inital parameters and priors
            
            %% Only for Non-Gaussian model, the interval of chi integration is estimated from data
            if strcmp(anl_params.process_noise,'Non-Gaussian')%If Non-Gaussian model is used 
                inputs = define_chi_interval(data,inputs,crstlk);%Define the interval of chi over which functions are integrated
            end
            %% Sampling from the posterior distribuiton of the parameters, p(\theta|D,M)
            param_posterior = sample_from_param_posterior(data,inputs);
            analyses_results.data = data;
            analyses_results.crstlk = crstlk;
            analyses_results.param_posterior = param_posterior;
            plot_posterior_parameters(inputs,param_posterior,save_dir)
            %% Sampling from the posterior distribuiton of the hidden state, p(E_k|D,M) or p(\chi_k|D,M)
            state_posterior = sample_from_state_posterior(data,inputs,param_posterior,save_dir);
            analyses_results.state_posterior = state_posterior;
            %% Computing the observables from the obtained results for a sanity check
            obs_pred = compute_observable_predictions(data,inputs,param_posterior,state_posterior);
            FRET_sgnl = compute_EFRET_signals(data,inputs,param_posterior,state_posterior,crstlk);
            plot_timeseries(data,obs_pred,FRET_sgnl,save_dir);
            analyses_results.obs_pred = obs_pred;
            analyses_results.FRET_sgnl = FRET_sgnl;
            inputs = rmfield(inputs,'mdl_fun');
            analyses_results.inputs = inputs;
            save('Bayes_FRET_result.mat','analyses_results');
            
            close all;
            
        end
        
    catch ME
        warning(['Error occured. Error message:' ME.message])
    end
toc    
end
rmpath(dir_info.root_dir);
close all;
disp('Done');
end

