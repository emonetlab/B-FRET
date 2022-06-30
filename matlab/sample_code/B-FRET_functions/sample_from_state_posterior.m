function state_posterior = sample_from_state_posterior(data,inputs,param_posterior,save_dir)
% SAMPLE_FROM_STATE_POSTERIOR samples from the posterior distribuiton of
% the states, i.e., p(chi_k|D,M) and p(E_k|D,M).
%
% Input variables
% data: A structure containing data
% inputs: A structure containing information about the model, priors, etc
% param_posterior: A structure containging posterior samples of the parameters
%
% Output variables:
% state_posterior.E: A matrix containging samples from the posterior distribution of E (= chi/DT), p(E_k|D,M)
% state_posterior.chi: A matrix containging samples from the posterior distribution of chi, p(chi_k|D,M)

%%
disp('Drawing samples form the posterior distribution of the states...')
%% expanding the input variables
prior = inputs.prior;
SSM = inputs.SSM;
anl_params = inputs.anl_params;
param_samples = param_posterior.samples;

%% Preparing some common variables
ind = ismember(fieldnames(prior),'DT'); %index for DT
N_D = length(data.IDD);
E_mat = zeros(size(param_samples,1),N_D);
chi_mat = zeros(size(param_samples,1),N_D);

%% Sampling posterior distribution of the state (chi and E)
if strcmp(anl_params.process_noise,'Gaussian')%If the process noise is Gaussian
    %% Gaussian case
    parfor i = 1:size(param_samples,1)
        p = param_samples(i,:);
        DT_i = p(ind);
        
        %RTS smoother
        RTS_state = RTS_smoothed_state(p,data,prior,inputs,anl_params);
        
        %11/4/2021; This for loop is faster than with cellfun
        %sampling from the posterior distribution of chi and E
        for k = 1:N_D
            chi = normrnd(RTS_state.ms{k}(2),sqrt(RTS_state.Ps{k}(4)));
            chi_mat(i,k) = chi;
            E_mat(i,k) = chi/DT_i;
        end
    end
    
else %If the process noise is Non-Gaussian
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
    x_interp = linspace(min_x,max_x,1E4);%Fine discritization for the sampling of chi;
    
    % Defining a part of the state-space model
    R = SSM.R;
    y = SSM.y;
    
    parfor i = 1:size(param_samples,1)
        % computing the filting and predictive distributions
        p = param_samples(i,:);
        H = SSM.H_fun(p);
        DT_i = p(ind);
        [distr, Q_tilde_mat]= NonGauss_Bayesian_filtering(ini,p,y,H,R,inputs,data);

        % computing the smoothing distribution
        distr = NonGauss_smoothing_distr(ini,distr,Q_tilde_mat);
        
        % sampling from the CDF of the smoothing distribution
        s_tilde_CDF_mat = distr.s_tilde_CDF_mat;
        for k = 1:N_D
            y_interp = interp1(ini.x,s_tilde_CDF_mat(k,:),x_interp);%Interpolatin to achieve fine discretization
            CDF_val = unifrnd(0,1);
            ind_x = find(CDF_val <= y_interp,1);
            chi = x_interp(ind_x);
            chi_mat(i,k) = chi;
            E_mat(i,k) = chi/DT_i;
            
            % For sanity check
            %figure,hold on;
            %plot(ini.x,s_tilde_CDF_mat(k,:),'Marker','o');
            %plot(x_interp,y_interp,'Marker','.');
        end

        
    end
    
    % Creating a figure for sanity check
    i = 1;
    p = param_samples(i,:);
    H = SSM.H_fun(p);
    [distr, Q_tilde_mat]= NonGauss_Bayesian_filtering(ini,p,y,H,R,inputs,data);
    distr = NonGauss_smoothing_distr(ini,distr,Q_tilde_mat);
    
    figure,
    subplot(2,3,1);hold on;
    N_D_int = round(N_D/10);
    for ii = 1:N_D_int:N_D
        plot(ini.x,distr.f_tilde_mat(ii,:),'Marker','.')
    end
    xlabel('\chi_k');ylabel('Filtering distr.');set(gca,'XLim',[min_x, max_x])
    subplot(2,3,2);hold on;
    for ii = 1:N_D_int:N_D
        plot(ini.x,distr.p_tilde_mat(ii,:),'Marker','.')
    end
    xlabel('\chi_k');ylabel('Predictive distr.');set(gca,'XLim',[min_x, max_x])
    subplot(2,3,3);hold on;
    for ii = 1:N_D_int:N_D
        plot(ini.x,distr.LH_of_chi_mat(ii,:),'Marker','.')
    end
    xlabel('\chi_k');ylabel('N({\bfy_k}|{\bfH_k}(1,\chi_k)^T,{\bfR_k})');set(gca,'XLim',[min_x, max_x])
    subplot(2,3,4);hold on;
    for ii = 1:N_D_int:N_D
        plot(ini.x,distr.s_tilde_mat(ii,:),'Marker','.')
    end
    xlabel('\chi_k');ylabel('Smoothing distribution');set(gca,'XLim',[min_x, max_x])  
    
    subplot(2,3,5);hold on;
    for ii = 1:N_D_int:N_D
        plot(ini.x,distr.s_tilde_CDF_mat(ii,:),'Marker','.')
    end
    xlabel('\chi_k');ylabel('Smoothing distribution (CDF)');set(gca,'XLim',[min_x, max_x])  
   
    cd(save_dir)
    cd('./sanity_check')
    set(gcf,'Visible','on')
    saveas(gcf,'approximated_distrs_examples','fig')
    %saveas(gcf,'approximated_distrs_examples','tif')
    cd(save_dir)
    
 
end
%% Organizing the output
state_posterior.chi = chi_mat;
state_posterior.E = E_mat;



end

