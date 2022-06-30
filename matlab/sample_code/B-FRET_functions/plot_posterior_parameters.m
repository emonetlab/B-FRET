function plot_posterior_parameters(inputs,param_posterior,save_dir)
% PLOT_POSTERIOR_PARAMETERS plots the distributions of the parameters for
% sanity check

%% Moving to the folder in which the figures are saved 
cd(save_dir)
cd('./sanity_check')
%% Figure panels
subplot_N = 4;%row
subplot_M = 5;%column

%% Expanding inputs
mdl_fun = inputs.mdl_fun;
prior = inputs.prior;
fields = fieldnames(prior);% list of fields
prior0 = mdl_fun.prior0;
anl_params = inputs.anl_params;

panel_count = 0;
fig_count = 0;
if anl_params.do_MCMC ~=1 % For non-MCMC results, i.e., results from Laplace's method
    p_MAP_vec = param_posterior.p_MAP_vec;%MAP estimate of the parameters
    mvnsigma = param_posterior.mvnsigma;%Variance-Covariance matrix of the distr. of log parameters

    for i = 1:length(fields)
        panel_count = panel_count +1;
        if mod(panel_count,subplot_N*subplot_M) == 1
            figure('Position',[50, 50, 1400 700]);
            fig_count = fig_count + 1;
        end
        subplot_ind = panel_count - subplot_N*subplot_M*(fig_count - 1);
        % Check if the prior distr. is log-normal or not to set the xscale accordingly
        p_ini_val = inputs.p_ini.p_ini_str.(fields{i});%Initial estimation of the parameters.
        fun2 = prior0.(fields{i});%To look at the actual definition of the function
        TF = contains(func2str(fun2),'log','Ignorecase',true);%If the prior is lognormal
        if TF == 0 % if prior is NOT lognormal
            x = linspace(p_ini_val./100, p_ini_val*100,1E6);
            xscale = 'Linear';
        else
            x = logspace(log10(p_ini_val)-4,log10(p_ini_val)+4,1E6);
            xscale = 'Log';
        end
        
        %Plotting the priors
        fun = prior.(fields{i});%Function of the prior distr.
        y = fun(x);
        ind = find(y > max(y)*0.1);
        subplot(subplot_N,subplot_M,subplot_ind);hold on;
        l1 = plot(x(ind),y(ind));
        l2 = plot([p_ini_val, p_ini_val],[0 max(y)*2],'k--','LineWidth',1.5);
        
        % Plotting the posteriors
        y2 = lognpdf(x,log(p_MAP_vec(i)),sqrt(mvnsigma(i,i)));
        l3 = plot(x,y2,'LineWidth',1);
        
        %axis setting and legend
        set(gca,'XLim',[x(ind(1)), x(ind(end))],'YLim',[0 max(y)*2],'XScale',xscale);
        xlabel(fields{i})
        if i == length(fields)
            lh = legend([l1 l2 l3],'prior','initial','Laplace');
            set(lh,'FontSize',8,'Box','off','Location','SouthEast')
        end
        
        if mod(i,subplot_N*subplot_M) == 0 || i == length(fields)
            figtitle = sprintf(['posterior_prior_%d'],fig_count);
            set(gcf,'Visible','on')
            saveas(gcf,figtitle,'fig');
            %saveas(gcf,figtitle,'tif'); 
        end
    end
    
else
    %% Full posterior distribution sampled from MCMC
    p_sample = param_posterior.samples;
    
    for i = 1:length(fields)
        panel_count = panel_count +1;
        if mod(panel_count,subplot_N*subplot_M) == 1
            figure('Position',[50, 50, 1400 700]);
            fig_count = fig_count + 1;
        end
        subplot_ind = panel_count - subplot_N*subplot_M*(fig_count - 1);
        
        subplot(subplot_N,subplot_M,subplot_ind);hold on;
        
        % Check if the prior distr. is log-normal or not to set the xscale accordingly
        p_ini_val = inputs.p_ini.p_ini_str.(fields{i});%Initial estimation of the parameters.
        fun2 = prior0.(fields{i});%To look at the actual definition of the function
        TF = contains(func2str(fun2),'log','Ignorecase',true);%If the prior is lognormal
        if TF == 0 % if prior is NOT lognormal
            x = linspace(p_ini_val./100, p_ini_val*100,1E6);
            xscale = 'Linear';
        else
            x = logspace(log10(p_ini_val)-4,log10(p_ini_val)+4,1E6);
            xscale = 'Log';
        end
        
        %Plotting the priors
        fun = prior.(fields{i});%Function of the prior distr.
        y = fun(x);
        ind = find(y > max(y)*0.1);
        subplot(subplot_N,subplot_M,subplot_ind);hold on;
        l1 = plot(x(ind),y(ind));
        l2 = plot([p_ini_val, p_ini_val],[0 max(y)*2],'k--','LineWidth',1.5);
        
        %Plotting the posteriors
        [f,xi] = ksdensity(p_sample(:,i));
        l3 = plot(xi,f);
        
        %axis setting and legend
        set(gca,'XLim',[x(ind(1)), x(ind(end))],'YLim',[0 max(y)*2],'XScale',xscale);
        xlabel(fields{i})
        if i == length(fields)
            lh = legend([l1 l2 l3],'prior','initial','post');
            set(lh,'FontSize',8,'Box','off','Location','SouthEast')
        end
        if mod(i,subplot_N*subplot_M) == 0 || i == length(fields)
            figtitle = sprintf(['posterior_prior_%d'],fig_count);
            set(gcf,'Visible','on')
            saveas(gcf,figtitle,'fig');
            %saveas(gcf,figtitle,'tif'); 
        end
    end
    
end


%% MCMC trajectories
if anl_params.do_MCMC == 1
    p_sample = param_posterior.samples;

    N = 6;%# of subplot per figure
    subplot_ind = mod([1:length(fields)],N);
    subplot_ind(subplot_ind == 0) = N;
    fig_num = 1;
    for i = 1:length(fields)
        
        if mod(i,N) == 1
            FH = figure('Position',[50 50 1000 700]);
            FH.Renderer = 'painters';
        end
        subplot(N,1,subplot_ind(i))
        plot(p_sample(:,i));
        ylabel(fields{i})
        if mod(i,N) == 0 || i == length(fields)
            figtitle = sprintf(['MCMC_traces_%d'],fig_num);
            saveas(gcf,figtitle,'fig');
            %saveas(gcf,figtitle,'tif');
            fig_num = fig_num + 1;
        end
        
    end
    
end


cd(save_dir)
end

