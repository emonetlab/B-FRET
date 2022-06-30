function [p_ini, prior] = define_prior(data,crstlk,inputs,save_dir)
% DEFINE_PRIOR estimates the rough values of the model parameters that
% are optimized later and define priors using the inital estimations. 
%
% Input variables:
% data: A structure containing 3 background-subtructed fluorescent intensity timeseries and their time
% crstlk: A structure containing cross-talk coefficients (a, d, and G);
% inputs: A structure containing analysis parameters and model functions etc. 
% savedir: The directly in which sanity check plots are stored

% Output variable:
% p_ini.p_ini_str: A structure that contains each parameter's initial value for optimization
% p_ini.p_ini_vec: A vector that contains all parameters's initial values for optimization
% prior: A structure that contains actual prior functions for each parameters. 
% Note the differnece between mdl_fun.prior0 and prior defined here. The former 
% is dependent on the input (p_ini), and thus once you specify p_ini, you get prior defined here. 

%% expanding inputs
mdl_fun = inputs.mdl_fun;
prior0 = mdl_fun.prior0;
anl_params = inputs.anl_params;

%% observables and imaging time
IDD = data.IDD;
IAA = data.IAA;
IDA = data.IDA;
tDD = data.tDD;
tAA = data.tAA;

%% Setting initial parameter values befor rough estimation
f_D_params0 = [mdl_fun.f_D.p_ini IDD(1)];
f_D_p_lb =  [mdl_fun.f_D.p_lb 0]; %lower bound
f_D_p_ub =  [mdl_fun.f_D.p_ub inf]; %upper bound

f_A_params0 = [mdl_fun.f_A.p_ini IAA(1)];
f_A_p_lb =  [mdl_fun.f_A.p_lb 0]; %lower bound
f_A_p_ub =  [mdl_fun.f_A.p_ub inf]; %upper bound


%% Rough parameter estimation that are optimized later
% f_D: Bleaching curve of the donor
fun_f_D = @(params,t) params(end)*mdl_fun.f_D.fun(params(1:end-1),t);
params_fit_f_D = lsqcurvefit(fun_f_D,f_D_params0,tDD,IDD,...
    f_D_p_lb,f_D_p_ub,anl_params.options1);

IDD_1 = params_fit_f_D(end);
for i = 1:length(params_fit_f_D)-1
    p_ini_str.(mdl_fun.f_D.p_name{i}) = params_fit_f_D(i);
end

% f_A: Bleaching curve of the acceptor
fun_f_A = @(params,t) params(end)*mdl_fun.f_A.fun(params(1:end-1),t);
params_fit_f_A = lsqcurvefit(fun_f_A,f_A_params0,tAA,IAA,...
    f_A_p_lb,f_A_p_ub,anl_params.options1);


IAA_1 = params_fit_f_A(end);
for i = 1:length(params_fit_f_A)-1
    p_ini_str.(mdl_fun.f_A.p_name{i}) = params_fit_f_A(i);
end

% Total concentrations of acceptor and donor
p_ini_str.AT = IAA_1;
p_ini_str.DT = (IDA(1) + (crstlk.G-crstlk.d)*IDD_1 - crstlk.a*IAA_1)/crstlk.G;%This DT "fits" the data point (IDD_1, IDA(1))


% Rough estimation of sigma_chi provided by the user
p_ini_str.sigma_chi = anl_params.sigma_chi_ini;


if ~strcmp(anl_params.process_noise,'Gaussian')% for Non-Gaussian process noise 
    p_ini_str.nu = anl_params.nu_ini;
end


%% Plotting results for sanity check 

cd(save_dir);
mkdir('./sanity_check');
cd('./sanity_check');

% Bleaching treand for IDD and IDA
figure,
subplot(2,1,1);hold on;
l1 = plot(tDD,IDD,'c');
l2 = plot(tDD, fun_f_D(params_fit_f_D,tDD),'k');
legend([l1 l2],'I_{DD}','fit');
ylabel('Intensity');xlabel('Time (sec)');
subplot(2,1,2); hold on;
l1 = plot(tAA,IAA,'m');
l2 = plot(tAA, fun_f_A(params_fit_f_A,tAA),'k');
legend([l1 l2],'I_{AA}','fit');
ylabel('Intensity');xlabel('Time (sec)');
set(gcf,'Visible','on')
saveas(gcf,'initial_est_bleach_trend','fig');
%saveas(gcf,'initial_est_bleach_trend','tif');


%% defining priors using the predefined functional forms
% list of fields
fields = fieldnames(prior0);

% The inital estimation value of each parameter defines the location of the
% prior distribution
for i = 1:length(fields)
    prior.(fields{i}) = @(x) prior0.(fields{i})(x,p_ini_str.(fields{i}));
end

%% The parameters are organized in the same order as the structure of priors 
p_ini_vec = zeros(1,length(fields));
for i = 1:length(fields)
   p_ini_vec(i) = p_ini_str.(fields{i});  
end

%% outputs
p_ini.p_ini_str = p_ini_str;
p_ini.p_ini_vec = p_ini_vec;

end

