function inputs = define_chi_interval(data,inputs,crstlk)
%% DEFINE_CHI_INTERVAL is used only when Non-Gaussian process noise is used and 
% determine the interval of chi, over which functions are integrated, in a
% way that is dependent on the data.

%% crosstalk coefficients
a = crstlk.a; 
d = crstlk.d;
G = crstlk.G;

%% raw data 
IDD = data.IDD;
IDA = data.IDA;
%IAA = data.IAA;
tDD = data.tDD;
%tAA = data.tAA;

%%
chi_interval_width = inputs.anl_params.chi_interval_width;

%% Getting estimation of IAA at the time points of donor excitation (i.e, tDD) 
mdl_fun = inputs.mdl_fun;
p_ini_vec = inputs.p_ini.p_ini_vec;
% list of fields
fields = fieldnames(mdl_fun.prior0);
% getting total concentrations of the acceptor from p
AT = p_ini_vec(ismember(fields,'AT'));
% parameters for f_A_fun
theta_A = p_ini_vec(ismember(fields,mdl_fun.f_A.p_name));
IAA_est = AT*mdl_fun.f_A.fun(theta_A,tDD);

%% (Crudely) Estimating chi

% Computing Ecorr
Fc = max(IDA - d*IDD - a*IAA_est,0);
R = Fc./IDD;
E_corr = (R./(R + G)).*(IAA_est(1)./IAA_est); 

% Estimating chi (= Ecorr times DT)
DT = p_ini_vec(ismember(fields,'DT'));
chi_est =  DT.*E_corr;

%% Defining the integral interval
chi_med = median(chi_est);
chi_range =  prctile(chi_est,99) - prctile(chi_est,1);
chi_integral_max = chi_med + (chi_interval_width/2)*chi_range;
chi_integral_min = chi_med - (chi_interval_width/2)*chi_range;
chi_integral_interval = [chi_integral_min chi_integral_max];

%% Keeping the result
inputs.anl_params.chi_intergral_interval = chi_integral_interval;

end

