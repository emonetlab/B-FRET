function FRET_sgnl = compute_EFRET_signals(data,inputs,param_posterior,state_posterior,crstlk)

%% data and crosstalk coefficients
IDD = data.IDD;
IDA = data.IDA;
IAA = data.IAA;

%% Computing algebraic E-FRET only if G is defined by the user

a = crstlk.a;
d = crstlk.d;
G = crstlk.G;

% prediction of IAA
prior = inputs.prior;
mdl_fun = inputs.mdl_fun;
anl_params = inputs.anl_params;
do_MCMC = anl_params.do_MCMC;
fields = fieldnames(prior);% list of fields
% Obtaining estimation of most probable parameters
if do_MCMC ~= 1
    p = param_posterior.p_MAP_vec;
else
    p = median(param_posterior.samples);
end
theta_A = p(ismember(fields,mdl_fun.f_A.p_name));
f_A_fun = @(t) mdl_fun.f_A.fun(theta_A,t);
AT = p(ismember(fields,'AT'));
IAA_pred = AT.*f_A_fun(data.tDD);

Fc = IDA - d.*IDD - a.*IAA_pred;
R = Fc./IDD;
FRET_sgnl.Ecorr = (R./(R + G)).*(IAA(1)./IAA_pred);


%% Computing E-FRET based on the results of Laplace's method
E = state_posterior.E;
FRET_sgnl.E_0p13 = prctile(E,0.13);
FRET_sgnl.E_2p27 = prctile(E,2.27);
FRET_sgnl.E_15p87 = prctile(E,15.87);
FRET_sgnl.E_med = median(E);
FRET_sgnl.E_84p13 = prctile(E,84.13);
FRET_sgnl.E_97p73 = prctile(E,97.73);
FRET_sgnl.E_99p87 = prctile(E,99.87);

end

