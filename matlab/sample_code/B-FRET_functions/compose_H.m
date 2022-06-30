function H = compose_H(p,data,crstlk,mdl_fun)
% COMPOSE_H computes the matrix H given model functions and parameters
%
% Inputs:
%p: A vector containing model parameters
%data: A structure containing 3 background-subtructed fluorescent intensity timeseries and their time
%crstlk: A structure containing the values of crosstalk coefficients (a, d, and G)
%mdl_fun: A structure containing the functional forms of the model and priors 
%
% Outputs:
% H: A cell array that stores the matrix H at each time point.

%%
% list of fields
fields = fieldnames(mdl_fun.prior0);

% getting total concentrations of the donor and acceptor from p
DT = p(ismember(fields,'DT'));
AT = p(ismember(fields,'AT'));

% parameters for f_D_fun and f_A_fun
theta_D = p(ismember(fields,mdl_fun.f_D.p_name));
theta_A = p(ismember(fields,mdl_fun.f_A.p_name));

%defining bleaching function (fluorescent intensities)
f_D_fun = @(t) mdl_fun.f_D.fun(theta_D,t);
f_A_fun = @(t) mdl_fun.f_A.fun(theta_A,t);

% H has the same size as IDD
L = length(data.IDD);
H = cell(1,L);

% time for IDD
tDD = data.tDD;

%crosstalk coefficients
a = crstlk.a;
d = crstlk.d;
G = crstlk.G;

% composing H
for k = 1:L
    t_k = tDD(k);
    fD = f_D_fun(t_k);
    fA = f_A_fun(t_k);
    H{k} = [DT*fD, -fD*fA; (d*DT*fD + a*AT*fA), (G-d)*fD*fA];
end

end

