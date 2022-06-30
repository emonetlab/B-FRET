function Q = compose_Q(p,data,mdl_fun)
% COMPOSE_Q computes the matrix Q given model functions and parameters
%
% Inputs:
% p: A vector containing model parameters
% data: A structure containing 3 background-subtructed fluorescent intensity timeseries and their time
% mdl_fun: A structure containing the functional forms of the model and priors
%
% Outputs:
% Q: A cell array that stores the matrix Q at each time point.

%%
% list of fields
fields = fieldnames(mdl_fun.prior0);

% getting sigma_chi from p
sigma_chi = p(ismember(fields,'sigma_chi'));

% Q has the same size as IDD
L = length(data.IDD);
Q = cell(1,L);

% Composing Q
for k = 1:L
    Q{k} = [0, 0; 0, sigma_chi^2];
end


end

