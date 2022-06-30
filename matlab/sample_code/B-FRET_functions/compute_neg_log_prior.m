function neg_log_prior = compute_neg_log_prior(p,prior,fields)
% DEFINE_NEG_LOG_POSTERIOR defines negative log prior distribution

% Input variables
% p: Vector containing parameter values
% prior: A structure containing function handles for prior functions
% fields: fieldnames(prior);

%% 
L = length(p);
neg_log_prior_vec = zeros(1,L);
for i = 1:L
    neg_log_prior_vec(i) = - log(prior.([fields{i}])(p(i)));
end

neg_log_prior = sum(neg_log_prior_vec);
end

