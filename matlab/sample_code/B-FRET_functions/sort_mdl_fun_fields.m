function mdl_fun = sort_mdl_fun_fields(mdl_fun,anl_params)

if strcmp(anl_params.process_noise,'Gaussian')
   C = [mdl_fun.f_D.p_name, 'DT', mdl_fun.f_A.p_name, 'AT', 'sigma_chi'];   
elseif strcmp(anl_params.process_noise,'Non-Gaussian')
   C = [mdl_fun.f_D.p_name, 'DT', mdl_fun.f_A.p_name, 'AT', 'sigma_chi', 'nu'];  
end

mdl_fun.prior0 = orderfields(mdl_fun.prior0,C); 

end

