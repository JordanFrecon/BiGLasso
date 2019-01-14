function [ w,output ] = optimGS_lower( y, X, theta, opt, param )
%OPTIMGS_LOWER Summary of this function goes here
%
%
%
% Comments:
%
% 1- Investigate if more efficient to define local functions
%
% 2- More efficient to use this
%   cst = matinv*X_adj_y;
%   nabla_smooth_dual_bis = @(varin) cst+matinv*varin;
%   w   = nabla_smooth_dual_bis(-opt.opA_star(theta,u) - opt.opA0_star(y,X,u0));
  



%% HEADER

%///// Pre-allocation
output = struct;

%///// Complexity Opt of the gradient of smooth part complexity
%- Since the smooth part is quadratic, the gradient is defined by affine operator 
nsd_vec             = opt.NSD_vec(y,X);             %can be stored
output.nsd_mat      = opt.NSD_mat(y,X);               %can be stored
nabla_smooth_dual   = @(varin) opt.nabla_smooth_dual(output.nsd_mat,nsd_vec,varin);


%///// Step size (can be stored)
if ~isfield(param,'stepsize')
    stepsize = .99/opt.lipschitz(y,X,theta);
else
    assert(param.stepsize < 1/opt.lipschitz(y,X,theta),'Invalid stepsize');
    stepsize = param.stepsize;
end


% Compute objective
if param.compObjective
    output.objective = NaN(param.itermax,1);
end

%///// Initialization
if ~isfield(param,'initialPoint')
    u = zeros(size(theta));
else
    u   = param.initialPoint.u;
end


%///// Clear memory space
f       = fieldnames(param);
param   = rmfield(param,[f(~ismember(f,{'itermax','nGroups','compObjective','saveIterates'}))]);
clear('nsd_vec','f')


%% MAIN
for iter = 1:param.itermax
    
    %%%%%%%%%%%%%%%%% Forward Backward Scheme %%%%%%%%%%%%%%%%%
    
    % Intermediate step: primal update
    w   = nabla_smooth_dual(-opt.opA_star(theta,u));
    Aw  = opt.opA(theta,w);
    
    % Forward step
    v = stepsize*Aw;
    for ll=1:param.nGroups
        v(:,ll) = v(:,ll) + opt.nabla_phi(u(:,ll));
    end
    
    % Backward step: dual update
    for ll=1:param.nGroups
        u(:,ll) = opt.nabla_phi_star(v(:,ll));
    end

    
    %%%%%%%%%%%%%%%%%%% Storing and measures %%%%%%%%%%%%%%%%%%%
    
    % Primal objective
    if param.compObjective
        output.objective(iter) = opt.objective_primal(y,X,w,Aw,theta);
    end
    
    % Storing: iterates (verify location)
    if param.saveIterates
        output.iterates.w{iter}     = w;
        output.iterates.v{iter}     = v;
        output.iterates.u{iter}     = u;
    end
    
end

%% OUTPUT
output.w        = w;
output.v        = v;
output.u        = u;
output.stepsize = stepsize;



end

