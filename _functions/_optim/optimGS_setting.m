function [ opt ] = optimGS_setting( param )
%OPTIMGS_INIT Summary of this function goes here
%   Detailed explanation goes here

lambda_reg  = param.lambda_reg;
EPS         = param.eps;

%% GROUPS

% // Linear operator (and its adjoint) defining the group structure
opt.opA         = @(var1,var2) var1.*var2(:,ones(1,size(var1,2))); %var1.*repmat(var2,[1 size(var1,2)]);
opt.opA_star    = @(var1,var2) sum(var1.*var2,2);





%% FIDELITY & REGULARIZER


    %This function is intended to bridge the gap between array and cell
    %computation. This will permit to handle the case where the tasks do
    %not have the same #observations
    function [otp_var] = sumOverTasks(varfun,vary,varX,varw)
        if iscell(vary)     %When given a group of tasks as a cell
            tmp_var = zeros(1,length(vary));
            for tt=1:length(vary)
                tmp_var(tt) = varfun(vary{tt},varX{tt},varw(:,tt))/length(vary{tt});
            end
            otp_var = sum(tmp_var);
        else                %When given a single task
            otp_var = varfun(vary,varX,varw)/length(vary);
        end
    end


% Functions defining the lower-level problem
l2fid                   = @(vary,varX,varw) .5*norm(vary-varX*varw)^2;
opt.fidelity            = @(y,X,w) sumOverTasks(l2fid,y,X,w);
opt.opA0                = @(y,X,w) 0;
opt.opA0_star           = @(y,X,t) 0;
opt.lipschitz           = @(y,X,theta) lambda_reg/EPS*normGroupOperator(opt.opA,theta);
opt.NSD_mat             = @(y,X) inv(X'*X/length(y)+EPS*eye(size(X,2)));
opt.NSD_vec             = @(y,X) X'*y/length(y);
opt.legendreFunction0   = 'void';
opt.dCdw                = @(yval,Xval,w) (Xval'*Xval*w - Xval'*yval)/length(yval);
opt.nabla_smooth_dual   = @(Mat,vec,var) Mat*(vec+var);    
opt.regularizer         = @(v) lambda_reg*sum(sqrt(sum(v.^2,1)));
opt.legendreFunction    = 'Hellinger-like';
opt.objective_primal    = @(y,X,w,v,theta) opt.fidelity(y,X,w) + opt.regularizer(v) + .5*EPS*norm(w)^2;


%% LEGENDRE FUNCTIONS
opt.nabla_phi       = @(varin) varin./(sqrt(lambda_reg^2 - norm(varin)^2));
opt.nabla_phi_star  = @(varin) lambda_reg*varin./sqrt(1+norm(varin)^2);
opt.nabla2_phi      = @(var1,var2) (var1'*var2)*var1/(lambda_reg^2 - norm(var1)^2)^(3/2) + var2/sqrt(lambda_reg^2-norm(var1)^2);
opt.nabla2_phi_star = @(var1,var2) -lambda_reg*(var1'*var2)*var1/(1 + norm(var1)^2)^(3/2) + lambda_reg*var2/sqrt(1+norm(var1)^2);
opt.prox_Phi0       = @(varin,cst) 0;
opt.nabla_Phi0      = @(varin) 0;
opt.nabla_prox_Phi0 = @(var1,var2,cst) 0;
opt.nabla2_Phi0     = @(var1,var2) 0;




end

