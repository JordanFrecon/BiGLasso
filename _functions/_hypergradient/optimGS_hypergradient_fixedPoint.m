function [ var, fun ] = optimGS_hypergradient_fixedPoint( opt, param )
% This function is intended to accelerate the reverse mode for computing
% the hypergradient when using the fixed point approach.
% Indeed, in that case, one can precompute and store some matrices to
% reduce the execution time
%
% General way to write some terms:
%   nabla2_Phi_star     :   (A'*var_a)*B + c*var_a
%   nabla2_Phi          :   (A'*var_a)*B + c*var_a
%   p1_mathcalB_transp  : ~ op(A*var_a)
%   p2_mathcalB_transp  : ~ op(A*var_a)


%% FUNCTIONS





%////  Nabla2_phi for hellinger-like function phi
%   (var1'*var2)*var1/(lambda_reg^2 - norm(var1)^2)^(3/2) + var2/sqrt(lambda_reg^2-norm(var1)^2);

    function [varout] = nabla_Hellinger_like(varin)
        varout.L    = varin';
        varout.R    = varin/(param.inner.lambda_reg^2 - norm(varin)^2)^(3/2);
        varout.M1   = 1;
        varout.M2   = 0;
        varout.f1   = varin;
        varout.f2   = 0;
        varout.k    = 1/sqrt(param.inner.lambda_reg^2-norm(varin)^2);
    end

%////  Nabla2_phi_star for hellinger-like function phi
%   -lambda_reg*(var1'*var2)*var1/(1 + norm(var1)^2)^(3/2) + lambda_reg*var2/sqrt(1+norm(var1)^2);
    function [varout] = nabla_Hellinger_like_star(varin)
        varout.L   = varin';
        varout.R   = -param.inner.lambda_reg*varin/(1 + norm(varin)^2)^(3/2);
        varout.M1  = 1;
        varout.M2  = 0;
        varout.f1  = varin;
        varout.f2  = 0;
    end

%% SAVING RELEVANT VARIABLES (can probably move this somewhere else)

var.nabla2_phi      = @(varin) nabla_Hellinger_like(varin);
var.nabla2_phi_star = @(varin) nabla_Hellinger_like_star(varin);

    function [varout] = gen_FP_nabla2(varsave,varin,indl)
        varout = varsave{indl}.L*( varsave{indl}.M1.*varsave{indl}.f1(varin) + varsave{indl}.M2.*varsave{indl}.f2(varin) )*varsave{indl}.R + varsave{indl}.k*(varin);
    end



fun.gen_nabla2 = @gen_FP_nabla2;





end

