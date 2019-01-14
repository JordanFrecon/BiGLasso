function [ b ] = optimGS_hypergradient(y, yval,X, Xval, theta, opt, param, var)
% [ b ] = hyperGradient_reverse(var, theta, opt, param, var.nsd_mat)
% Reverse mode differentiation for computing the hypergradient 'b'
%
% Versions:
%   24 Oct. 2018
%       - need to get rid of y and X as inputs
%       - working version for group Lasso
%   17 Oct. 2018: creation
%       - purpose: Generalize the hyperGradient function to any loss/reg




%% Legendre function Phi


    % /// Extension from phi -> Phi (over all param.nGroups groups)
    %(Possibly too costly)
    
    function [otp_var] = nabla2_Phi(var1,var2)
        otp_var = NaN(length(var1),param.inner.nGroups);
        for ll=1:param.inner.nGroups
           otp_var(:,ll) = opt.nabla2_phi(var1(:,ll),var2(:,ll)); 
        end
    end
    
    function [otp_var] = nabla2_Phi_star(var1,var2)
        otp_var = NaN(length(var1),param.inner.nGroups);
        for ll=1:param.inner.nGroups
           otp_var(:,ll) = opt.nabla2_phi_star(var1(:,ll),var2(:,ll)); 
        end
    end



%% Partial derivatives of mapping B     (Be more general)

    function [otp_var] = prodCustom(var1,var2)
        if iscell(var1)
            otp_var = NaN(size(var2));
            for tt=1:size(var2,2)
                otp_var(:,tt) = var1{tt}*var2(:,tt);
            end
        else
            otp_var = var1*var2;
        end
    end


    % H -> R^NxHL^L
    function [otp_var,otp_var0] = p1_mathcalB_transp(var_a)
        prodvar     = prodCustom(var.nsd_mat,var_a);
        otp_var     = -opt.opA(theta,prodvar);
        otp_var0    = -opt.opA0(y,X,prodvar);
    end

    % H -> Theta
    function [otp_var] = p2_mathcalB_transp(var_u,var_a)
        otp_var     = -opt.opA(var_u, prodCustom(var.nsd_mat,var_a));
    end




%% Partial derivatives of mapping A     (Be more general)


    % R^NxH^L -> R^NxH^L
    function [otp_var,otp_var0] = p1_mathcalA_transp(var_u,var_u0,var_v,var_v0,var_a,var_a0)
        vartmp  = nabla2_Phi_star( var_v,var_a );
        otp_var = nabla2_Phi(var_u,vartmp) + var.stepsize*p1_mathcalB_transp( opt.opA_star(theta,vartmp) );
        
        vartmp  = opt.nabla_prox_Phi0( var_v0,var_a0,var.stepsize ); 
        [~,tmp2] = p1_mathcalB_transp( opt.opA0_star(y,X,vartmp) );
        otp_var0 =  opt.nabla2_Phi0(var_u0,vartmp) + var.stepsize*tmp2;
    end

    % R^NxH^L -> Theta
    function [otp_var_sum] = p2_mathcalA_transp(var_u,var_v,var_v0,var_w,var_a,var_a0)
        vartmp = nabla2_Phi_star( var_v,var_a );
        tmp = opt.opA( vartmp, var_w );
        otp_var = var.stepsize*p2_mathcalB_transp( var_u, opt.opA_star(theta,vartmp) ) + var.stepsize*tmp;
        
        vartmp2  = opt.nabla_prox_Phi0( var_v0,var_a0,var.stepsize ); 
        otp_var0 = var.stepsize*p2_mathcalB_transp( var_u, opt.opA0_star(y,X,vartmp2) ); 
        
        otp_var_sum = otp_var + otp_var0;
    end



%% Fixed-point vs. standard algorithm

% A1: R^NxH^L -> R^NxH^L
    function [otp_var,otp_var0] = A1(var_a,var_a0,iiter)    %(Might be too costly)
        if param.outer.fixedPointHG
            [otp_var,otp_var0] = p1_mathcalA_transp(var.u, var.u0, var.v, var.v0, var_a,var_a0);
        else
            [otp_var,otp_var0] = p1_mathcalA_transp(var.iterates.u{iiter}, var.iterates.u0{iiter}, var.iterates.v{iiter}, var.iterates.v0{iiter}, var_a,var_a0);
        end
    end

% A2: R^NxH^L -> Theta
    function [otp_var] = A2(var_a,var_a0,iiter)
        if param.outer.fixedPointHG
            [otp_var] = p2_mathcalA_transp(var.u, var.v, var.v0, var.w, var_a,var_a0);
        else
            [otp_var] = p2_mathcalA_transp(var.iterates.u{iiter}, var.iterates.v{iiter}, var.iterates.v0{iiter}, var.iterates.w{iiter}, var_a,var_a0);
        end
    end

% B1: H -> R^NxH^L
%B1 = @(var_a)   p1_mathcalB_transp(var_a);
function [ otp_var, otp_var0 ] = B1(var_a)
     [ otp_var, otp_var0 ] = p1_mathcalB_transp(var_a);
end

% B2: H -> Theta
B2 = @(var_a)   p2_mathcalB_transp(var.u,var_a);
    


%% Reverse mode (solely fixed point for now)

dCdw = opt.dCdw(yval,Xval,var.w);
[a,a0] = B1(dCdw);
b = B2(dCdw);

for ii= param.inner.itermax:-1:1
    b = A2(a,a0,ii) + b;
    [a,a0] = A1(a,a0,ii);
end




end




