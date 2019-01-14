function [ b ] = optimGS_hypergradient_v2(y, yval,X, Xval, theta, opt, param, var)
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

if param.outer.fixedPointHG
    [ FP_var, FP_fun ] = optimGS_hypergradient_fixedPoint( opt, param );
    
    FP_n2p  = cell(param.inner.nGroups,1);
    FP_n2ps = cell(param.inner.nGroups,1);
    for xll=1:param.inner.nGroups
        if ndims(var.u) == 3
            FP_n2p{xll}  = FP_var.nabla2_phi(var.u(:,:,xll));
            FP_n2ps{xll} = FP_var.nabla2_phi_star(var.v(:,:,xll));
        else
            FP_n2p{xll}  = FP_var.nabla2_phi(var.u(:,xll));
            FP_n2ps{xll} = FP_var.nabla2_phi_star(var.v(:,xll));
        end
            
    end
    
    FP_nabla2_phi       = @(varin,indl)  FP_fun.gen_nabla2(FP_n2p,varin,indl);
    FP_nabla2_phi_star  = @(varin,indl)  FP_fun.gen_nabla2(FP_n2ps,varin,indl);
    
end



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
    function [otp_var] = p1_mathcalB_transp(var_a)
            prodvar     = prodCustom(var.nsd_mat,var_a);
            otp_var     = -opt.opA(theta,prodvar);
    end

    % H -> Theta
    function [otp_var] = p2_mathcalB_transp(var_u,var_a)
            otp_var     = -opt.opA(var_u, prodCustom(var.nsd_mat,var_a));
    end




%% Partial derivatives of mapping A 

    % R^NxH^L -> R^NxH^L
    function [otp_var] = p1_mathcalA_transp(var_u,var_v,var_a)
        vartmp  = nabla2_Phi_star( var_v,var_a );
        otp_var = nabla2_Phi(var_u,vartmp) + var.stepsize*p1_mathcalB_transp( opt.opA_star(theta,vartmp) );        
    end

    % R^NxH^L -> Theta
    function [otp_var] = p2_mathcalA_transp(var_u,var_v,var_w,var_a)
        vartmp = nabla2_Phi_star( var_v,var_a );
        tmp = opt.opA( vartmp, var_w );
        otp_var = var.stepsize*p2_mathcalB_transp( var_u, opt.opA_star(theta,vartmp) ) + var.stepsize*tmp;
  
    end



%% Fixed-point vs. standard algorithm (can save some computation time here!!)

% A1: R^NxH^L -> R^NxH^L
    function [otp_var] = A1(var_a,iiter)    %(Might be too costly)
        if param.outer.fixedPointHG
            [otp_var] = p1_mathcalA_transp(var.u, var.v, var_a);
        else
            [otp_var] = p1_mathcalA_transp(var.iterates.u{iiter}, var.iterates.v{iiter}, var_a);
        end
    end

% A2: R^NxH^L -> Theta
    function [otp_var] = A2(var_a,iiter)
        if param.outer.fixedPointHG
            [otp_var] = p2_mathcalA_transp(var.u, var.v, var.w, var_a);
        else
            [otp_var] = p2_mathcalA_transp(var.iterates.u{iiter}, var.iterates.v{iiter}, var.iterates.w{iiter}, var_a);
        end
    end

% B1: H -> R^NxH^L
%B1 = @(var_a)   p1_mathcalB_transp(var_a);
function [ otp_var ] = B1(var_a)
     [ otp_var] = p1_mathcalB_transp(var_a);
end

% B2: H -> Theta
B2 = @(var_a)   p2_mathcalB_transp(var.u,var_a);
    


%% Reverse mode

dCdw = opt.dCdw(yval,Xval,var.w);
[a] = B1(dCdw);
b = B2(dCdw);


for ii= param.inner.itermax:-1:1
    b = A2(a,ii) + b;
    [a] = A1(a,ii);
end




end




