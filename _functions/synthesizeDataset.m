function [y,X,theta,w,Xw ] = synthesizeDataset( N, P, T, L, S,synth )
%SYNTHESIZEDATASET Summary of this function goes here
%   Detailed explanation goes here


%\\\\  Design matrices

if ~isfield(synth.design,'setting')
    synth.design.setting = 'general';
end

%\\\\ Features

if ~isfield(synth.features,'distrib')
    synth.features.distrib = 'normal';
end

if synth.features.param(2)==0
    synth.features.param(2) = eps;
end



%% GROUP STRUCTURE



nvar = P;


switch synth.groups.distrib
    case 'equal'
        assert(mod(nvar,L)==0,'nvar should be multiple of L for groups of equal size (synthesis convention)');
        tmp = repmat([1:L]',[1 nvar/L])';
        groupBelonging = tmp(1:end);
    case 'randn1'
        groupBelonging = sort(datasample([1:L],nvar,'Replace',true)');
    case 'randn2'
        r                   = randn(L,nvar);
        prob                = exp(r)./(sum(exp(r),1));
        [~,groupBelonging]  = max(prob);
        groupBelonging      = sort(groupBelonging);
    case 'rand'
        r                   = randn(L,1);
        groupSize           = round(P*exp(r)./(sum(exp(r))));
        groupSize(1)        = groupSize(1) + (nvar-sum(groupSize));
        groupBelonging      = 1+sum([1:nvar]>cumsum(groupSize));
end


theta = zeros(P,L);
for pp=1:P
    theta(pp,groupBelonging(pp)) = 1;
end



%% ORACLE FEATURES

w   = zeros(P,T);
w0  = randraw(synth.features.distrib,[synth.features.param],[P T]);
for tt=1:T
    for ss=1:S
        indi = randi(L);
        w(groupBelonging==indi,tt) = w0(groupBelonging==indi,tt);
    end
end



%% DESIGN MATRICES

for tt=1:T
    switch synth.design.setting
        case 'general'  %Regression with DIFFERENT regr matrices for train, val & test sets
            
            X_trtmp = randn(N,P);
            X_vltmp = randn(N,P);
            X_tstmp = randn(N,P);
            
        case 'same'  %Regression with SAME regr matrices for train, val & test sets
            
            X_trtmp = randn(N,P);
            X_vltmp = X_trtmp;
            X_tstmp = X_trtmp;
            
        case 'denoising' %Denoising
            assert(N==P)
            X_trtmp = eye(N);
            X_vltmp = eye(N);
            X_tstmp = eye(N);
    end
    

  
    if synth.design.renorm  % Renormalization
        for j = 1:P     
            X_trtmp(:,j) = X_trtmp(:,j)/norm(X_trtmp(:,j));
            X_vltmp(:,j) = X_vltmp(:,j)/norm(X_vltmp(:,j));
            X_tstmp(:,j) = X_tstmp(:,j)/norm(X_tstmp(:,j));
        end
        

    end
    
    
    X.trn{tt}       = X_trtmp;
    X.val{tt}       = X_vltmp;
    X.tst{tt}       = X_tstmp;
  
    Xw.trn(:,tt)    = X.trn{tt}*w(:,tt);
    Xw.val(:,tt)    = X.val{tt}*w(:,tt);
    Xw.tst(:,tt)    = X.tst{tt}*w(:,tt);
end


%% NOISY VECTORS OF OUTPUTS

if 0    %Noise variance independent of Var[Xw]
y.trn   = Xw.trn + randraw(synth.noise.distrib,synth.noise.param,[N T]);
y.val   = Xw.val + randraw(synth.noise.distrib,synth.noise.param,[N T]);
y.tst   = Xw.tst + randraw(synth.noise.distrib,synth.noise.param,[N T]);

else    %Noise variance proportional to Var[Xw]: same signal/noise ratio for all tasks
y.trn   = Xw.trn + ones(N,1)*std(Xw.trn,1).*randraw(synth.noise.distrib,synth.noise.param,[N T]);
y.val   = Xw.val + ones(N,1)*std(Xw.val,1).*randraw(synth.noise.distrib,synth.noise.param,[N T]);
y.tst   = Xw.tst + ones(N,1)*std(Xw.tst,1).*randraw(synth.noise.distrib,synth.noise.param,[N T]);
    
end




%from array to cell
ytmp = y;
clear y;
for tt=1:T
    y.trn{tt} = ytmp.trn(:,tt);
    y.val{tt} = ytmp.val(:,tt);
    y.tst{tt} = ytmp.tst(:,tt);
end


end

