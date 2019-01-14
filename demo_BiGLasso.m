%--------------------------------------------------------------------------
% demo_BiGLasso: Demo for the Bilevel learning of Group Lasso Structure  
%                                                                         
% For theoretical aspects please refer to :                               
%
% #J. Frecon, S. Salzo, and M. Pontil
%  Bilevel Learning of Group Lasso Structure
%  NIPS 2018                            
%--------------------------------------------------------------------------


clearvars;
close all;

%\\\\\ Synthesis parameters
N                           = 20;          %Number of samples
P                           = 50;          %Number of features
T                           = 500;         %Number of tasks
L                           = 5;           %Number of groups
S                           = 1;           %Number of non-zero groups per task
synth.groups.distrib        = 'equal';     %Group size ('equal' or 'rand')
synth.design.setting        = 'general';   
synth.design.renorm         = true;
synth.noise.distrib         = 'normal';       
synth.noise.param           = [0 .2];       %~Normal with mean 'param(1)' and var 'param(2)'
synth.features.distrib      = 'normal'; 
synth.features.param        = [0 1];        %~Normal with mean 'param(1)' and var 'param(2)'


%\\\\\ Bilevel Parameters
lambda_reg                  = .01;          %Regularization parameter
EPS                         = 10^(-3);  

%\\\\\ Bilevel Parameters
% - Inner problem
param.inner.itermax         = 500;
param.inner.compObjective   = true;
% - Outer problem
param.outer.compObjective   = true;         %Compute objective at each iteration
param.outer.fixedPointHG    = true;         %Accelerated scheme for computing the hypergradient
param.outer.projection      = 'simplex';    %Space Theta (see also, 'posSphere' and 'box')
%param.outer.stepsize        = 2;           %Outer step-size (optional)
param.outer.itermax         = 250;          %Number of outer iterations
param.outer.nGroups         = L;            %Number of groups
param.outer.batchSize       = 1;            %Batch size for stochastic optimization (see below) 
param.outer.optimizer       = 'SAGA';       %Outer optimizer (see also, 'GD', 'SGD' and 'SAGD')
param.outer.displayOnline   = true;         %Online display of groups




%% SYNTHESIS 

[y,X,thetastar,wstar,Xwstar ] = synthesizeDataset( N, P, T, L, S,synth);

figure(101);clf;
subplot(121);
imagesc(thetastar);
xlabel('Groups','Interpreter','latex','fontsize',2)
ylabel('Features','Interpreter','latex','fontsize',2)
title('Oracle $\theta^*$','Interpreter','latex','fontsize',2)
set(gca,'fontsize',15,'clim',[0 1])
colorbar;
colormap(parula);
c=subplot(122);
wlim = max([abs(min(min(wstar))),abs(max(max(wstar)))]);
imagesc(wstar);
colorbar;
colormap(parula);
xlabel('Tasks','Interpreter','latex','fontsize',2)
ylabel('Features','Interpreter','latex','fontsize',2)
title('Oracle $w^*$','Interpreter','latex','fontsize',2)
set(gca,'fontsize',15,'CLim',[-wlim wlim])


%% ANALYSIS

thetaHat = BiGLasso( y,X,lambda_reg,EPS,param );



