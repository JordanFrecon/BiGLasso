function [ opt_upper ] = optimBLGS_setting( opt_lower,param )
%OPTIMBLGS_INIT Summary of this function goes here
%   Detailed explanation goes here



%% HEADER

if ~isfield(param.outer,'compObjective')
    param.outer.compObjective = false;
end



% - Optimizer & batch-size for stochastic descent
if strcmp(param.outer.optimizer,'GD')
    param.outer.batchSize = param.outer.nTasks;
elseif ~isfield(param.outer,'batchSize')
    param.outer.batchSize = 1;
    if ~isfield(param.outer,'optimizer')
        param.outer.optimizer   = 'SAGA';
    end
end




if ~isfield(param.outer,'projection')
    param.outer.projection = 'simplex';
end



if param.outer.linesearch
    if ~isfield(param.outer,'linesearch_c1')
        param.outer.linesearch_c1 = 0.1;
    end
    if ~isfield(param.outer,'linesearch_contraction')
        param.outer.linesearch_contraction = 0.8;
    end
end



%% - PROJECTION OF THE GROUP STRUCTURE ONTO THETA

switch param.outer.projection
    case 'simplex'
        proxl_v = @(varx) proj_unitSimplex(varx')';
    case 'posSphere'
        proxl_v = @(varx) proj_positiveUnitSphere(varx')';
    case 'box'
        proxl_v = @(varx) proj_1dset( varx', [0 1] )';
    otherwise
        disp('/!\ param.projection not well defined.')
        disp('either choose `simplex` or `positiveSphere`.')
end

proxl = @(varx) proxl_v(varx);



%% INITIALIZATION


    function [theta,hyperGrad_aux,plt] = optimBLGS_initialize()
        
        
        
        theta = proxl(  (1/param.inner.nGroups)*ones(param.inner.nFeatures,param.inner.nGroups) + .01*randn(param.inner.nFeatures,param.inner.nGroups) );
        plt   = optimBLGS_display_init(theta);
        
        
        switch param.outer.optimizer
            case 'GD'
                hyperGrad_aux.all   = NaN;
                hyperGrad_aux.mean  = NaN;
            case 'SGD'
                hyperGrad_aux.all   = NaN;
                hyperGrad_aux.mean  = NaN;
            case 'SAGD'
                hyperGrad_aux.all   = zeros(param.inner.nFeatures,param.inner.nGroups,param.inner.nTasks);
                hyperGrad_aux.mean  = NaN;
            case 'SAGA'
                hyperGrad_aux.all   = zeros(param.inner.nFeatures,param.inner.nGroups,param.inner.nTasks);
                hyperGrad_aux.mean  = zeros(param.inner.nFeatures,param.inner.nGroups);
        end
        

        
    end



%% HYPERGRADIENT

    function [hyperGrad,hyperGrad_aux,outerObj] = optimBLGS_hyperGradient(y,yval,X,Xval,theta,hyperGrad_aux)
        
        if ~param.outer.compObjective
            outerObj = NaN;
        end
        
        % Partial Hypergradients on a Batch of Tasks
        batch = datasample([1:param.inner.nTasks],param.outer.batchSize,'Replace',false);
        hyperGrad_batch = NaN(param.inner.nFeatures,param.inner.nGroups,param.outer.batchSize);
        W   = NaN(param.inner.nFeatures,param.outer.batchSize);
        
        for xtt=1:length(batch) %Better to use parfor if batchsize>1
            tt = batch(xtt);
            
            if param.outer.compObjective
                [wt,var] = optimGS_lower( y{tt}, X{tt}, theta, opt_lower, param.inner );
                W(:,xtt) = wt;
            else
                [~,var] = optimGS_lower( y{tt}, X{tt}, theta, opt_lower, param.inner );
            end
            hyperGradInstant            = optimGS_hypergradient_v2(y{tt},yval{tt},X{tt},Xval{tt},theta, opt_lower, param,var);
            hyperGrad_batch(:,:,xtt)    = hyperGradInstant;
        end
        
        if param.outer.compObjective
            outerObj = opt_upper.objective(yval(batch),Xval(batch),W);
        end
        
        
        
        
        switch param.outer.optimizer
            
            case 'GD'   %Gradient Descent (since batch size = T)
                hyperGrad_aux.all   = NaN;
                hyperGrad           = squeeze(mean(hyperGrad_batch,3));
                hyperGrad_aux.mean  = NaN;
                
            case 'GD+' %Gradient Descent for matrix case
                hyperGrad_aux.all   = NaN;
                hyperGrad           = hyperGradInstant;
                hyperGrad_aux.mean  = NaN;
                
            case 'SGD'  %Stochastic Gradient Descent
                hyperGrad_aux.all   = NaN;
                hyperGrad           = squeeze(mean(hyperGrad_batch,3));
                hyperGrad_aux.mean  = NaN;
                
            case 'SAGD' %Stochastic Averaged Gradient Descent
                for xtt=1:length(batch)
                    hyperGrad_aux.all(:,:,batch(xtt)) = hyperGrad_batch(:,:,xtt);
                end
                hyperGrad           = squeeze(mean(hyperGrad_aux.all,3));
                hyperGrad_aux.mean  = NaN;
                
            case 'SAGA' %Stochastic Averaged Gradient Descent with Variance Reduction
                hyperGrad_diff = NaN(param.inner.nFeatures,param.inner.nGroups);
                for xtt=1:length(batch)
                    hyperGrad_diff(:,:,xtt)             = hyperGrad_batch(:,:,xtt) - hyperGrad_aux.all(:,:,batch(xtt));
                    hyperGrad_aux.all(:,:,batch(xtt))   = hyperGrad_batch(:,:,xtt);
                end
                hyperGrad           = mean(hyperGrad_diff,3)    + hyperGrad_aux.mean;
                hyperGrad_aux.mean  = sum(hyperGrad_diff,3)/param.inner.nTasks   + hyperGrad_aux.mean;
                
        end
        
    end






%% DISPLAY

    function [plt] = optimBLGS_display_init(theta)
        if param.outer.dispOnline
            
            Fig=figure;
            plt=imagesc(theta);
            title('Iterate $\theta$','Interpreter','latex','fontsize',13)
            ylabel('Features','Interpreter','latex','fontsize',2)
            xlabel('Group indices','Interpreter','latex','fontsize',2)
            set(gca,'fontsize',15,'clim',[0 1])
            xticks([1:param.outer.nGroups]);
            colormap(flipud(gray))
            refreshdata(Fig,'caller');
            drawnow;
              
        else
            plt = [];
        end
    end

    function [] = optimBLGS_display_refresh(plt,theta)
        if param.outer.dispOnline
            plt.CData = theta;
            refreshdata(plt,'caller');
            drawnow limitrate
        end
    end



%% OUTPUT

opt_upper           = struct('hypergradient',@optimBLGS_hyperGradient,'proj_Theta',proxl,'initialize',@optimBLGS_initialize,'objective',@opt_lower.fidelity,'linesearch',@optimBLGS_linesearch);
opt_upper.display   = struct('init',@optimBLGS_display_init,'refresh',@optimBLGS_display_refresh);



end

