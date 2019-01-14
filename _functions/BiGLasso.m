function [ theta,otp ] = BiGLasso( y,X,lambda_reg,EPS, param )
% (January, 14th, 2019)
%
% Author:
% Jordan Frecon (jordan.frecon@iit.it) 
% 
% Contributors:
% Saverio Salzo (saverio.salzo@iit.it)
% Massimiliano Pontil (massimiliano.pontil@iit.it) 
%
% This software is governed by the CeCILL license under French law and
% abiding by the rules of distribution of free software.  You can  use,
% modify and/ or redistribute the software under the terms of the CeCILL
% license as circulated by CEA, CNRS and INRIA at the following URL
% "http://www.cecill.info".
% 
% As a counterpart to the access to the source code and  rights to copy,
% modify and redistribute granted by the license, users are provided only
% with a limited warranty  and the software's author,  the holder of the
% economic rights,  and the successive licensors  have only  limited
% liability.
% 
% In this respect, the user's attention is drawn to the risks associated
% with loading,  using,  modifying and/or developing or reproducing the
% software by the user in light of its specific status of free software,
% that may mean  that it is complicated to manipulate,  and  that  also
% therefore means  that it is reserved for developers  and  experienced
% professionals having in-depth computer knowledge. Users are therefore
% encouraged to load and test the software's suitability as regards their
% requirements in conditions enabling the security of their systems and/or
% data to be ensured and,  more generally, to use and operate it in the
% same conditions as regards security.
% 
% The fact that you are presently reading this means that you have had
% knowledge of the CeCILL license and that you accept its terms.
%
%--------------------------------------------------------------------------
% BiGLasso: Bilevel learning of Group Lasso Structure  
%                                                                         
% For theoretical aspects please refer to :                               
%
% #J. Frecon, S. Salzo, and M. Pontil
%  Bilevel Learning of Group Lasso Structure
%  NIPS 2018                            
%--------------------------------------------------------------------------
%
% [ theta,otp ] = BiGLasso( y,X,lambda_reg,EPS, param )
%
% INPUTS:
%           - Vector of output 'y' cell of size 'T' with fields:
%               - 'trn' for training set
%               - 'val' for validation set
%           - Design matrix 'X' cell of size 'T'  with fields:
%               - 'trn' for training set
%               - 'val' for validation set
%           - Regularization parameter 'lambda_reg'
%           - Regularization parameter 'EPS' (default: 10^-3)
%           - Parameters of the algorithm 'param' with fields
%               - (Inner problem)
%                   'inner.itermax'         number of inner iterations (default: 500)
%                   'inner.compObjective'   if true, compute & store inner-objective
%               - (Outer problem)
%                   'outer.compObjective'   if true, compute & store inner-objective
%                   'outer.fixedPointHG'    if true, accelerated scheme for computing the hypergradient
%                   'outer.projection'      projection on space Theta (default 'simplex'. See also 'posSphere' and 'box')
%                   'outer.stepsize'        outer step size
%                   'outer.itermax'         number of outer iterations
%                   'outer.nGroups'         number of groups
%                   'outer.batchSize'       batch size for stochastic optimization (see below) 
%                   'outer.optimizer'       outer optimizer (default 'SAGA'. See also, 'GD', 'SGD' and 'SAGD')
%                   'outer.displayOnline':  Online display of groups('false' or 'true')
%
% OUTPUTS:
%           - Recovered groups 'theta'
%           - Struct 'otp' with fields 'objective' giving (if asked) the
%           upper objective wrt iterations

otp = struct;

if ~isfield(param,'misc')
    param.misc = struct;
end

if ~isfield(param.misc,'saveiter')
    param.misc.saveiter = Inf;
end



%% Header needed for upper-level problem


if ~isfield(param,'outer')
    param.outer = struct;
end

if ~isfield(param.outer,'compObjective')
    param.outer.compObjective = false;
end

if ~isfield(param.outer,'linesearch')
    param.outer.linesearch = false;
end

if ~isfield(param.outer,'dispOnline')
    param.outer.dispOnline = true;
end

% Fixed-point approach for computing the hypergradient
if ~isfield(param.outer,'fixedPointHG')
    param.outer.fixedPointHG = true;
end

if ~isfield(param.outer,'stepsize')
    param.outer.stepsize    = 1/(5*1.2)*mean(cellfun('length',y.val)); %correspond to 1/(5*T*Lips) with Lips=1.2/T
end
    
% Parameters of the model
param.outer.nTasks      = size(y.val,2);
param.outer.nFeatures   = size(X.val{1},2);
if ~isfield(param.outer,'nGroups')
    param.outer.nGroups   = param.outer.nFeatures;
end

% Variables
if param.outer.compObjective
    upperObjective = NaN(1,param.outer.itermax);
end

%% Header needed for lower-level problem

if ~isfield(param,'inner')
    param.inner = struct;
end

% Saving iterates (needed if not using the fixed-point approach)
if ~isfield(param.inner,'saveIterates') && param.outer.fixedPointHG
    param.inner.saveIterates = false;
elseif ~param.outer.fixedPointHG
    param.inner.saveIterates = true;
end

% Compute objective
if ~isfield(param.inner,'compObjective')
    param.inner.compObjective = false;
end

% Number of iterations
if ~isfield(param.inner,'itermax')
    param.inner.itermax = 500;
end

% Parameters of the model
param.inner.lambda_reg  = lambda_reg;
param.inner.eps         = EPS;
param.inner.nTasks      = size(y.trn,2);
param.inner.nObs        = size(X.trn{1},1);
param.inner.nFeatures   = size(X.trn{1},2);
param.inner.nGroups     = param.outer.nGroups;


%% Main body


% Optimizers of the lower and upper problem 
opt_lower = optimGS_setting( param.inner );
opt_upper = optimBLGS_setting( opt_lower,param );


% Initialization of the upper optimizer
[theta,hyperGrad_aux,plt] = opt_upper.initialize();



for iter=1:param.outer.itermax
    
    
    % Gradient step
    [hyperGrad,hyperGrad_aux,upperObj] = opt_upper.hypergradient(y.trn,y.val,X.trn,X.val,theta,hyperGrad_aux);
    theta = opt_upper.proj_Theta( theta - param.outer.stepsize*hyperGrad);
    
    % Display groups [optional]
    opt_upper.display.refresh(plt,theta)
    
    % Objective & Misc [optimal]
    if param.outer.compObjective
        upperObjective(iter) = upperObj;
    end
    if mod(iter,param.misc.saveiter)==0
        save(strcat(param.misc.savename,'_iter',num2str(iter),'.mat'));
    end
    
   
    
end

%% OUTPUT

if param.outer.compObjective
   otp.objective = upperObjective; 
end



end

