function [ valsup ] = normGroupOperator(linOp,theta)
%[ valsup ] = normGroupOperator(theta,linOp)
%   Compute the l_2^2 norm of the linear operator:
%       A : w -> (theta(:,1).*w, ..., theta(:,end).*w)
%
% Version: 18 October 2018

Nreal   = 10;
[P]     = size(theta,1);
valsup  = -Inf;


for ii=1:Nreal
    w = randn(P,1);
    w = w/norm(w);

    valsup = max(valsup,norm(linOp(theta,w),'fro')^2);
end



end
