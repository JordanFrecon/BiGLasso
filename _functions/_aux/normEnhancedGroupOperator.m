function [ valsup ] = normEnhancedGroupOperator(linOp1,linOp2,theta,y,X)
%NORMATHETAY Summary of this function goes here
%   Detailed explanation goes here
Nreal   = 10;
[P]     = size(theta,1);
valsup  = -Inf;


for ii=1:Nreal
    w = randn(P,1);
    w = w/norm(w);

    norm1 = norm(linOp1(theta,w),'fro')^2;
    norm2 = norm(linOp2(y,X,w),'fro')^2; 
    valsup = max(valsup,norm1+norm2);
end

valsup = valsup*(1.005);


end
