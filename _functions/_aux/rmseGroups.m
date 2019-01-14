function [ valinf, thetaBest] = rmseGroups( theta,thetastar )
%RMSEGROUPS Summary of this function goes here
%   Find the permutation of groups to minimize the relative MSE

L = size(theta,1);
allperms = perms([1:L]);

valinf = +Inf;
bestPerm = NaN;
for ii=1:size(allperms,1)
    
    val=norm(theta(allperms(ii,:),:)-thetastar,'fro')^2/norm(thetastar,'fro')^2;
    if val < valinf
        valinf = val;
        bestPerm = allperms(ii,:);
    end
    
end

thetaBest = theta(bestPerm,:);

end

