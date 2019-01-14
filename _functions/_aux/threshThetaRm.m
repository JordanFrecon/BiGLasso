function [ ThetaThreshRm ] = threshThetaRm( Theta )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

[~, ind] = max(Theta,[],1);
%ind=find(Theta==max(Theta,[],1));

[L,P] = size(Theta);
ThetaThresh = zeros(L,P);
for pp=1:P
ThetaThresh(ind(pp),pp)=1;
end



indb= find(sum(ThetaThresh == zeros(1,P),2) == P);
ThetaThreshRm = ThetaThresh(setdiff([1:L],indb),:);

end

