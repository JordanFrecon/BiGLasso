function [ Y ] = compTestLabelsCell( X,D )
%COMPTESTLABELS Summary of this function goes here
%   Detailed explanation goes here


Nclasses = size(X,2);
Ntst = size(D{1},1);

Y = cell(1,Nclasses);

for tt=1:Nclasses
    Y{tt} = D{tt}*X(:,tt);
end



end

