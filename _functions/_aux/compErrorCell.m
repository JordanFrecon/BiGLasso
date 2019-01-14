function [mse] = compErrorCell(cell1,cell2)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here


T = length(cell1);
error = zeros(1,T);
varsize = zeros(1,T);
for tt=1:T
    error(tt)=norm(cell1{tt} - cell2{tt})^2;
    varsize(tt) = length(cell1{tt});
end

mse = sum(error)/sum(varsize);

end

