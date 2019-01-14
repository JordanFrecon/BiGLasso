function [ projy ] = proj_unitSimplex( y )
%See "Fast Projection onto the Simplex and the ?1 Ball", Laurent Condat, 2015

projy =  max(y-max((cumsum(sort(y,1,'descend'),1)-1)./(1:size(y,1))'),0);


end

