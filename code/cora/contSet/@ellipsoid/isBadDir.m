function res = isBadDir(E1,E2,L)
% isBadDir - checks if specified directions are bad directions for
%    the Minkowski difference of E1 and E2
%
% Syntax:
%    res = isBadDir(E1,E2,L)
%
% Inputs:
%    E1 - ellipsoid object
%    E2 - ellipsoid object
%    L  - (n x N) matrix of directions, where n is the set dimension, and N
%         is the number of directions to check
%
% Outputs:
%    res - true/false for each direction
%
% Other m-files required: none
% Subfunctions: none
% MAT-files required: none
%
% See also: none

% Authors:       Victor Gassmann
% Written:       13-March-2019
% Last update:   10-June-2022
%                04-July-2022
% Last revision: ---

% ------------------------------ BEGIN CODE -------------------------------

% input check
inputArgsCheck({{E1,'att','ellipsoid','scalar'};
                {E2,'att','ellipsoid','scalar'};
                {L,'att','numeric',{@(L) size(L,1) == dim(E1)}}});

% check dimension
equalDimCheck(E1,E2);

TOL = min(E1.TOL,E2.TOL);
[~,D] = simdiag(E1.Q,E2.Q,TOL);
r = 1/max(diag(D));
res = true(1,size(L,2));
for i=1:size(L,2)
    l = L(:,i);
    res(i) = sqrt(l'*E1.Q*l)/sqrt(l'*E2.Q*l) > r+TOL;
end

% ------------------------------ END OF CODE ------------------------------
