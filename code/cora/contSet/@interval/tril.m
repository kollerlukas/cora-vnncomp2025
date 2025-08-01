function res = tril(I,varargin)
% tril - gets lower triangular part of I
%
% Syntax:
%    res = tril(I)
%    res = tril(I,K)
%
% Inputs:
%    I - interval object
%    K - (see built-in tril for matrices)
%
% Outputs:
%    res - lower triangular interval object
%
% Example: 
%    I = interval(-ones(2), ones(2));
%    tril(I)
%
% Other m-files required: none
% Subfunctions: none
% MAT-files required: none
%
% See also: ---

% Authors:       Victor Gassmann
% Written:       12-October-2022 
% Last update:   ---
% Last revision: ---

% ------------------------------ BEGIN CODE -------------------------------

narginchk(1,2);

if nargin == 1
    K = 0;
elseif nargin == 2
    K = varargin{1};
    inputArgsCheck({{K,'att',{'double'},{'scalar',@(K) K>=0}}});
end

res = I;
res.inf = tril(res.inf,K);
res.sup = tril(res.sup,K);

% ------------------------------ END OF CODE ------------------------------
