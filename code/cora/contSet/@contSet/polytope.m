function P = polytope(S,varargin)
% polytope - conversion to polytope objects
%
% Syntax:
%    P = polytope(S)
%
% Inputs:
%    S - contSet object
%
% Outputs:
%    P - polytope object
%
% Example:
%    Z = zonotope([1;1],[1 1 1; 1 -1 0]);
%    P = polytope(Z);
%
% Other m-files required: none
% Subfunctions: none
% MAT-files required: none
%
% See also: -

% Authors:       Mark Wetzlinger
% Written:       23-December-2023
% Last update:   ---
% Last revision: ---

% ------------------------------ BEGIN CODE -------------------------------

% is overridden in subclass if implemented; throw error
throw(CORAerror('CORA:noops',S));

% ------------------------------ END OF CODE ------------------------------
