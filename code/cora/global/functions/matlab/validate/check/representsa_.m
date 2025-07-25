function [res,S] = representsa_(S,type,varargin)
% representsa_ - this function overloads the contSet/representsa function
%     for'numeric' variables
%
% Syntax:
%    res = representsa_(S,type)
%    res = representsa_(S,type,tol)
%    [res,S] = representsa_(S,type)
%    [res,S] = representsa_(S,type,tol)
%
% Inputs:
%    S - numeric vector/matrix
%    type - char array
%    tol - (optional) tolerance
%
% Outputs:
%    res - true/false
%    S - converted set
%
% Example: 
%    S = [];
%    representsa_(S,'emptySet')
%
% Other m-files required: none
% Subfunctions: none
% MAT-files required: none
%
% See also: contSet/representsa, isempty, contSet/isemptyobject

% Authors:       Mark Wetzlinger
% Written:       21-July-2023
% Last update:   21-October-2023 (TL, cellfun)
% Last revision: ---

% ------------------------------ BEGIN CODE -------------------------------

if nargin < 3
    tol = 1e-12;
else
    tol = varargin{1};
    varargin = varargin(2:end);
end

if iscell(S)
    res = cellfun(@(S) representsa_(S,type,tol,varargin{:}),S);
    return
end

% check if input argument is numeric
if ~isnumeric(S)
    throw(CORAerror('CORA:wrongValue','first',...
        'has to be numeric or contSet class.'));
end

% dimensions
[n,s] = size(S);

% only a single point?
isPoint = s == 1;

switch type
    case 'origin'
        res = isPoint && all(withinTol(S,0,tol));

    case 'convexSet'
        res = isPoint;

    case 'point'
        res = isPoint;

    case 'capsule'
        % max. two points
        res = s <= 2;
        % note: more points if collinear...

    case 'halfspace'
        % point clouds cannot be unbounded
        res = false;
    
    case 'hyperplane'
        % point clouds cannot be unbounded
        res = false;

    case 'emptySet'
        res = isempty(S);

    case 'fullspace'
        % point clouds cannot be unbounded
        res = false;

        
    case {'conHyperplane','conPolyZono','conZonotope','ellipsoid','interval','levelSet','polytope','polyZonotope','probZonotope','zonoBundle','zonotope','parallelotope'}
        % unsupported set representations
        throw(CORAerror('CORA:notSupported'));

    otherwise
        % throw error
        throw(CORAerror('CORA:wrongValue','second',...
            'has to be an admissible type.'));

end

% ------------------------------ END OF CODE ------------------------------
