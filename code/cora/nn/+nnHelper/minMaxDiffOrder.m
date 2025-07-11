function [diffl,diffu] = minMaxDiffOrder(coeffs, l, u, f, der1l,der1u)
% minMaxDiffOrder - compute the maximum and the minimum difference between the activation
% function and a polynomial fit
%
% Syntax:
%    L = nnHelper.minMaxDiffOrder(coeffs, l, u, f, der1)
%
% Inputs:
%    coeffs - coefficients of polynomial
%    l - lower bound of input domain
%    u - upper bound of input domain
%    f - function handle of activation function
%    der1 - bounds for derivative of activation functions
%
% Outputs:
%    [diffl,diffu] - interval bounding the lower and upper error
%
% Other m-files required: none
% Subfunctions: none
% MAT-files required: none
%
% See also: -

% Authors:       Tobias Ladner
% Written:       28-March-2022
% Last update:   31-August-2022 (adjust tol)
%                30-May-2023 (output bounds)
%                02-May-2025 (added maxPoints)
% Last revision: ---

% ------------------------------ BEGIN CODE -------------------------------

tol = 1e-4;
minPoints = 1e4;
maxPoints = 5e9; % requires 40GB

if l == u
    % compute exact result directly
    diff = f(l);
    yp = polyval(coeffs, l);
    diff = diff-yp;
    diffl = diff;
    diffu = diff;
    return
end

% calculate bounds for derivative of polynomial
[der2l,der2u] = nnHelper.getDerInterval(coeffs, l, u);

% der = der1 - -der2; % '-' as we calculate f(x) - p(x)
der = max(abs([ ...
    der1l - -der2l; ...
    der1l - -der2u; ...
    der1u - -der2l; ...
    der1u - -der2u; ...
]));

% determine number of points to sample
dx = tol / der;
reqPoints = ceil((u - l)/dx);
numPoints = min(max(reqPoints, minPoints), maxPoints);

% re-calculate tolerance with number of used points
dx = (u-l)/numPoints;
tol = der * dx;

% sample points
x = linspace(l, u, numPoints);
x = [l, x, u]; % add l, u in case x is empty (der = 0)
diff = f(x) - polyval(coeffs, x);

% find bounds
diffl = min(diff)-tol;
diffu = max(diff)+tol;
end

% ------------------------------ END OF CODE ------------------------------
