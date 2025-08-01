function int = minus(minuend,subtrahend)
% minus - Overloaded '-' operator for STL intervals
%
% Syntax:
%    int = minus(minuend,subtrahend)
%
% Inputs:
%    minuend - stlInterval object
%    subtrahend - stlInterval object
%
% Outputs:
%    int - interval
%
% Example:
%    minuend = stlInterval(1,2);
%    subtrahend = stlInterval(0,1,false,false);
%    minuend - subtrahend
%
% Other m-files required: none
% Subfunctions: none
% MAT-files required: none
%
% See also: none

% Authors:       Florian Lercher
% Written:       06-February-2024
% Last update:   ---
% Last revision: ---

% ------------------------------ BEGIN CODE -------------------------------

if representsa(minuend,'emptySet') || representsa(subtrahend,'emptySet')
    int = stlInterval();
    return;
end

if isa(subtrahend,'stlInterval')
    % subtract another stlInterval
    lb = minuend.lower - subtrahend.upper;
    ub = minuend.upper - subtrahend.lower;
    lc = minuend.leftClosed && subtrahend.rightClosed;
    rc = minuend.rightClosed && subtrahend.leftClosed;
elseif isa(subtrahend,'interval')
    % subtract a regular CORA interval
    if dim(subtrahend) ~= 1
        % stlIntervals are always 1D
        throw(CORAerror('CORA:dimensionMismatch',minuend,subtrahend));
    end
    lb = minuend.lower - supremum(subtrahend);
    ub = minuend.upper - infimum(subtrahend);
    % CORA intervals are always closed
    lc = minuend.leftClosed;
    rc = minuend.rightClosed;
elseif isnumeric(subtrahend)
    % subtract scalar
    lb = minuend.lower - subtrahend;
    ub = minuend.upper - subtrahend;
    lc = minuend.leftClosed;
    rc = minuend.rightClosed;
else
    throw(CORAerror('CORA:noops',minuend,subtrahend));
end

% clamp lower bound to 0 if it is negative
if lb < 0
    lb = 0;
    lc = true;
end

int = stlInterval(lb,ub,lc,rc);

% ------------------------------ END OF CODE ------------------------------
