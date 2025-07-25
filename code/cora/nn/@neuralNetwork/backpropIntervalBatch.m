function [gl, gu] = backpropIntervalBatch(nn, gl, gu, options, varargin)
% backpropIntervalBatch - compute the backpropagation for the previous input
%    with batches of intervals
%
% Syntax:
%    [gl,gu] = nn.backpropIntervalBatch(gl, gu, options, idxLayer)
%
% Inputs:
%    nn - neuralNetwork
%    gl - gradient of lower bounds
%    gu - gradient of upper bounds
%    options - training parameters
%    idxLayer - indices of layers that should be evaluated
%    updateWeights - boolean, if gradient should be stored
%
% Outputs:
%    c, G - zonotope gradient w.r.t the input
%
% Other m-files required: none
% Subfunctions: none
% MAT-files required: none
%
% See also: neuralNetwork/backprop

% Authors:       Lukas Koller
% Written:       22-January-2024
% Last update:   22-February-2024 (merged options.nn, moved input storage handling from layer to network)
% Last revision: ---

% ------------------------------ BEGIN CODE -------------------------------

[idxLayer,updateWeights] = setDefaultValues({1:length(nn.layers),true}, varargin);

for i = flip(idxLayer)
    layeri = nn.layers{i};
    % Retrieve stored input
    input = layeri.backprop.store.input;
    l = input.inf;
    u = input.sup;
    [gl, gu] = layeri.backpropIntervalBatch(l, u, gl, gu, options, updateWeights);
end

end

% ------------------------------ END OF CODE ------------------------------
