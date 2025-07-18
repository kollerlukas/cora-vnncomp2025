classdef (Abstract) nnLayer < matlab.mixin.Copyable
% nnLayer - abstract class for nn layers
%
% Syntax:
%    nnLayer
%
% Inputs:
%    name - name of the layer, defaults to type
%
% Outputs:
%    obj - generated object
%
% Other m-files required: none
% Subfunctions: none
% MAT-files required: none
%
% See also: neuralNetwork

% Authors:       Tobias Ladner
% Written:       28-March-2022
% Last update:   15-February-2023 (re-organized, simplified)
%                24-February-2023 (added options to all evaluate func)
%                31-July-2023 (LK, modified 'backprop' signature)
%                02-August-2023 (LK, zonotope batch-eval & -backprop)
%                22-January-2022 (LK, functions for IBP-based training)
%                24-February-2023 (added options to all evaluate func)
%                18-August-2024 (MW, updateGradient -> policy grad)
% Last revision: 10-August-2022 (renamed)

% ------------------------------ BEGIN CODE -------------------------------

properties
    name = [];          % name of the layer

    % stores the size of the input. Needed to propagate images.
    inputSize = [];

    % sensitivity of input
    sensitivity = [];

    % for backpropagation
    backprop = struct('store', struct)
end

methods
    function obj = nnLayer(name)
        if nargin < 1 || isempty(name)
            name = obj.getDefaultName();
        end

        obj.name = name;
        obj.inputSize = [];
    end

    function outputSize = computeSizes(obj, inputSize)
        obj.inputSize = inputSize;
        outputSize = getOutputSize(obj, inputSize);
    end

    function name = getDefaultName(obj)
        % get name from class name 
        name = class(obj);
        if startsWith(name, 'nn')
            name = name(3:end);
        end
        name = strrep(name,'Layer','');

        % add unique number to name
        name = sprintf("%s_%i", name, nnLayer.getCount());
    end

    function str = getLayerInfo(obj)
        str = sprintf('%-30s %-30s', class(obj), obj.name);
        if ~isempty(obj.inputSize)
            input = obj.inputSize;
            input = join(string(input), 'x');
            output = obj.getOutputSize(obj.inputSize);
            output = join(string(output), 'x');

            str = str + sprintf(" %+10s -> %-10s", input, output);
        else
            str = str + sprintf("\t(Input size not set)");
        end

        % additional information
        if isa(obj, 'nnGNNGlobalPoolingLayer')
            str = str + sprintf('(pooling across nodes (%s))', obj.type);
        elseif isa(obj, 'nnGNNLayer')
            str = str + "(per node)";
        end
    end
end

% evaluate ----------------------------------------------------------------

methods
    function r = evaluate(obj, varargin)
        % wrapper to propagate a single layer
        nn = neuralNetwork({obj});
        r = nn.evaluate(varargin{:});
    end

    function castWeights(obj, x)
        % Callback when data type of learnable parameters was changed
    end
end

methods(Abstract, Access = {?nnLayer, ?neuralNetwork})
    % numeric
    r = evaluateNumeric(obj, input, options)
end

methods (Access = {?nnLayer, ?neuralNetwork})
    % evaluation functions for each input set
    % will be overwritten in subclasses if supported by that layer

    % sensitivity
    function S = evaluateSensitivity(obj, S, options)
        throw(CORAerror('CORA:nnLayerNotSupported', obj, 'evaluate/sensitivity'))
    end

    % interval
    function bounds = evaluateInterval(obj, bounds, options)
        bounds = obj.evaluateNumeric(cat(3,bounds.inf,bounds.sup), options);
        bounds = interval(bounds(:,:,1),bounds(:,:,2));
    end

    % zonotope/polyZonotope
    function [c, G, GI, E, id, id_, ind, ind_] = evaluatePolyZonotope(obj, c, G, GI, E, id, id_, ind, ind_, options)
        throw(CORAerror('CORA:nnLayerNotSupported', obj, 'evaluate/polyZonotope'))
    end

    function [c, G] = evaluateZonotopeBatch(obj, c, G, options)
        throw(CORAerror('CORA:nnLayerNotSupported', obj, 'evaluate/zonotope (batch)'))
    end

    % taylm
    function r = evaluateTaylm(obj, input, options)
        throw(CORAerror('CORA:nnLayerNotSupported', obj, 'evaluate/taylm'))
    end

    % conZonotope
    function [c, G, C, d, l, u] = evaluateConZonotope(obj, c, G, C, d, l, u, options)
        throw(CORAerror('CORA:nnLayerNotSupported', obj, 'evaluate/conZonotope'))
    end

    % backprop ------------------------------------------------------------

    function grad_in = backpropNumeric(obj, input, grad_out, options, updateWeights)
        throw(CORAerror('CORA:nnLayerNotSupported', obj, 'backprop/numeric'))
    end

    function [l,u] = backpropIntervalBatch(obj, l, u, gl, gu, options, updateWeights)
        throw(CORAerror('CORA:nnLayerNotSupported', obj, 'backprop/interval'))
    end

    function [c,G] = backpropZonotopeBatch(obj, c, G, gc, gG, options, updateWeights)
        throw(CORAerror('CORA:nnLayerNotSupported', obj, 'backprop/zonotope (batch)'))
    end
end

% Auxiliary functions -----------------------------------------------------

methods
    function layerStruct = exportAsStruct(obj)
       layerStruct = struct;
       layerStruct.type = class(obj);
       layerStruct.name = obj.name;
       layerStruct.fields = getFieldStruct(obj);
    end

    function fieldStruct = getFieldStruct(obj)
        throw(CORAerror('CORA:nnLayerNotSupported',obj,'field struct'));
    end
end

methods (Static)
    function layer = importFromStruct(layerStruct)
        fieldStruct = layerStruct.fields;

        % init layer
        switch(layerStruct.type)
            % linear ---
            case 'nnLinearLayer'
                W = fieldStruct.W;
                W = reshape(W,reshape(fieldStruct.size_W,1,[]));
                b = fieldStruct.b;
                b = reshape(b,fieldStruct.size_W(1),[]);
                layer = nnLinearLayer(W,b);
                d = fieldStruct.d;
                if ~isempty(d)
                    d = interval(d.inf, d.sup);
                    d = reshape(d,fieldStruct.size_W(1),[]);
                end
                layer.d = d;
            case 'nnElementwiseAffineLayer'
                scale = fieldStruct.scale;
                offset = fieldStruct.offset;
                layer = nnElementwiseAffineLayer(scale, offset);
                % nonlinear ---
            case 'nnReLULayer'
                layer = nnReLULayer();
            case 'nnLeakyReLULayer'
                alpha = fieldStruct.alpha;
                layer = nnLeakyReLULayer(alpha);
            case 'nnSigmoidLayer'
                layer = nnSigmoidLayer();
            case 'nnTanhLayer'
                layer = nnTanhLayer();
                % special ---
            case 'nnReshapeLayer'
                idx_out = fieldStruct.idx_out;
                layer = nnReshapeLayer(idx_out);
            otherwise
                throw(CORAerror('CORA:specialError',sprintf('Unknown layer type: %s',layerStruct.type)))
        end

        % update name            
        layer.name = layerStruct.name;

        % network reduction: read merged neurons if available
        if isfield(layerStruct.fields,'merged_neurons')
            layer.merged_neurons = layerStruct.fields.merged_neurons;
        end
    end
end

methods (Abstract)
    [nin, nout] = getNumNeurons(obj)
    outputSize = getOutputSize(obj, inputSize)
end

methods (Access=protected)
    function checkInputSize(obj)
        % checks if input size is set
        if isempty(obj.inputSize)
            throw(CORAerror("CORA:notSupported", 'Input size has to be set to compute CNNs. See neuralNetwork/setInputSize.'))
        end
    end

    function updateGrad(obj, name, grad_i, options)
        % Add gradient.
        obj.backprop.grad.(name) = obj.backprop.grad.(name) + grad_i;
    end
end

methods
    function names = getLearnableParamNames(obj)
        % list of learnable properties
        names = {}; % default none
    end
end

methods (Static, Access = protected)
    function res = getCount()
        % used to give each layer a unique name
        persistent count;
        if isempty(count)
            count = 0;
        end

        count = count + 1;
        res = count;
    end
end

methods (Access = protected)
    % overwrite copy function
    function cp = copyElement(obj)
        cp = copyElement@matlab.mixin.Copyable(obj);
        cp.name = sprintf('%s_copy', obj.name);
    end
end

end

% ------------------------------ END OF CODE ------------------------------
