classdef nnCompositeLayer < nnLayer
% nnCompositeLayer - realize multiple parallel computation paths, e.g. res
%    connections
%
% Syntax:
%    obj = nnCompositeLayer(layers,aggregation)
%
% Inputs:
%    layers - cell array with the different parallel computation paths
%    aggregation - 'add' or 'concant'
%
% Outputs:
%    obj - generated object
%
% Example:
%   % A res connection.
%   obj = nnCompositeLayer({{nnLinearLayer(1,0), nnReLULayer}; {}}, 'add')
%
% Other m-files required: none
% Subfunctions: none
% MAT-files required: none
%
% See also: neuralNetwork

% Authors:       Lukas Koller
% Written:       27-June-2024
% Last update:   ---
% Last revision: ---

% ------------------------------ BEGIN CODE -------------------------------

properties (Constant)
    is_refinable = false
end

properties
    layers
    aggregation
end

methods
    % constructor
    function obj = nnCompositeLayer(layers, aggregation, varargin)
        obj@nnLayer(varargin{:})
        obj.layers = layers;
        obj.aggregation = aggregation;
    end

    function outputSize = getOutputSize(obj, inputSize)
        % Compute the output size of the first computation path.
        outputSizes = cell(length(obj.layers),1);
        for i=1:length(obj.layers)
            layersi = obj.layers{i};
            outputSizes{i} = inputSize;
            for j=1:length(layersi)
                outputSizes{i} = layersi{j}.computeSizes(outputSizes{i});
            end
        end

        if strcmp(obj.aggregation,'add')
            % Same output size as the individual outputs.
            outputSize = outputSizes{1};
        elseif strcmp(obj.aggregation,'concat')
            % TODO
            outputSize = outputSizes{1};
        else
            throw(CORAerror('CORA:wrongFieldValue', ...
                'nnCompositeLayer.aggregation', ...
                "Only supported value is 'add'!"));
        end
    end

    function [nin, nout] = getNumNeurons(obj)
        if isempty(obj.inputSize)
            nin = [];
            nout = [];
        else
            % We can only compute the number of neurons if the input
            % size was set.
            nin = prod(obj.inputSize);
            outputSize = getOutputSize(obj, obj.inputSize);
            nout = prod(outputSize);
        end
    end
end

% evaluate ----------------------------------------------------------------

methods (Access = {?nnLayer, ?neuralNetwork})
    
    % numeric
    function r = evaluateNumeric(obj, input, options)
        r = 0;
        for i=1:length(obj.layers)
            % Compute result for the i-th computation path.
            layersi = obj.layers{i};
            ri = input;
            for j=1:length(layersi)        
                % Store input for backpropgation.
                if options.nn.train.backprop
                    layersi{j}.backprop.store.input = ri;
                end
                ri = layersi{j}.evaluateNumeric(ri, options);
            end
            % Aggreate results.
            if strcmp(obj.aggregation,'add')
                % Add results.
                r = r + ri;
            % elseif strcmp(obj.aggregation,'concat')
            %     % Concatenate results.
            %     % TODO
            else
                throw(CORAerror('CORA:wrongFieldValue', ...
                    'nnCompositeLayer.aggregation', ...
                    "Only supported value is 'add'!"));
            end
        end
    end

    % sensitivity
    function S = evaluateSensitivity(obj, S, options)
        % Retain input sensitivity.
        Sin = S;
        % Initialize output sensitvity.
        S = 0;
        for i=1:length(obj.layers)
            % Compute result for the i-th computation path.
            layersi = obj.layers{i};
            Si = Sin;
            for j=length(layersi):-1:1
                Si = layersi{j}.evaluateSensitivity(Si,options);
            end
            % Aggregate results.
            if strcmp(obj.aggregation,'add')
                % Add results.
                S = S + Si;
            % elseif strcmp(obj.aggregation,'concat')
            %     % Concatenate results.
            %     % TODO
            else
                throw(CORAerror('CORA:wrongFieldValue', ...
                    'nnCompositeLayer.aggregation', ...
                    "Only supported value is 'add'!"));
            end
        end

        if options.nn.store_sensitivity
            % Store the gradient (used for the sensitivity computation).
            obj.sensitivity = S;
        end
    end

    % zonotope batch
    function [rc, rG] = evaluateZonotopeBatch(obj, c, G, options)
        rc = [];
        rG = [];
        for i=1:length(obj.layers)
            % Initialize output of the i-th computation path. 
            rci = c;
            rGi = G;
            % Compute result for the i-th computation path.
            layersi = obj.layers{i};
            for j=1:length(layersi)
                layersij = layersi{j};
                % Store input for backpropgation.
                if options.nn.train.backprop
                    layersij.backprop.store.inc = rci;
                    layersij.backprop.store.inG = rGi;
                end
                [rci,rGi] = layersij.evaluateZonotopeBatch(rci,rGi,options);
            end

            if isempty(rc)
                rc = rci;
                rG = rGi;
            else
                % Aggregate results.
                if strcmp(obj.aggregation,'add')
                    % Add final results.
                    rc = rc + rci;
                    if size(rG,2) < size(rGi,2)
                        rGi(:,1:size(rG,2),:) = rGi(:,1:size(rG,2),:) + rG;
                        rG = rGi;
                    else
                        rG(:,1:size(rGi,2),:) = rG(:,1:size(rGi,2),:) + rGi;
                    end
                % elseif strcmp(obj.aggregation,'concat')
                %     % Concatenate results.
                %     % TODO
                else
                    throw(CORAerror('CORA:wrongFieldValue', ...
                        'nnCompositeLayer.aggregation', ...
                        "Only supported value is 'add'!"));
                end
            end
        end
    end

    function grad_in = backpropNumeric(obj, input, grad_out, options, updateWeights)
        % Initialize the gradient.
        grad_in = 0;
        for i=1:length(obj.layers)
            % Initialize output of the i-th computation path. 
            grad_in_i = grad_out;
            % Backpropagate the gradient through the i-th computation path.
            layersi = obj.layers{i};
            for j=length(layersi):-1:1
                % Retrieve stored input.
                inputsij = layersi{j}.backprop.store.input;
                % Compute the gradient.
                grad_in_i = layersi{j}.backpropNumeric(inputsij, ...
                    grad_in_i,options,updateWeights);
            end

            % Aggregate results.
            if strcmp(obj.aggregation,'add')
                grad_in = grad_in + grad_in_i;
            % elseif strcmp(obj.aggregation,'concat')
            %     % Concatenate results.
            %     % TODO
            else
                throw(CORAerror('CORA:wrongFieldValue', ...
                    'nnCompositeLayer.aggregation', ...
                    "Only supported value is 'add'!"));
            end
        end
    end

    % zonotope batch
    function [rgc, rgG] = backpropZonotopeBatch(obj, c, G, ...
            gc, gG, options, updateWeights)
        rgc = [];
        rgG = [];
        for i=1:length(obj.layers)
            % Initialize output of the i-th computation path. 
            rgci = gc;
            rgGi = gG;
            % Compute result for the i-th computation path.
            layersi = obj.layers{i};
            idxLayeri = flip(1:length(layersi));
            for j=idxLayeri
                layersij = layersi{j};
                % Retrieve stored input
                c = layersij.backprop.store.inc;
                G = layersij.backprop.store.inG;
                % Compute the gradient.
                [rgci,rgGi] = layersij.backpropZonotopeBatch(c,G, ...
                    rgci,rgGi,options,updateWeights);
            end

            if isempty(rgc)
                rgc = rgci;
                rgG = rgGi;
            else
                % Aggregate results.
                if strcmp(obj.aggregation,'add')
                    % Add final results.
                    rgc = rgc + rgci;
                    if size(rgG,2) < size(rgGi,2)
                        rgGi(:,1:size(rgG,2),:) = rgGi(:,1:size(rgG,2),:) + rgG;
                        rgG = rgGi;
                    else
                        rgG(:,1:size(rgGi,2),:) = rgG(:,1:size(rgGi,2),:) + rgGi;
                    end
                % elseif strcmp(obj.aggregation,'concat')
                %     % Concatenate results.
                %     % TODO
                else
                    throw(CORAerror('CORA:wrongFieldValue', ...
                        'nnCompositeLayer.aggregation', ...
                        "Only supported value is 'add'!"));
                end
            end
        end
    end
end

% Auxiliary functions -----------------------------------------------------

end

% ------------------------------ END OF CODE ------------------------------
