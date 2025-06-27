function obj = readONNXNetwork(file_path, varargin)
% readONNXNetwork - reads and converts a network saved in onnx format
%    Note: If the onnx network contains a custom layer, this function will
%    create a +CustomLayer package folder containing all custom layers in
%    your current MATLAB directory.
%
% Syntax:
%    res = neuralNetwork.readONNXNetwork(file_path)
%
% Inputs:
%    file_path - path to file
%    verbose - bool if information should be displayed
%    inputDataFormats - dimensions of input e.g. 'BC' or 'BSSC'
%    outputDataFormats - see inputDataFormats
%    targetNetwork - ...
%    containsCompositeLayers - there are residual connections in the
%    network
%
% Outputs:
%    obj - generated object
%
% Other m-files required: none
% Subfunctions: none
% MAT-files required: none
%
% See also: -

% Authors:       Tobias Ladner
% Written:       30-March-2022
% Last update:   07-June-2022 (specify in- & outputDataFormats)
%                30-November-2022 (removed neuralNetworkOld)
%                13-February-2023 (simplified function)
%                21-October-2024 (clean up DLT function call)
% Last revision: ---

% ------------------------------ BEGIN CODE -------------------------------

% validate parameters
narginchk(1,6)
[verbose, inputDataFormats, outputDataFormats, targetNetwork, ...
    containsCompositeLayers] = setDefaultValues({false, 'BC', 'BC', ...
        'dagnetwork', false}, varargin);

% valid in-/outputDataFormats for importONNXNetwork
validDataFormats = {'','BC','BCSS','BSSC','CSS','SSC','BCSSS','BSSSC', ...
    'CSSS','SSSC','TBC','BCT','BTC','1BC','T1BC','TBCSS','TBCSSS'};
inputArgsCheck({ ...
    {verbose, 'att', 'logical'}; ...
    {inputDataFormats, 'str', validDataFormats}; ...
    {outputDataFormats, 'str', validDataFormats}; ...
    {targetNetwork, 'str', {'dagnetwork', 'dlnetwork'}}; ...
})


if verbose
    disp("Reading network...")
end

% try to read ONNX network using dltoolbox
try
    dltoolbox_net = aux_readONNXviaDLT(file_path,inputDataFormats,outputDataFormats,targetNetwork);

catch ME
    if strcmp(ME.identifier, 'MATLAB:javachk:thisFeatureNotAvailable') && ...
            contains(ME.message,'Swing is not currently available.')
        % matlab tries to indent the code of the generated files for 
        % custom layers, for which (somehow?) a gui is required.
        % As e.g. docker runs don't have a gui, we just try to remove the
        % 'indentcode' function call here ...
        aux_removeIndentCodeLines(ME);

        % re-read network
        dltoolbox_net = aux_readONNXviaDLT(file_path,inputDataFormats,outputDataFormats,targetNetwork);

    else
        rethrow(ME)
    end
end

if containsCompositeLayers
    % Combine multiple layers into blocks to realize residual connections and
    % parallel computing paths.
    layers = aux_groupCompositeLayers(dltoolbox_net.Layers,dltoolbox_net.Connections);
else
    layers = num2cell(dltoolbox_net.Layers);
end

% convert DLT network to CORA network
% obj = neuralNetwork.convertDLToolboxNetwork(dltoolbox_net.Layers, verbose);
obj = neuralNetwork.convertDLToolboxNetwork(layers, verbose);


end


% Auxiliary functions -----------------------------------------------------

function dltoolbox_net = aux_readONNXviaDLT(file_path,inputDataFormats,outputDataFormats,targetNetwork)
    % read ONNX network via DLT

    % build name-value pairs
    NVpairs = {};

    % input data format
    if ~isempty(inputDataFormats)
        NVpairs = [NVpairs, {'InputDataFormats', inputDataFormats}];
    end

    % output data format
    if ~isempty(outputDataFormats)
        NVpairs = [NVpairs, {'OutputDataFormats', outputDataFormats}];
    end

    % custom layers generated from DLT will be stored in this folder
    % https://de.mathworks.com/help/deeplearning/ref/importnetworkfromonnx.html#mw_ccdf29c9-84cf-4175-a8ce-8e6ab1c89d4c
    customLayerName = 'DLT_CustomLayers';
    if isMATLABReleaseOlderThan('R2024a')
        % legacy
        NVpairs = [NVpairs, {'PackageName',customLayerName}];
    else
        NVpairs = [NVpairs, {'NameSpace',customLayerName}];
    end

    % load network
    if isMATLABReleaseOlderThan('R2023b')
        % legacy
        dltoolbox_net = importONNXNetwork(file_path, NVpairs{:}, 'TargetNetwork', targetNetwork);
    else
        dltoolbox_net = importNetworkFromONNX(file_path, NVpairs{:});
    end
end

function aux_removeIndentCodeLines(ME)

    % remove 'indentcode' function call 

    files = {'nnet.internal.cnn.onnx.fcn.ModelTranslation', 'nnet.internal.cnn.onnx.CustomLayerManager'};

    for i=1:length(files)
        % error happens in this file
        internalPath = which(files{i});
    
        % read text and comment failing line
        filetext = fileread(internalPath);
        filetext = strrep(filetext,"indentcode(","(");
    
        % try to read file with write permission
        fid  = fopen(internalPath,'w');
        if fid == -1
            % rethrowing error
            rethrow(ME)
        end
    
        % write new filetext
        fprintf(fid,'%s',filetext);
        fclose(fid);

    end

end

function layers = aux_groupCompositeLayers(layerslist, connections)
    % We use a digraph as a utility to convert the table of connections to
    % a nested cell array.

    % Preprocess for digraph conversion.
    [layernames, conns, dltLayers] = ...
        nnHelper.buildDigraphIngredientsFromNodesAndConns( ...
            layerslist, connections);
    % Convert to digraph.
    G = digraph( ...
        table(cell2mat(conns),'VariableNames', {'EndNodes'}), ...
        table(layernames{1},'VariableNames', {'Names'}));
    % Sort the nodes topologically.
    N = toposort(G);
    % Use a reverse depth-first traversal to construct a nested cell array.
    layers = aux_buildNestedCellArray(G,dltLayers,N(end));
end

function [layers,ni] = aux_buildNestedCellArray(G,dltLayers,nK)
    % We use a reverse depth-first traversal to construct a nested cell
    % array. We have to reverse the traversal to avoid not creating nested 
    % cells. 

    % Initialize the cell array.
    layers = {};
    % Initialize with last node.
    ni = nK;
    % Count the number of loop iteration to prevent an infinite loop.
    iter = 1;
    % Traverse the graph and build the nested cell array.
    while iter < height(G.Nodes)
        % Obtain the successors.
        succs = successors(G,ni);
        if length(succs) > 1 && ni ~= nK
            % We reached a forking node; we break out of the current
            % computation path.
            break;
        end

        % Prepend the current layer.
        layers = [dltLayers(ni) layers];

        % Obtain the predecessors.
        preds = predecessors(G,ni);

        if isempty(preds)
            % There are no predecessors; we have reached the input node.
            break;
        elseif length(preds) <= 1
            % Move to the only predecessor.
            ni = preds(1);
        else % if length(preds) > 1
            % There are multiple computation paths merging in the current
            % node. We have to construct the layer cell arrays for each
            % computation path.
            layersPreds = {};
            % Create a new computation path for each predecessor.
            for j=1:length(preds)
                % Check if the computation path is just a resiudal
                % connection.
                isResidualConn = length(successors(G,preds(j))) > 1;
                if isResidualConn
                    % Prepend the empty cell for the j-th computation path.
                    layersPreds = [{}; layersPreds];
                else
                    % Compute the layers of the j-th computation path.
                    [layersj,ni] = aux_buildNestedCellArray(G,dltLayers,preds(j));
                    % Prepend the layers of the j-th computation path.
                    layersPreds = [{layersj}; layersPreds];
                end
            end
            % Traversed all computation paths; prepend the layers.
            layers = [{layersPreds} layers];
            % Update the starting node.
            nK = ni;
        end
        % Increment iteration counter.
        iter = iter + 1;
    end
end

% ------------------------------ END OF CODE ------------------------------
