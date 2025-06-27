

function [layernames, conns, layerobjectcell] = buildDigraphIngredientsFromNodesAndConns(networkLayers, networkConnections)
% buildDigraphIngredientsFromNodesAndConns - returns a preprocessing of ingredients in order to create a CORA NN digraph object
%                                            The inputs are the NodeTable and EdgeTable of the matlab digraph object (see https://www.mathworks.com/help/matlab/ref/digraph.html#mw_6ee207c1-b4f0-4f84-957e-d6d4816748d6)
% 
% Syntax:
%    [layernames, conns, layerobjectcell] = buildDigraphIngredientsFromNodesAndConns(networkLayers, networkConnections)
%
% Inputs:
%    networkLayers      - the matlab DLT objects, of which we extract properties
%                         This is a NodeTable object from matlab digraph holding only the 'Name' property 
%    networkConnections - the connections between the neural network nodes
%                         This is a EdgeTable object from matlab digraph holding only the 'EndNodes' property
%    incoming_zonotope - a packed zonotope: A cell array consisting of center c and generator matrix G 
%
% Outputs:
%    layernames - array of strings. These are the names (identifiers) of the soon-to-be CORA layers
%    conns      - table of integer-pairs. These pairs are the indices mapping a layer to another layer
%                 They symbolize the neural network connections between layers
%    layerobjectcell    - A cell of matlab DLT layer objects. The order of these DLT objects is referenced by the 
%                         indices in the "conns" output
%
% Other m-files required: none
% Subfunctions: none
% MAT-files required: none
%
% See also: -
% Authors:       Stefan Schaerdinger
% Written:       2-Feb-2025

% ------------------------------ BEGIN CODE -------------------------------


    % --- transforming the connections table in preparation for the digraph matlab object
    conns = networkConnections;
    layerobjectcell = {};
    conns = table(table2cell(conns), 'VariableNames',{'EndNodes'});
    layers = networkLayers;
    layers = table(layers, 'VariableNames',{'Name'});
    newlayers = {};

    for i = 1:size(layers,1) % converting entries to string identifiers
        newlayers{i} = layers.Name(i).Name;
        layerobjectcell{i} = layers.Name(i);
    end
    
    % creating a table for the layer names
    newlayers = table(newlayers', 'VariableNames' , {'Name'});
    newconns_a = [];
    newconns_b = []; % matching node string identifiers to numeric ids 
    for i = 1:size(conns,1)
        startnode = conns.EndNodes{i,1};
        
        % filtering and modifying merge layers
        if contains(startnode, '/')
            slashidx = strfind(startnode, '/');
            startnode = startnode( 1 : (slashidx-1));
        end

        for j = 1:size(newlayers,1)
            if strcmp(newlayers.Name{j} , startnode)
                newconns_a(i) = j;
                break;
            end
        end
        endnode = conns.EndNodes{i,2};
        
        % filtering and modifying merge layers
        if contains(endnode, '/')
            slashidx = strfind(endnode, '/');
            endnode = endnode( 1 : (slashidx-1));
        end

        for j = 1:size(newlayers,1)
            if strcmp(newlayers.Name{j} , endnode)
                newconns_b(i) = j;
                break;
            end
        end
    end
    
    layernames = {table2cell(newlayers)};
    conns = {[newconns_a' , newconns_b']};
