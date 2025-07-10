function [res, x_, y_] = verify(nn, x, r, A, b, safeSet, varargin)
% verify - automated verification for specification on neural networks.
%
% Syntax:
%    [res, z] = nn.verify(x, r, A, b, options)
%
% Inputs:
%    nn - object of class neuralNetwork
%    x, r - center and radius of the initial set (can already be a batch)
%    A, b - specification, prove A*y <= b
%    safeSet - bool, safe-set or unsafe-set
%    options - evaluation options
%    timeout - timeout in seconds
%    verbose - print verbose output
%    plotDims - 2x2 plot dimensions; empty for no plotting; 
%           plotDims(1,:) for input and plotDims(2,:) for output; sets 
%           are stored in res.Xs, res.uXs
%
% Outputs:
%    res - result: true if specification is satisfied, false if not, empty if unknown
%    x_ - counterexample in terms of an initial point violating the specs
%    y_ - output for x_
%
% References:
%    [1] VNN-COMP'24
%
% Other m-files required: none
% Subfunctions: none
% MAT-files required: none
%
% See also: -

% Authors:       Lukas Koller
% Written:       23-November-2021
% Last update:   14-June-2024 (LK, rewritten with efficient splitting)
%                20-January-2025 (LK, constraint zonotope splitting)
% Last revision: ---

% ------------------------------ BEGIN CODE -------------------------------

% Check number of input arguments.
narginchk(6,10);

% Validate parameters.
[options, timeout, verbose, plotDims] = ...
    setDefaultValues({struct, 100, false, []}, varargin);
plotting = ~isempty(plotDims);

% Validate parameters.
inputArgsCheck({ ...
    {nn, 'att','neuralNetwork'}; ...
    {x, 'att',{'numeric','gpuArray'}}; ...
    {r, 'att',{'numeric','gpuArray'}}; ...
    {A, 'att',{'numeric','gpuArray'}}; ...
    {b, 'att',{'numeric','gpuArray'}}; ...
    {options,'att','struct'}; ...
    {timeout,'att','numeric','scalar'}; ...
    {verbose,'att','logical'}; ...
    {plotting,'att','logical'}; ...
})
options = nnHelper.validateNNoptions(options,true);

nSplits = options.nn.num_splits; % Number of input splits per dimension.
nDims = options.nn.num_dimensions; % Number of input dimension to splits.
nNeur = options.nn.num_neuron_splits; % Number of neurons to split.

% Extract parameters.
bSz = options.nn.train.mini_batch_size;

% Obtain number of input dimensions.
n0 = size(x,1);
% Limit the number of dimensions to split.
nDims = min(nDims,n0);
% Check the maximum number of input generators.
numInitGens = min(options.nn.train.num_init_gens,n0);
% Obtain the number of approximation error generators per layer.
numApproxErrGens = options.nn.train.num_approx_err;
% Obtain the maximum number of approximation errors in an activation layer.
nk_max = max(cellfun(@(li) ...
    isa(li,'nnActivationLayer')*prod(li.getOutputSize(li.inputSize)), ...
    nn.layers) ...
);
% We always have to use the approximation during set propagation to ensure
% soundness.
options.nn.use_approx_error = true;
% Ensure the interval-center flag is set, if there are less generators than
% input dimensions.
options.nn.interval_center = ...
    (numApproxErrGens < nk_max) | (numInitGens < n0);

% To speed up computations and reduce GPU memory, we only use single 
% precision.
inputDataClass = single(1);
% Check if a GPU is used during training.
useGpu = options.nn.train.use_gpu;
if useGpu
    % Training data is also moved to GPU.
    inputDataClass = gpuArray(inputDataClass);
end
% (potentially) move weights of the network to GPU.
nn.castWeights(inputDataClass);

% Specify indices of layers for propagation.
idxLayer = 1:length(nn.layers);

% In each layer, store ids of active generators and identity matrices 
% for fast adding of approximation errors.
q = nn.prepareForZonoBatchEval(x,options,idxLayer);
% Allocate generators for initial perturbation set.
batchG = zeros([n0 q bSz],'like',inputDataClass);

% Initialize queue.
xs = x;
rs = r;
% Compute number of union constraints (all intersection constrains -> 
% union of one single constraint).
if safeSet
    numUnionConst = size(A,1);
else
    numUnionConst = 1;
end
% Initialize result.
res.str = 'UNKNOWN';
x_ = [];
y_ = [];

% Initialize iteration stats.
numVerified = 0;
% Initialize iteration counter.
iter = 1;

if plotting
    % Initialize cell arrays to store intermediate results for plotting.
    res.Xs = {};
    res.Ys = {};
    res.xs_ = {};
    res.ys_ = {};

    % if startsWith(options.nn.refinement_method,'zonotack')
    %     % Unsafe sets are only required for the 'zonotack'.
    %     res.uXs = {};
    %     res.uYs = {};
    % end
    
    % Compute samples.
    sampX = randPoint(interval(x - r,x + r),1000);
    sampY = gather(double(nn.evaluate(sampX)));

    % Create a new figure.
    fig = figure;
    % Initialize plot.
    [fig,hx0,hspec] = aux_initPlot(fig,plotDims, ...
        sampX,sampY,x,r,A,b,safeSet);
    drawnow;
end

if verbose
    % Setup table.
    table = CORAtable('double', ...
        {'Iteration','#Queue','#Verified','Avg. radius', ...
            'Unknown Vol. [%]'}, ...
        {'d','d','d','.3e','.4f'});
    table.printHeader();
end

% Obtain initial split parameters.
initSplits = options.nn.init_split(1);
initDims = options.nn.init_split(2);
% Apply initial split.
if initSplits > 1
    % Compute the sensitivity of the center.
    [S,~] = nn.calcSensitivity(x,options);
    % The sensitivity should not be lower than 1e-6, otherwise it is too 
    % low to be effective for the (neuron-) splitting heuristic.
    S_ = max(abs(S),1e-6);
    sens = reshape(max(S_,[],1),size(r)); 

    % Obtain order of the size of the different dimensions.
    [~,dimOrder] = sort(sens.*r,1,'descend');
    % Split the dimensions.
    for i=1:initDims
        % Construct sensitivity matrix s.t. the i-th dimension in dimOrder
        % is split.
        sensi = zeros(size(rs),'like',rs);
        sensi(dimOrder(i),:) = 1;
        % Split the dimension.
        [xs,rs] = aux_split(xs,rs,sensi,initSplits);
    end

    % Check the centers for quick falsification examples.
    [~,critVal,falsified,x_,y_] = ...
        aux_checkPoints(nn,options,idxLayer,A,b,safeSet,xs);
    
    if any(falsified)
        % Found a counterexample.
        res.str = 'COUNTEREXAMPLE';
        return;
    end

    % Re-order the splits based on their criticallity.
    [~,idx] = sort(critVal,'ascend');
    xs = xs(:,idx);
    rs = rs(:,idx);
end

% Specify the heuristics (TODO: make this an argument).
% Input-Split Options: 
%  {'most-sensitive-input-radius',
%   'ival-norm-gradient'}.
inputSplitHeuristic = 'ival-norm-gradient';

% Neuron-Split Options: 
%  {'least-unstable', 
%   'most-sensitive-approx-error',
%   'most-sensitive-input-radius',
%   'ival-norm-gradient',
%   'most-sensitive-input-aligned'}.
neuronSplitHeuristic = 'ival-norm-gradient';

% The inputs are needed for the neuron splitting, relu tightening 
% constraints, or layerwise refinement.
storeInputs = nNeur > 0 || options.nn.num_relu_constraints > 0 ...
     || strcmp(options.nn.refinement_method,'zonotack-layerwise') ...
     || strcmp(inputSplitHeuristic,'ival-norm-gradient') ...
     || strcmp(neuronSplitHeuristic,'ival-norm-gradient');
% The sensitivity is used for selecting input generators, neuron
% -splitting, and FGSM attacks.
storeSensitivity =  (nNeur > 0) || options.nn.num_relu_constraints > 0 ...
    || strcmp(options.nn.approx_error_order,'sensitivity*length');

timerVal = tic;

% Main splitting loop.
while size(xs,2) > 0

    % Check if we reach the maximum number of iterations.
    if iter > options.nn.max_verif_iter
        break;
    end

    time = toc(timerVal);
    if time > timeout
        % Time is up.
        res.time = time;
        break;
    end

    if verbose
        % Compute iteration stats.
        queueLen = size(xs,2);
        avgRad = mean(rs,'all');
        unknVol = sum(prod(2*rs,1),'all')/sum(prod(2*r),'all')*100;
        % Print new table row.
        table.printContentRow({iter,queueLen,numVerified,avgRad,unknVol});
    end

    % Pop next batch from the queue.
    [xi,ri,xs,rs] = aux_pop(xs,rs,bSz,options);
    % Obtain the current batch size.
    [~,cbSz] = size(xi);
    % Move the batch to the GPU.
    xi = cast(xi,'like',inputDataClass);
    ri = cast(ri,'like',inputDataClass);

    % Compute the sensitivity (store sensitivity for neuron-splitting).
    [S,~] = nn.calcSensitivity(xi,options,storeSensitivity);
    % The sensitivity should not be lower than 1e-3, otherwise it is too 
    % low to be effective for the (neuron-) splitting heuristic.
    S_ = max(abs(S),1e-6);
    sens = reshape(max(S_,[],1),[n0 cbSz]); 

    % TODO: investigate a more efficient implementation of the sensitivity
    % computation using backpropagation.

    % 1. Verification -----------------------------------------------------
    % 1.1. Use batch-evaluation of zonotopes.

    % Construct input zonotope. 
    [cxi,Gxi,inputDimIdx] = aux_constructInputZonotope(xi,ri, ...
        ri.*sens,batchG,numInitGens,options);

    % Store inputs for each layer by enabling backpropagation. 
    options.nn.train.backprop = storeInputs;
    % Compute output enclosure.
    [yi,Gyi] = nn.evaluateZonotopeBatch_(cxi,Gxi,options,idxLayer);
    % Disable backpropagation.
    options.nn.train.backprop = false;

    % Obtain number of output dimensions.
    [nK,~] = size(yi);

    % 2.2. Compute logit difference.
    [ld_yi,ld_Gyi,ld_Gyi_err,yic,yid,Gyi] = ...
        aux_computeLogitDifference(yi,Gyi,A,options);
    % Compute the radius of the logit difference.
    ld_ri = sum(abs(ld_Gyi),2) + ld_Gyi_err;
    % 2.3. Check specification.
    if safeSet
        % safe iff all(A*y <= b) <--> unsafe iff any(A*y > b)
        % Thus, unknown if any(A*y > b).
        unknown = any(ld_yi + ld_ri(:,:) > b,1);
    else
        % unsafe iff all(A*y <= b) <--> safe iff any(A*y > b)
        % Thus, unknown if all(A*y <= b).
        unknown = all(ld_yi - ld_ri(:,:) <= b,1);
    end
    % Update counter for verified patches.
    numVerified = numVerified + sum(~unknown,'all');

    if plotting
        % % Reset the figure.
        % clf(fig);
        % [fig,hx0,hspec] = aux_initPlot(fig,plotDims,x,r,A,b,safeSet);
        % Store input sets.
        if options.nn.interval_center
            xid = 1/2*(cxi(:,2,:) - cxi(:,1,:));
            res.Xs{end+1} = struct( ...
                'c',gather(xi),'G',gather(cat(2,Gxi,xid.*eye(n0))),...
                'verified',gather(~unknown) ...
            );
            % Store the output set.
            res.Ys{end+1} = struct( ...
                'c',gather(yic),'G',gather(cat(2,Gyi,yid.*eye(nK))) ...
            );
        else
            res.Xs{end+1} = struct('c',gather(cxi),'G',gather(Gxi),...
                'verified',gather(~unknown));
            % Store the output set.
            res.Ys{end+1} = struct('c',gather(yic),'G',gather(Gyi));
        end
        % Plot current input sets and propagated output sets.
        [fig,hxi,hx,hxv,hy,hyv] = aux_plotInputAndOutputSets(fig, ...
            plotDims,x,r,res);
        drawnow;
    end

    if all(~unknown)
        % Verified all subsets; skip to next iteration.
        iter = iter + 1;
        continue;
    elseif any(~unknown)
        % Only keep un-verified patches.
        xi(:,~unknown) = [];
        ri(:,~unknown) = [];
        sens(:,~unknown) = [];
        S(:,:,~unknown) = [];
    
        if options.nn.interval_center
            cxi(:,:,~unknown) = [];
            yi(:,:,~unknown) = [];
        else
            cxi(:,~unknown) = [];
            yi(:,~unknown) = [];
        end
        Gxi(:,:,~unknown) = [];
        inputDimIdx(:,~unknown) = [];
        yic(:,~unknown) = [];
        Gyi(:,:,~unknown) = [];
        yid(:,:,~unknown) = [];
        ld_yi(:,~unknown) = [];
        ld_Gyi(:,:,~unknown) = [];
    
        if storeSensitivity
            % Re-compute sensitivity to only store the sensitivity of the 
            % unknown batch entries.
            nn.calcSensitivity(xi,options,storeSensitivity);
        end
 
        if storeInputs
            % Re-compute output enclosure to store only the unknown batch 
            % entries within the layers.
            options.nn.train.backprop = true;
            nn.evaluateZonotopeBatch_(cxi,Gxi,options,idxLayer);
            options.nn.train.backprop = false;
        end
    end
    % Update the current batch size.
    [~,cbSz] = size(xi);

    % 2. Falsification ----------------------------------------------------

    % 2.1. Compute adversarial examples.
    switch options.nn.falsification_method
        case 'fgsm'
            % Try to falsification with a FGSM attack.
            
            if safeSet
                grad = pagemtimes(A,S);
            else
                grad = pagemtimes(-A,S);
            end
            % Obtain number of constraints.
            [p,~] = size(A);
            % If there are multiple output constraints we try to falsify
            % each one individually.
            sgrad = reshape(permute(sign(grad),[2 3 1]),[n0 cbSz*p]);
            
            % Compute adversarial attacks based on the sensitivity.
            xi_ = repmat(xi,1,p) + repmat(ri,1,p).*sgrad;
        case 'center'
            % Use the center for falsification.
            xi_ = xi;
        case 'zonotack'
            % TODO: more clever attacks based on the output set.

            % Obtain number of constraints.
            [p,~] = size(A);

            % Compute the vertex that minimizes the distance to each 
            % halfspace.
            beta_ = permute(sign(ld_Gyi(:,1:numInitGens,:)),[2 4 3 1]);
            if ~safeSet
                beta_ = -beta_;
            end
            % Put multiple candidates into the batch.
            beta = reshape(beta_,[numInitGens 1 cbSz*p]);

            % Specify number of samples per batch entry.
            nrSamp = 0;
            % Add some random samples.
            beta = cat(3,beta,rand([numInitGens 1 cbSz*nrSamp],'like',beta));

            % Compute attack.
            delta = pagemtimes( ...
                repmat(Gxi(:,1:numInitGens,:),1,1,p+nrSamp),beta);
            delta = reshape(delta,[n0 cbSz*(p+nrSamp)]);
            % Compute candidates for falsification.
            xi_ = repmat(xi,1,p+nrSamp) + delta;
        otherwise
            % Invalid option.
            throw(CORAerror('CORA:wrongFieldValue', ...
                'options.nn.falsification_method', ...
                    {'fgsm','center','zonotack'}));
    end

    % 2.2. Check the specification for adversarial examples.
    
    % Check the adversarial examples.
    [~,critVal,falsified,x_,y_] = ...
        aux_checkPoints(nn,options,idxLayer,A,b,safeSet,xi_);

    if any(falsified)
        % Found a counterexample.
        res.str = 'COUNTEREXAMPLE';
        break;
    end

    % Check if the batch was extended with multiple candidates.
    if size(critVal,2) > cbSz
        critVal_ = reshape(critVal,1,cbSz,[]);
        % Find the worst candidate: If any candicate has a negative 
        % criticallity value, we have a counterexample.
        critVal = min(critVal_,[],3);
    end

    % 3. Refine input sets. -----------------------------------------------

    switch options.nn.refinement_method
        case 'naive'
            % The sets are not refined; split the input dimensions.
            for i=1:nDims
                % Compute heuristic.
                hi = sens.*ri;
                % Split the input sets along one dimensions.
                [xi,ri] = aux_split(xi,ri,hi,nSplits);
                % Replicate sensitivity and criticallity value.
                sens = repmat(sens,1,nSplits);
                critVal = repmat(critVal,1,nSplits);
            end
        case {'zonotack','zonotack-layerwise'}

            % Initialize number of splitted sets.
            newSplits = 1;
    
            if strcmp(inputSplitHeuristic,'ival-norm-gradient') ...
                || strcmp(neuronSplitHeuristic,'ival-norm-gradient')
                % Store the gradients of the approximation errors.
                options.nn.store_approx_error_gradients = true;
                % Compute gradient of the interval norm of the output set; the
                % gradient is used to split the neuron in the network as well
                % as input dimensions.
                [~,ivalGrad] = nn.backpropZonotopeBatch( ...
                    zeros(size(yi),'like',yi),sign(Gyi),options, ...
                        idxLayer,false);
            end

            if nNeur > 0 && nSplits > 1
                % Create split constraints for neurons within the network.
                
                % Construct the constraints.
                [As,bs,~] = aux_neuronConstraints(nn,options,[], ...
                    neuronSplitHeuristic,nSplits,nNeur,numInitGens);

                % % Compute output-split constraints.
                % [As,bs] = aux_outputDimSplitConstraints(yi,Gyi,nSplits,nNeur);

                % Compute number of new splits.
                newSplits = nSplits^nNeur*newSplits;
            else
                % There are no general-split constraints.
                As = zeros([0 size(Gyi,2) cbSz],'like',Gyi);
                bs = zeros([0 1 cbSz],'like',Gyi);
            end

            if nDims > 0 && nSplits > 1
                % When not all input dimensions get an assigned generator
                % we have to restrict and reorder the dimensions.
                % Therefore, we compute indices.
                permIdx = reshape(sub2ind(size(xi), ...
                    inputDimIdx,repelem(1:cbSz,numInitGens,1)), ...
                        [numInitGens cbSz]);

                switch inputSplitHeuristic
                    case 'most-sensitive-input-radius'
                        % Compute the heuristic.
                        hi = sens(permIdx).*ri(permIdx);
                    case 'ival-norm-gradient'
                        % Compute indices for the gradient of the interval 
                        % norm w.r.t. the different generators.
                        dimGenIdx = reshape(sub2ind(size(ivalGrad), ...
                            inputDimIdx, ...
                            repmat((1:numInitGens)',1,cbSz), ...
                            repelem(1:cbSz,numInitGens,1)),[numInitGens cbSz]);
                        % Compute gradient of the interval norm.
                        % hi = reshape(ivalGrad(dimGenIdx),[numInitGens cbSz]) ...
                        %     .*ri(permIdx);
                        hi = log(reshape(ivalGrad(dimGenIdx),[numInitGens cbSz]) + 1).*ri(permIdx);
                end

                % Compute input-split constraints.
                [Ai,bi] = aux_dimSplitConstraints(hi(:,:),nSplits,nDims);

                % Update number of new splits.
                newSplits = nSplits^nDims*newSplits;
            else
                % There are no input-split constraints.
                Ai = zeros([0 size(Gyi,2) cbSz],'like',Gyi);
                bi = zeros([0 1 cbSz],'like',Gyi);
            end

            % Refine the input set based on the output specification.
            [li,ui,~] = aux_refineInputSet(nn,options,storeInputs, ...
                cxi,Gxi,yi,Gyi,A,b,safeSet,Ai,bi,As,bs, ...
                    numInitGens,numUnionConst);

            % Identify empty sets.
            isEmpty = any(isnan(li),1) | any(isnan(ui),1);

            % Compute center and radius of refined sets.
            xi = 1/2*(ui + li);
            ri = 1/2*(ui - li);

            % Check contained of the refined sets.
            isContained = aux_isContained(xi(:,~isEmpty),ri(:,~isEmpty));
            % Check the specification for the points.
            if any(~isEmpty)
                [~,critVal,falsified,x_,y_] = aux_checkPoints(nn,options, ...
                    idxLayer,A,b,safeSet,xi(:,~isEmpty));
                if any(falsified)
                    % Found a counterexample.
                    res.str = 'COUNTEREXAMPLE';
                    break;
                end
            else
                critVal = [];
            end
            % Identify which sets were refined to just being a point.
            isPoint = all(ri(:,~isEmpty) == 0,1);

            % Remove the empty sets.
            remIdx = isEmpty;
            remIdx(~isEmpty) = isContained | isPoint;
            % Remove sets that are empty or contained.
            xi(:,remIdx) = [];
            ri(:,remIdx) = [];
            critVal(:,isContained | isPoint) = [];
            
            % All removed subproblems are verified.
            numVerified = numVerified + sum(isPoint & ~isContained);

            if plotting && isfield(res,'uYs') && isfield(res,'uXs')
                % Add a slack variable to convert between equality and 
                % inequality constraints.
                uYi = struct('c',yic,'G',Gyi,'r',yid, ...
                    'A',ld_Gyi,'b',b - ld_yi - ld_Gyi_err);
                % Store constraint zonotope.
                res.uYs{end+1} = aux_2ConZonoWithEqConst(uYi,0);

                % Store input constraint zonotope.
                res.uXs{end+1} = aux_2ConZonoWithEqConst(uXi,0);
            end
        otherwise
            % Invalid option.
            throw(CORAerror('CORA:wrongFieldValue', ...
                'options.nn.refinement_method', ...
                    {'naive','zonotack','zonotack-layerwise'}));
    end

    % Order remaining sets by their criticality.
    [~,idx] = sort(critVal.*1./max(ri,[],1),'ascend');
    % Order sets.
    xi = xi(:,idx);
    ri = ri(:,idx);

    % Add new splits to the queue.
    switch options.nn.verify_enqueue_type
        case 'append'
            xs = [xs xi];
            rs = [rs ri];
        case 'prepend'
            xs = [xi xs];
            rs = [ri rs];
        otherwise
            % Invalid option.
            throw(CORAerror('CORA:wrongFieldValue', ...
                'options.nn.verify_enqueue_type',{'append','prepend'}));
    end

    if plotting
        % Delete previously contained sets.
        if exist('hx_','var')
            cellfun(@(hxi_) delete(hxi_), hx_);
        end
        % Compute the bounds of the new sets.
        li = xi - ri;
        ui = xi + ri;
        if ~exist('isContained','var')
            isContained = zeros([size(li,2) 1],'logical');
        end
        % Plot the unsafe output sets and the new input sets.
        [fig,huy,huy_,hux,hx,hx_] = ...
            aux_plotUnsafeOutputAndNewInputSets(fig,plotDims,res, ...
                li,ui,isContained,nSplits^nDims);
        drawnow;
    end

    % To save memory, we clear all variables that are no longer used.
    batchVars = {'xi','ri','xGi','yi','Gyi','ld_yi','ld_Gyi','ld_ri'};
    clear(batchVars{:});
     
    % Increment iteration counter.
    iter = iter + 1;
end

if size(xs,2) == 0 && ~strcmp(res.str,'COUNTEREXAMPLE')
    % Verified all patches.
    res.str = 'VERIFIED';
    x_ = [];
    y_ = [];
end

% Store time.
res.time = toc(timerVal);
% Store number of verified patches.
res.numVerified = gather(numVerified);

if verbose
    % Compute final stats.
    queueLen = size(xs,2);
    if ~isempty(rs)
        avgRad = mean(rs,'all');
        unknVol = sum(prod(2*rs,1),'all')/sum(prod(2*r),'all')*100;
    else
        avgRad = 0;
        unknVol = 0;
    end
    % Print new table row.
    table.printContentRow({iter,queueLen,numVerified,avgRad,unknVol});
    % Print table footer.
    table.printFooter();
    % Print the result.
    fprintf('--- Result: %s (time: %.3f [s])\n',res.str,res.time);
end

end


% Auxiliary functions -----------------------------------------------------

function [xi,ri,xs,rs] = aux_pop(xs,rs,bSz,options)
    % Obtain the number of elements in the queue.
    nQueue = size(xs,2);
    % Construct indices to pop.
    switch options.nn.verify_dequeue_type
        case 'front'
            % Take the first entries.
            idx = 1:min(bSz,nQueue);
        case 'half-half'
            % Half from the front and half from the back.
            idx = 1:min(bSz,nQueue);
            offsetIdx = ceil(length(idx)/2 + 1):length(idx);
            idx(offsetIdx) = idx(offsetIdx) + nQueue - length(idx);
        otherwise
            % Invalid option.
            throw(CORAerror('CORA:wrongFieldValue', ...
                'options.nn.verify_enqueue_type',{'append','prepend'}));
    end
    % Pop centers.
    xi = xs(:,idx);
    xs(:,idx) = [];
    % Pop radii.
    ri = rs(:,idx);
    rs(:,idx) = [];
end

function [critValPerConstr,critVal,falsified,x_,y_] = ...
    aux_checkPoints(nn,options,idxLayer,A,b,safeSet,xs)
    % Compute the output of the adversarial examples.
    ys = nn.evaluate_(xs,options,idxLayer);
    % Compute the logit difference.
    ld_ys = A*ys;
    % Check the specification and compute a value indicating how close we 
    % are to finding an adversarial example (< 0 mean the specification is 
    % violated).
    critValPerConstr = ld_ys - b;
    if safeSet
        % safe iff all(A*y <= b) <--> unsafe iff any(A*y > b)
        % Thus, unsafe if any(-A*y < -b).
        falsified = any(ld_ys > b,1);
        critValPerConstr = -critValPerConstr;
        critVal = min(critValPerConstr,[],1);
    else
        % unsafe iff all(A*y <= b) <--> safe iff any(A*y > b)
        % Thus, unsafe if all(A*y <= b).
        falsified = all(ld_ys <= b,1);
        critVal = max(critValPerConstr,[],1);
    end

    if any(falsified)
        % Found a counterexample.
        idNzEntry = find(falsified);
        id = idNzEntry(1);
        x_ = gather(xs(:,id));
        % Gathering weights from gpu. There is are precision error when 
        % using single gpuArray.
        nn.castWeights(single(1));
        y_ = nn.evaluate_(x_,options,idxLayer); % yi_(:,id);
    else
        % We have not found a counterexample.
        x_ = [];
        y_ = [];
    end
end

function isContained = aux_isContained(xs,rs)
    % Specify a tolerance.
    tol = 1e-6;

    % Obtain the number of dimensions and batch size.
    [~,bSz] = size(xs);

    % Compute the bounds of the sets.
    ls = permute(xs - rs,[1 2 4 3]);
    us = permute(xs + rs,[1 2 4 3]);

    % We sort the items to prevent a previous batch item to be contained
    % in a previous one.
    [~,idx] = sortrows([ls; -us]');
    ls = ls(:,idx);
    us = us(:,idx);
    
    % We identify the sets that are contained within other sets.
    isContained_ = reshape(any( ...
        all(ls - tol <= permute(ls,[1 3 2 4]),1) ... lower bounds are larger
        & all(permute(us,[1 3 2 4]) <= us + tol,1) ... upper bounds are smaller
        & 1:bSz < permute(1:bSz,[1 3 2]) ... not the same
            ,2),1,bSz,[]);
    % Reorder.
    revIdx(idx) = 1:bSz;
    isContained = isContained_(revIdx);
end

function [cxi,Gxi,dimIdx] = aux_constructInputZonotope(xi,ri,hi, ...
    batchG,numInitGens,options)
    % Obtain the number of input dimensions and the batch size.
    [n0,bSz] = size(xi);

    % Initialize the generator matrix.
    Gxi = batchG(:,:,1:bSz);

    if numInitGens >= n0
        % We create a generator for each input dimension.
        dimIdx = repmat((1:n0)',1,bSz);
    else
        % Find the input pixels that affect the output the most.
        [~,dimIdx] = sort(hi,'descend');
        % Select the most important input dimensions and add a generator
        % for each of them.
        dimIdx = dimIdx(1:numInitGens,:);
    end
    % Compute indices for non-zero entries.
    gIdx = sub2ind(size(Gxi),dimIdx, ...
        repmat((1:numInitGens)',1,bSz),repelem(1:bSz,numInitGens,1));
    % Set non-zero generator entries.
    Gxi(gIdx) = ri(sub2ind(size(ri),dimIdx,repelem(1:bSz,numInitGens,1)));
    % Sum generators to compute remaining set.
    ri_ = (ri - reshape(sum(Gxi,2),[n0 bSz]));

    % Construct the center.
    if options.nn.interval_center
        % Put remaining set into the interval center.
        cxi = permute(cat(3,xi - ri_,xi + ri_),[1 3 2]);
    else
        % The center is just a vector.
        cxi = xi;
    end
end

function [ld_yi,ld_Gyi,ld_Gyi_err,yic,yid,Gyi] = ...
    aux_computeLogitDifference(yi,Gyi,A,options)
    % Obtain number of output dimensions and batch size.
    [nK,~,bSz] = size(Gyi);

    if options.nn.interval_center
        % Compute the center and the radius of the center-interval.
        yic = reshape(1/2*(yi(:,2,:) + yi(:,1,:)),[nK bSz]);
        % Compute approximation error.
        yid = 1/2*(yi(:,2,:) - yi(:,1,:));
    else
        % The center is just a vector.
        yic = yi;
        % There are no approximation errors stored in the center.
        yid = zeros([nK 1 bSz],'like',yi);
    end

    % Compute the logit difference of the input generators.
    ld_yi = A*yic;
    ld_Gyi = pagemtimes(A,Gyi);
    % Compute logit difference of the approximation errors.
    ld_Gyi_err = sum(abs(A.*permute(yid,[2 1 3])),2);
end

% Bounded Polytope Approximation ------------------------------------------

function [bl,bu] = aux_boundsOfBoundedPolytope(A,b,options)
    % Compute the bounds [bl,bu] of a bounded polytope P:
    % Given P=(A,b) \cap [-1,1], compute its bounds, i.e., 
    % [bl,bu]\supseteq {x\in\R^q\mid A\,x\leq b} \cap [-1,1].

    % Specify a numerical tolerance to avoid numerical instability.
    tol = 1e-8;

    % Initialize bounds of the factors.
    bl = -ones(size(A,[2 3]),'like',A);
    bu = ones(size(A,[2 3]),'like',A);

    if ~options.nn.exact_conzonotope_bounds

        % Efficient approximation by isolating the i-th variable. ---------
        % We compute a box-approximation of the valid factor for the 
        % constraint zonotope, 
        % i.e., [\underline{\beta},\overline{\beta}] 
        %   \supseteq \{\beta \in [-1,1]^q \mid A\,\beta\leq b\}.
        % We view each constraint separately and use the tightest bounds.
        % For each constraint A_{(i,\cdot)}\,\beta\leq b_{(i)}, we isolate 
        % each factor \beta_{(j)} and extract bounds:
        % A_{(i,\cdot)}\,\beta\leq b_{(i)} 
        %   \implies A_{(i,j)}\,\beta_{(j)} \leq 
        %       b_{(i)} - \sum_{k=1,...,q, k\neq j} A_{(i,k)}\,\beta_{(k)}
        % Based on the sign of A_{(i,j)} we can either tighten the lower or
        % upper bound of \beta_{(j)}.
    
        % Specify maximum number of iterations.
        maxIter = options.nn.polytope_bound_approx_max_iter;
        
        % Permute the dimension of the constraints for easier handling.
        A_ = permute(A,[2 1 3]);
        b_ = permute(b,[3 1 2]);
        % Reshape factor bounds for easier multiplication.
        bl_ = permute(bl,[1 3 2]);
        bu_ = permute(bu,[1 3 2]);
        % Extract a mask for the sign of the coefficient of the i-th 
        % variable in the j-th constraint.
        nMsk = (A_ < 0);
        pMsk = (A_ > 0);
        % Decompose the matrix into positive and negative entries.
        An = A_.*nMsk;
        Ap = A_.*pMsk;
        % Do summation with matrix multiplication: sum all but the i-th 
        % entry.
        sM = (1 - eye(size(A,2),'like',A));
    
        % Initialize iteration counter.
        iter = 1;
        tighterBnds = 1;
        while tighterBnds && iter <= maxIter
            % Scale the matrix entries with the current bounds.
            ABnd = Ap.*bl_ + An.*bu_;
            % Isolate the i-th variable of the j-th constraint.
            sABnd = pagemtimes(sM,ABnd);
            % Compute right-hand side of the inequalities.
            rh = min(max((b_ - sABnd)./A_,bl_),bu_);
            % Update the bounds.
            bl_ = max(nMsk.*rh + (~nMsk).*bl_,[],2);
            bu_ = min(pMsk.*rh + (~pMsk).*bu_,[],2);
            % Check if the bounds could be tightened.
            tighterBnds = any( ...
                (bl + tol < bl_(:,:) | bu_(:,:) < bu - tol) ... tighter bounds
                    & bl_(:,:) <= bu_(:,:), ... not empty
                'all');
            bl = bl_(:,:);
            bu = bu_(:,:);
            % Increment iteration counter.
            iter = iter + 1;
        end
        % fprintf('--- aux_boundsOfBoundedPolytope --- Iteration: %d\n',iter);

    else
        % Slow implementation with exact bounds for validation.
        
        % Obtain the batch size.
        [p,q,bSz] = size(A);

        for i=1:bSz
            % Obtain parameters of the i-th batch entry.
            Ai = double(gather(A(:,:,i)));
            bi = double(gather(b(:,i)));
            % Construct linear program.
            prob = struct('Aineq',Ai,'bineq',bi, ...
                'lb',-ones(q,1),'ub',ones(q,1));
            if any(isnan(Ai),'all') || any(isnan(bi),'all')
                % The given set is already marked as empty.
                bl(:,i) = NaN;
                bu(:,i) = NaN;
            else
                % Loop over the dimensions.
                for j=1:q
                    % Find the lower bound for the j-th dimension.
                    prob.f = double((1:q) == j);
                    % Solve the linear program.
                    [~,blij,efl] = CORAlinprog(prob);
                    % Find the upper bound for the j-th dimension.
                    prob.f = -double((1:q) == j);
                    % Solve the linear program
                    [~,buji,efu] = CORAlinprog(prob);
                    if efl > 0 && efu > 0
                        % Solutions found; assign values.
                        bl(j,i) = blij;
                        bu(j,i) = -buji;
                    else
                        % No solution; the polytope is empty.
                        bl(:,i) = NaN;
                        bu(:,i) = NaN;
                        continue;
                    end
                end
            end
        end

        % -----------------------------------------------------------------
    end
end

function [l,u,bl,bu] = aux_boundsOfConZonotope(cZs,numUnionConst,options)
    % Input arguments represent a constraint zonotope with inequality
    % constraints.
    % numUnionConst: number of unions constraints; the first #numUnionConst
    % constraints of cZs are unified (needed for safeSet specifications).
    % options.nn.exact_conzonotope_bounds: use linear programs to compute the bounds.
    % options.nn.batch_union_conzonotope_bounds: batch union constraints

    % Extract parameters of the constraint zonotope.
    c = cZs.c;
    G = cZs.G;
    r = cZs.r;
    A = cZs.A;
    b = cZs.b;

    % Obtain number of dimensions, generators, and batch size.
    [n,q,bSz] = size(G);

    % Specify indices of intersection constraints.
    intConIdx = (numUnionConst+1):size(A,1);

    if options.nn.batch_union_conzonotope_bounds
        % The safe set is the union of all constraints. Thus, we 
        % have to create a new set for each constraint.
        % Move union constraints into the batch.
        Au = reshape(permute(A(1:numUnionConst,:,:),[4 2 3 1]),...
            [1 q bSz*numUnionConst]);
        bu = reshape(permute(b(1:numUnionConst,:),[3 2 1]),...
            [1 bSz*numUnionConst]);
        % Replicate intersection constraints.
        Ai = repmat(A(intConIdx,:,:),1,1,numUnionConst);
        bi = repmat(b(intConIdx,:),1,numUnionConst);
        % Append intersection constraints.
        A = cat(1,Au,Ai);
        b = cat(1,bu,bi);
    end

    if options.nn.batch_union_conzonotope_bounds
        % Approximate the bounds of the hypercube (bounded polytope).
        [bl,bu] = aux_boundsOfBoundedPolytope(A,b,options);

        % Unify sets if a safe set is specified.
        bl = min(reshape(bl,[q bSz numUnionConst]),[],3);
        bu = max(reshape(bu,[q bSz numUnionConst]),[],3);
    else
        bl = [];
        bu = [];
        % Loop over the union constraints.
        for k=1:numUnionConst
            % Use the k-th union constraint and all intersection
            % constraints.
            Ak = A([k intConIdx],:,:);
            bk = b([k intConIdx],:,:);
            % Approximate the bounds of the hypercube.
            [blk,buk] = aux_boundsOfBoundedPolytope(Ak,bk,options);
            % Unify constraints.
            if isempty(bl)
                bl = blk;
                bu = buk;
            else
                bl = min(bl,blk);
                bu = max(bu,buk);
            end
        end
    end

    % Map bounds of the factors to bounds of the constraint zonotope. 
    % We use interval arithmetic for that.
    bc = 1/2*permute(bu + bl,[1 3 2]);
    br = 1/2*permute(bu - bl,[1 3 2]);

    % Map bounds of the factors to bounds of the constraint zonotope.
    c = c + reshape(pagemtimes(G,bc),[n bSz]);
    r = r(:,:) + reshape(pagemtimes(abs(G),br),[n bSz]);
    l = c - r;
    u = c + r;

    % Identify empty sets.
    isEmpty = any(bl > bu,1);
    l(:,isEmpty) = NaN;
    u(:,isEmpty) = NaN;
    bl(:,isEmpty) = 0;
    bu(:,isEmpty) = 0;
end

% Set Refinement ----------------------------------------------------------

function [l,u,wasRefined,x,Gx,y,Gy] = aux_refineInputSet(nn,options, ...
    storeInputs,x,Gx,y,Gy,A,b,safeSet,Ai,bi,As,bs,numInitGens,numUnionConst)

    % Specify a numerical tolerance to avoid numerical instability.
    tol = 1e-8;

    % Specify the heurstic for selecting the ReLU-neurons to constrain 
    % (TODO: make this an argument).
    reluConstrHeuristic = 'most-sensitive-approx-error';

    % Specify the maximum number of refinement iterations per layer.
    maxRefIter = options.nn.refinement_max_iter;

    % Extract type of refinement.
    layerwise = strcmp(options.nn.refinement_method,'zonotack-layerwise');

    % Enumerate the layers of the neural networks.
    [layers,ancIdx] = nn.enumerateLayers();

    if layerwise
        % TODO: we cannot refine through composite layer, because we loose
        % dependencies between the computation paths. We recompute
        % the output from the refined layer; we cannot recompute from the
        % input set, there we need to compensate for the approximation
        % errors.

        % We can only refine the top-level computation path (there are no
        % parallel paths).
        % layers = nn.layers;

        % Specify the layer indices.
        refIdxLayer = (1:length(layers));
        % Only refine activation layers.
        refIdxLayer = refIdxLayer(arrayfun(@(i) ...
            isa(layers{i},'nnActivationLayer'),refIdxLayer));

        % Flip the layers for a backward refinement.
        refIdxLayer = [fliplr(refIdxLayer) 1];
    else       
        % We only refine the input.
        refIdxLayer = 1;
    end

    % Obtain number of generators and batchsize.
    [nK,q,bSz] = size(Gy);

    % Pad offsets if there are different number of offsets in general
    % split and input split constraints.
    if size(bs,2) ~= size(bi,2)
        bs = cat(2,bs,NaN([size(bs,1) size(bi,2) - size(bs,2) size(bs,3)]));
        bi = cat(2,bi,NaN([size(bi,1) size(bs,2) - size(bi,2) size(bi,3)]));
    end 
    % Append zeros for generators.
    Ai_ = cat(2,Ai,zeros([size(Ai,1) q-size(Ai,2) size(Ai,3)],'like',Ai));
    % Convert and join the general- & input-split constraints.
    [C,d,newSplits] = aux_convertSplitConstraints([As; Ai_],[bs; bi]);

    if newSplits > 1
        % Replicate set for split constraints.
        if options.nn.interval_center
            x = repelem(x,1,1,newSplits);
            y = repelem(y,1,1,newSplits);
        else
            x = repelem(x,1,newSplits);
            y = repelem(y,1,newSplits);
        end
        Gx = repelem(Gx,1,1,newSplits);
        Gy = repelem(Gy,1,1,newSplits);
    end
    % Update the batch size.
    bSz = bSz*newSplits;

    % Initialize scale and offset of the generators.
    bc = zeros([q bSz],'like',Gy);
    br = ones([q bSz],'like',Gy);

    % Initialize loop variables.
    refIdx = 1; % index into refIdxLayer
    refIter = 1; % Counter for number of refinement iterations of the 
    % current layer.

    % Keep track of empty sets.
    isEmpty = zeros([1 bSz],'logical');

    % Keep track of which inputs sets of which layers need scaling.
    scaleInputSets = ones([1 length(layers)],'logical');

    % Iterate layers in a backward fashion to propagate the constraints
    % through the layers.
    while refIdx <= length(refIdxLayer)
        % Obtain layer index.
        i = refIdxLayer(refIdx);

        % Append the index of the current layer to update its input set 
        % in the next iterations.
        idxLayer = ancIdx(i):length(nn.layers);
   
        % Construct the unsafe output set.
        uYi = aux_constructUnsafeOutputSet(options,y,Gy,A,b,safeSet);

        % Scale and offset constraints with current hypercube.
        [d_,C_] = aux_scaleAndOffsetZonotope(d,C,-bc,br);

        if options.nn.num_relu_constraints > 0
            % Compute tightening constraints for unstable ReLU neurons.
            [At,bt] = aux_reluTightenConstraints(nn,options, ...
                [],reluConstrHeuristic,bc,br,scaleInputSets);
        else
            % There are no relu constraints; 
            At = [];
            bt = [];
        end

        if ~isempty(At)
            % Join the tightening and split constraints.
            C_ = [C_; At]; 
            d_ = [d_; bt];
        end
   
        if ~isempty(C_)
            % Append split constraints.
            uYi.A = [uYi.A; C_];
            % Append the offset.
            uYi.b = [uYi.b; d_]; 
        end

        % Compute the bounds of the unsafe inputs (hypercube).
        [ly,uy,bli,bui] = aux_boundsOfConZonotope(uYi,numUnionConst,options);
        % Update empty sets.
        isEmpty = isEmpty | any(isnan(ly),1) | any(isnan(uy),1);
   
        % Compute the center and radius of the new inner hypercube 
        % (the new hypercube is relative to the current hypercube).
        bci = 1/2*(bui + bli);
        bri = 1/2*(bui - bli);

        % Update the hypercube.
        bc = bc + br.*bci;
        br = br.*bri;

        if layerwise
            % We have to the refine the input set of an ancestor layer.
            % Obtain the input set of the ancestor of the i.th layer.
            layerAnc = nn.layers{ancIdx(i)};
    
            % Obtain number of input generators of the i-th layer.
            ancQiIds = layerAnc.backprop.store.genIds;
            cAnc = layerAnc.backprop.store.inc;    
            GAnc = layerAnc.backprop.store.inG(:,ancQiIds,:);
            if size(Gy,3) ~= size(GAnc,3) % iff newSplits > 1
                % Replicate set for split constraints.
                if options.nn.interval_center
                    cAnc = repelem(cAnc,1,1,newSplits);
                else
                    cAnc = repelem(cAnc,1,newSplits);
                end
                GAnc = repelem(GAnc,1,1,newSplits);
            end
        else
            % We refine the input set of the neural network.
            cAnc = x;
            ancQiIds = 1:numInitGens;
            GAnc = Gx(:,ancQiIds,:);
        end

        % Update scale and offset of the input set to compute a smaller 
        % output set.
        if ~storeInputs || scaleInputSets(ancIdx(i))
            [cAnc,GAnc] = aux_scaleAndOffsetZonotope(cAnc,GAnc,bc,br);
        else
            [cAnc,GAnc] = aux_scaleAndOffsetZonotope(cAnc,GAnc,bci,bri);
        end

        if newSplits > 1
            % TODO: fix the number of sensitivity batch entries might 
            % be incorrect due to split constraints.
            options.nn.approx_error_order = 'length';
        end

        % Compute a new output enclosure.
        [y,Gy] = nn.evaluateZonotopeBatch_(cAnc,GAnc,options,idxLayer);

        if storeInputs
            % New input sets are computed for the layers; update scaling 
            % index.
            scaleInputSets(ancIdx >= ancIdx(i)) = false;
        end
        
        % fprintf([' --- Layer %d (Iteration %d) ' ...
        %     '---> Refined hypercube: [%s] (avg. radius)\n'],i,refIter, ...
        %         join(string(mean(br(1:numInitGens,:),1)),', '));

        % fprintf('aux_refineInputSet (Layer %d) --- Iteration: %d\n',i,refIter);

        % Check if we can further refine the current layer.
        if refIter < maxRefIter && ...
                any(~isEmpty & min(bri,[],1) < 1 - tol,'all')
            % Do another refinement iteration on the current layer.
            refIter = refIter + 1;
        else
            % No more refinement possible (either maximum number of
            % iteration reached or last iteration did not further refine
            % the hypercube).
            refIdx = refIdx + 1;
            % Reset refinement iteration counter.
            refIter = 1;
        end
    end

    % Compute an indicator if a set was refined.
    wasRefined = (min(br,[],1) < 1);
    % Return the refined input set.
    [x,Gx] = aux_scaleAndOffsetZonotope(x,Gx,bc,br);
    % Compute the input bounds.
    r = reshape(sum(abs(Gx),2),size(Gx,[1 3]));

    if options.nn.interval_center
        % Compute center and center radius.
        cl = reshape(x(:,1,:),size(x,[1 3]));
        cu = reshape(x(:,2,:),size(x,[1 3]));
    else
        % The radius is zero.
        cl = x;
        cu = x;
    end
    l = cl - r;
    u = cu + r;

    % Update bounds to represent empty sets.
    l(:,isEmpty) = NaN;
    u(:,isEmpty) = NaN;
end

function uYi = aux_constructUnsafeOutputSet(options,y,Gy,A,b,safeSet)
    % Obtain the number of output dimensions and batch size.
    [nK,~,bSz] = size(Gy);

    if ndims(y) > 2 % iff options.nn.interval_center 
        % Compute center and center radius.
        yc = reshape(1/2*(y(:,2,:) + y(:,1,:)),[nK bSz]);
        yr = 1/2*(y(:,2,:) - y(:,1,:));
    else
        % The radius is zero.
        yc = y;
        yr = 0;
    end

    % Compute the output constraints.
    [ld_yi,ld_Gyi,ld_Gyi_err,~,~,~] = ... 
        aux_computeLogitDifference(y,Gy,A,options);
    % Compute output constraints.
    if safeSet
        % safe iff all(A*y <= b) ...
        % <--> unsafe iff any(A*y > b) <--> unsafe iff any(-A*y < -b)
        % Thus, unsafe if any(-A*Gy*\beta < -b + A*y)
        A_ = -ld_Gyi;
        b_ = ld_yi - b;
    else
        % unsafe iff all(A*y <= b)
        % Thus, unsafe if all(A*Gy*\beta <= b - A*y)
        A_ = ld_Gyi;
        b_ = b - ld_yi;
    end

    % Construct a struct for the output set.
    uYi = struct('c',yc,'r',yr,'G',Gy); 
    % Apply the output constraints to the input set of the i-th layer.
    uYi.A = A_;
    % Offset by refinement errors.
    uYi.b = b_ + ld_Gyi_err(:,:);
end

function uYi = aux_refineSpecificationOffset(y,Gy,options, ...
    A,b,safeSet,numUnionConst)
    % Using binary search we refine the offset of the specification.
    % Therefore, we try to reduce the unsafe output set as long as
    % possible, e.g., check if the safe set is empty.

    % Specify the maximum number of refinement iterations (TODO: make 
    % this an argument).
    maxRefIter = 3;

    % Initialize iteration counter.
    refIter = 0;

    % Obtain the number of constraints.
    [p,~] = size(A);
    % Obtain the batch size.
    [~,~,bSz] = size(Gy);
    % Construct the unsafe output set.
    sYi = aux_constructUnsafeOutputSet(options,y,Gy,A,b,~safeSet);

    % We view the specification as a split constraint. We refine the offset
    % of the constraint as long as the "safe"-side of the constraint is
    % empty, i.e., we include all possible unsafe outputs.
    % The upperbound of the offset represents an empty "safe"-side, while
    % the lower bound of the offset is the lowest possible offset s.t. the
    % constraint still is inside the output set. 
    % E.g., (i) uYi & A*y <= b & A*y >= b - offsetu is empty and 
    % (ii) uYi & A*y >= b - offsetl is empty.

    offsetu = 0;
    offsetl = reshape(sum(abs(sYi.A),2),[p bSz]);

    % Append the specification constraint for the "safe"-side.

    % We want to make the "safe"-side as large as possible.

    while maxRefIter > refIter
        % Compute middle offset.
        offset = 1/2*(offsetu + offsetl);

        % Update offset.
        sYi.b = uYi.b + offset;

        [ly,uy,~,~] = aux_boundsOfConZonotope(uYi,numUnionConst,options);
        % Update empty sets.
        isEmpty = isEmpty | any(isnan(ly),1) | any(isnan(uy),1);

        if isEmpty
            offsetu = offset;
        else 
            offsetl = offset;
        end
        
        % Increment iteration counter.
        refIter = refIter + 1;
    end
end

function [c,G] = aux_scaleAndOffsetZonotope(c,G,bc,br)
    % Obtain indices of generator.
    qiIds = 1:min(size(G,2),size(bc,1));
    % Scale and offset the zonotope to a new hypercube with center bic and 
    % radius bir.
    offset = pagemtimes(G(:,qiIds,:),permute(bc(qiIds,:),[1 3 2]));
    % Offset the center.
    if ndims(c) > 2 % iff options.nn.interval_center
        c = c + offset;
    else
        c = c + offset(:,:);
    end
    % Scale the generators.
    G(:,qiIds,:) = G(:,qiIds,:).*permute(br(qiIds,:),[3 1 2]);
end

function [A,b,newSplits] = aux_convertSplitConstraints(As,bs)
    % Consider all combinations between the given constraints.
    if ~isempty(As)
        % Obtain the number of split-constraints.
        [ps,q,bSz] = size(As);
        % Obtain the number of pieces.
        [~,pcs,~] = size(bs);
        % Compute number of new splits.
        newSplits = (pcs+1)^ps;

        % Duplicate each halfspace for a lower and an upper bound.
        As_ = repelem(As,2,1,1);
        % Scale the constraints; -1 for upper bound and 1 for lower bound.
        As_ = repmat([-1; 1],ps,1).*As_;
        % Duplicate the constraint for the new splits.
        A_ = permute(repelem(As_,1,1,1,newSplits),[2 1 4 3]);
        % Duplicate offsets for lower and upper bound.
        bs_ = repelem(bs,1,2,1);
        % Mark unused bounds by NaN.
        bs_ = cat(2,NaN(ps,1,bSz),bs_,cat(2,NaN(ps,1,bSz)));
        % Scale the offsets; -1 for upper bound and 1 for lower bound.
        bs_ = repmat([-1 1],1,pcs+1).*bs_;
        % Reshape and combine the lower and upper bounds.
        bs_ = reshape(permute(reshape(permute(bs_,[4 2 1 3]), ...
            [2 pcs+1 ps bSz]),[1 3 2 4]),[2*ps pcs+1 bSz]);
        % Extend the offsets.
        b_ = cat(2,bs_,zeros([2*ps newSplits - (pcs+1) bSz],'like',bs_));
        % Compute all combinations of the splits.
        idx = pcs+1;
        for i=1:(ps-1)
            % Increase the index.
            idx_ = idx*(pcs+1);
            % Repeat the current combined splits.
            b_(1:2*i,1:idx_,:) = repmat(b_(1:2*i,1:idx,:),1,pcs+1,1);
            % Repeat the elements of the next split and append them.
            b_(2*i + (1:2),1:idx_,:) = ...
                repelem(b_(2*i + (1:2),1:(pcs+1),:),1,(pcs+1)^i,1);
            % Update the index of the combined splits.
            idx = idx_;
        end

        % Find all unused constraints.
        nanIdx = isnan(b_);
        % Set all not needed constraints to zero.
        A_(:,nanIdx) = 0;
        b_(nanIdx) = 0;

        % Reshape the constraint matrix and offset.
        A = reshape(permute(A_,[2 1 3 4]),[2*ps q newSplits*bSz]);
        b = reshape(b_,[2*ps newSplits*bSz]);
    else
        % There are no additional constraints.
        newSplits = 1;
        A = zeros([0 size(As,[2 3])],'like',As);
        b = zeros([0 size(bs,2)],'like',bs);
    end
end

% Constraints & Splitting -------------------------------------------------

function [xis,ris] = aux_split(xi,ri,hi,nSplits)
    % Split one input dimension into nSplits pieces.
    [n,bSz] = size(xi);
    % Split each input in the batch into nSplits parts.
    % 1. Find the input dimension with the largest heuristic.
    [~,sortDims] = sort(hi,1,'descend');
    dimIds = sortDims(1,:); 
    % Construct indices to use sub2ind to compute the offsets.
    splitsIdx = repmat(1:nSplits,1,bSz);
    bSzIdx = repelem((1:bSz)',nSplits);

    dim = dimIds(1,:);
    linIdx = sub2ind([n bSz nSplits], ...
        repelem(dim,nSplits),bSzIdx(:)',splitsIdx(:)');
    % 2. Split the selected dimension.
    xi_ = xi;
    ri_ = ri;
    % Shift to the lower bound.
    dimIdx = sub2ind([n bSz],dim,1:bSz);
    xi_(dimIdx) = xi_(dimIdx) - ri(dimIdx);
    % Reduce radius.
    ri_(dimIdx) = ri_(dimIdx)/nSplits;
   
    xis = repmat(xi_,1,1,nSplits);
    ris = repmat(ri_,1,1,nSplits);
    % Offset the center.
    xis(linIdx(:)) = xis(linIdx(:)) + (2*splitsIdx(:) - 1).*ris(linIdx(:));
    
    % Flatten.
    xis = xis(:,:);
    ris = ris(:,:);
end

function [Ai,bi] = aux_dimSplitConstraints(hi,nSplits,nDims)
    % Construct dimension split constraints that splits #nDims dimensions 
    % into #nSplits pieces for subsequent refinement.

    % Obtain the number of dimensions and batch size.
    [n,bSz] = size(hi);
    nDims = min(nDims,n);

    % Split each input in the batch into nSplits parts.
    % 1. Find the input dimension with the largest heuristic.
    [~,sortDims] = sort(hi,1,'descend');
    dimIds = sortDims(1:nDims,:); 

    % Compute dimension indices.
    dimIdx = sub2ind([nDims n bSz],repelem((1:nDims)',1,bSz), ...
        dimIds,repelem(1:bSz,nDims,1));

    % 2. Construct the constraints.
    Ai = zeros([nDims n bSz],'like',hi);
    Ai(dimIdx) = 1;
    bi = repelem(-1 + (1:(nSplits-1)).*(2/nSplits),nDims,1,bSz); % Specify offsets.
end

function [Ao,bo] = aux_outputDimSplitConstraints(y,Gy,nSplits,nDims)
    % Construct output split constraints that splits #nDims dimensions 
    % into #nSplits pieces for subsequent refinement.

    % Obtain the number of dimensions and batch size.
    [nK,qK,bSz] = size(Gy);
    nDims = min(nDims,nK);

    % Compute the radius of each output dimension.
    r = reshape(sum(abs(Gy),2),size(Gy,[1 3]));

    % 1. Find the output dimensions with the largest radius.
    [~,sortDims] = sort(r,1,'descend');
    dimIds = sortDims(1:nDims,:); 

    % 2. Construct the constraints.
    Ao = Gy(sub2ind([nK qK bSz], ...
        permute(repelem(dimIds,1,1,qK),[1 3 2]), ...
        repelem((1:qK),nDims,1,bSz), ...
        repelem(permute(1:bSz,[1 3 2]),nDims,qK,1) ...
    ));
    bo = permute(r(sub2ind(size(r),dimIds,repelem(1:bSz,nDims,1))),[1 3 2])...
        .*repelem(-1 + (1:(nSplits-1)).*(2/nSplits),nDims,1,bSz); % Specify offsets.
end

function [As,bs,nrIdx] = aux_neuronConstraints(nn,options, ...
    idxLayer,heuristic,nSplits,nNeur,numInitGens)
    % Assume: input was propagated and stored including sensitivity.
    % Output: 
    % - As, bs: individual constraints for neuron splits nrConst
    %   e.g. A(i,:)*beta <= b(i) and -A(i,:)*beta >= -b(i)
    % - nrIdx: indices of neuron splits

    % Compute batch size.
    % bSz = nnz(unknown);
    
    % Initialize constraints.
    As = [];
    bs = [];
    q = 0; % Number of considered generators.
    p = 0; % Number of constraints.
    % Initial heuristics.
    h = [];
    % Initialize indices of neuron split.
    nrIdx = struct('layerIdx',[],'dimIdx',[]);
    
    % Enumerate the layers of the neural networks.
    [layers,ancIdx] = nn.enumerateLayers();
    
    if isempty(idxLayer)
        idxLayer = 1:length(layers);
    end
    % Compute the indices of ReLU layers.
    idxLayer = idxLayer(arrayfun(@(i) ...
        isa(layers{i},'nnActivationLayer'),idxLayer));
    
    % Iterate through the layers and find max heuristics and propagate
    % constrains.
    for i=idxLayer
        % Obtain i-th layer.
        layeri = layers{i};
        % Obtain the i-th input.
        ci = layeri.backprop.store.inc;
        Gi = layeri.backprop.store.inG; 
        % Obtain the indices of the approximation error generators.
        approxErrGenIds = layeri.backprop.store.approxErrGenIds;

        % Obtain the approximation errors.
        dl = layeri.backprop.store.dl;
        du = layeri.backprop.store.du;
        % Compute center and radius of approximation errors.
        dc = 1/2*(du + dl);
        dr = 1/2*(du - dl);
        % Obtain number of hidden neurons.
        [nk,qi,bSz] = size(Gi);
        % Compute splitting heuristic.
        ri = reshape(sum(abs(Gi),2),[nk bSz]);
        if options.nn.interval_center
            % Obtain lower and upper center bound.
            cl = reshape(ci(:,1,:),[nk bSz]);
            cu = reshape(ci(:,2,:),[nk bSz]);
            ci = 1/2*(cu + cl);
        else
            % The lower and upper center bound are identical.
            cl = ci;
            cu = ci;
        end
        % Compute the bounds.
        li = cl - ri;
        ui = cu + ri;
    
        % Obtain the sensitivity for heuristic.
        Si_ = max(abs(layeri.sensitivity),1e-6);
        sens = reshape(max(Si_,[],1),[nk bSz]);

        switch heuristic
            case 'least-unstable'
                % Least unstable neuron (normalize the un-stability).
                minBnd = 1./min(-li,ui);
                % Compute the heuristic.
                hi = minBnd.*sens;
            case 'most-sensitive-approx-error'
                % Compute the heuristic.
                hi = dr.*sens;
            case 'most-sensitive-input-radius'
                % Compute the heuristic.
                hi = ri.*sens;
            case 'ival-norm-gradient'
                % Obtain the stored gradient of the approximation error
                % w.r.t. the interval norm of the output set.
                ivalGrad = layeri.backprop.store.approx_error_gradients;
                % Compute the heuristic.
                hi = dr.*ivalGrad;
            case 'most-sensitive-input-aligned'
                % Compute an alignment score.
                algnScr = reshape(max(abs(Gi(:,1:numInitGens,:)),[],2)...
                    ./(sum(abs(Gi(:,1:numInitGens,:)),2) + 1e-6),[nk bSz]);
                % Compute the heuristic.
                hi = dr.*ri.*algnScr.*(2^i);
        end

        % Only consider unstable neurons. 
        hi = (li < 0 & 0 < ui).*hi;
        % Prefer earlier layers.
        hi = hi.*1/(2^i);
    
        if q < qi
            % Pad constraints with zeros.
            As = cat(1,As,zeros([qi - q p bSz],'like',As));
            % Update number of constraints.
            q = qi;
        else
            % Pad generators with zeros.
            Gi = cat(2,Gi,zeros([nk q - qi bSz],'like',Gi));
        end
    
        % Append new constraints.
        Asi = Gi;
        As = cat(2,As,permute(Asi,[2 1 3]));

        % % Split into #nSplits pieces around 0.
        % nSplits_ = floor((nSplits-1)/2);
        % splitEnum = 1/(nSplits_+1).*(1:floor((nSplits-1))/2)';
        % bil = flip(splitEnum).*permute(li,[3 1 2]);
        % biu = splitEnum.*permute(ui,[3 1 2]);
        % if mod(nSplits,2) == 0
        %     % Include the center in the lower bounds.
        %     bil = [bil; zeros([1 nk bSz],'like',ci)];
        % end
        % % Combine the bounds.
        % bsi = [bil; biu];

        % Split into #nSplits pieces around the middle.
        splitEnum = linspace(-1,1,nSplits+1)';
        splitEnum = splitEnum(2:end-1);
        bsi = splitEnum.*permute(ri,[3 1 2]);

        % Subtract the center and append the new offsets.
        % bsi = reshape(bsi - permute(ci,[3 1 2]),[nSplits-1 nk bSz]);
        bs = cat(2,bs,bsi);
      
        % % Check if the new constraints even split the input dimensions.
        % doesSplit = aux_doesSplitInput(Asi,permute(bsi,[2 1 3]), ...
        %     numInitGens,options);
        % % Only consider splits that split the hypercube of the input space.
        % hi = hi.*doesSplit;
    
        % Append heuristic and sort.
        [h,idx] = sort([h; hi(:,:)],1,'descend');
    
        % Only keep the constraints for the top neurons.
        nNeur_ = min(nNeur,size(h,1));
        h = h(1:nNeur_,:);

        % Obtain the indices for the relevant constraints.
        sIdx = sub2ind(size(As,2:3), ...
            idx(1:nNeur_,:),repmat(1:bSz,nNeur_,1));
    
        % Extract constraints.
        As = reshape(As(:,sIdx),[q nNeur_ bSz]);
        bs = reshape(bs(:,sIdx),[nSplits-1 nNeur_ bSz]);
    
        % Update indices.
        nrIdx.layerIdx = [nrIdx.layerIdx; repelem(i,nk,bSz)];
        nrIdx.layerIdx = reshape(nrIdx.layerIdx(idx(1:nNeur_,:)),...
            [nNeur_ bSz]);
        nrIdx.dimIdx = [nrIdx.dimIdx; repmat((1:nk)',1,bSz)];
        nrIdx.dimIdx = reshape(nrIdx.dimIdx(idx(1:nNeur_,:)),...
            [nNeur_ bSz]);
    end

    % Transpose constraint matrix.
    As = permute(As,[2 1 3]);
    bs = permute(bs,[2 1 3]);

    if options.nn.add_orth_neuron_splits
        % Add the orthogonal neuron splits.

        % Obtain the number of constraints.
        [p,~,~] = size(As);

        % Extract the most important input dimension; make the constraint
        % orthogonal w.r.t. that dimension constraints.
        As_ = max(abs(As(:,1:numInitGens,:)),1e-6);
        [~,dimIds] = max(As_,[],2);

        % 1. Generate unit vector along most important dimension.
        v = zeros([p numInitGens bSz],'like',As_);
        dimIdx = sub2ind([p numInitGens bSz], ...
            repmat((1:p)',1,bSz),dimIds(:,:),repelem(1:bSz,p,1));
        v(dimIdx) = 1;
        % Move everything into the batch for easier computations.
        As_ = reshape(permute(As_,[2 1 3]),[1 p*numInitGens*bSz]);
        v = reshape(permute(v,[2 1 3]),[1 p*numInitGens*bSz]);
        
        % 2. Make the vector orthogonal to the input dimensions 
        % of the split constraints.
        proj = As_.*pagemtimes(v,'none',As_,'transpose') ...
            ./pagemtimes(As_,'none',As_,'transpose');
        v_orth = permute(reshape(v - proj,[numInitGens p bSz]),[2 1 3]);
        
        % 3. Normalize the orthogonal vector and embed into the full space.
        As_orth = As;
        As_orth(:,1:numInitGens,:) = v_orth./pagenorm(v_orth);
        
        % 4. Append the orthogonal constraints (the offsets are identical
        % because we rotate the constraints along the origin).
        As = [As; As_orth];
        bs = repmat(bs,2,1,1); % [bs; zeros(size(bs),'like',bs)];
    end
end

function doesSplit = aux_doesSplitInput(As,bs,numInitGens,options)
    % Obtain the number of constraints, number of generators, and batch
    % size.
    [p,q,bSz] = size(As);
    % Obtain the number of offsets.
    [~,nOffs,~] = size(bs);
    % Move all splits into the batch.
    As_ = reshape(repelem( ...
        permute(As,[2 1 3]),1,nOffs,1),[1 q p*nOffs*bSz]);
    bs_ = reshape(bs,[1 p*nOffs*bSz]);
    % Compute the bounds of the split polytope.
    options.nn.polytope_bound_approx_max_iter = 1;
    [bl_,bu_] = aux_boundsOfBoundedPolytope(As_,bs_,options);
    % Reshape the computed bounds.
    bl = reshape(bl_(1:numInitGens,:),[numInitGens p nOffs bSz]);
    bu = reshape(bu_(1:numInitGens,:),[numInitGens p nOffs bSz]);
    % Check if any bounds could be tightend; then there is a split.
    doesSplit = reshape(any(-1 < bl | bu < 1,[1 3]),[p bSz]);
end

function [At,bt] = aux_reluTightenConstraints(nn,options,idxLayer, ...
    heuristic,bc,br,scaleInputSets)
    % Assume: input was propagated and stored.
    % Output: 
    % - At, bt: constraints for unstable neurons, 
    %   i.e., ReLU(x) >= 0 and ReLU(x) >= x

    % Obtain the number of constraints.
    numConstr = options.nn.num_relu_constraints;
    
    % Initialize constraints.
    q = 0; % Number of considered generators.
    p = 0; % Number of constraints.
    % (i) ReLU(x) >= 0
    At0 = [];
    bt0 = [];
    % (ii) ReLU(x) >= x
    Atd = [];
    btd = [];
    % Initial heuristics.
    h = [];
    % Initialize indices of neuron split.
    nrIdx = struct('layerIdx',[],'dimIdx',[]);
    
    % Enumerate the layers of the neural networks.
    [layers,~,~,succIdx] = nn.enumerateLayers();

    if isempty(scaleInputSets)
        scaleInputSets = zeros([1 length(layers)],'logical');
    end

    if isempty(idxLayer)
        idxLayer = 1:length(layers);
    end
    % Compute the indices of ReLU layers.
    idxLayer = idxLayer( ...
        arrayfun(@(i) isa(layers{i},'nnReLULayer'),idxLayer));
    
    % Iterate through the layers and find maximal unstable neurons.
    for i=idxLayer
        % Obtain i-th layer.
        layeri = layers{i}; 
        % Obtain successor of the i-th layer.
        layerSucci = layers{succIdx(i)}; 
        % Obtain the i-th input center.
        ci = layeri.backprop.store.inc;
        % Obtain the i-th input generators.
        Gi = layeri.backprop.store.inG;
        % Obtain the outputs.
        co = layerSucci.backprop.store.inc;
        Go = layerSucci.backprop.store.inG;

        % Obtain the number of generators and the batch size.
        [nk,qi,bSz] = size(Go);
    
        % Obtain the slope and approximation errors.
        dl = layeri.backprop.store.dl;
        du = layeri.backprop.store.du;
        % Compute center and radius of approximation errors.
        dr = 1/2*(du - dl);

        % Compute center and error radii.
        ri = reshape(sum(abs(Gi),2),[nk bSz]);
        if options.nn.interval_center
            % Obtain lower and upper center bound.
            cil = reshape(ci(:,1,:),[nk bSz]);
            ciu = reshape(ci(:,2,:),[nk bSz]);
            cou = reshape(co(:,2,:),[nk bSz]);
        else
            % The lower and upper center bound are identical.
            cil = ci;
            ciu = ci;
            cou = co;
        end
        % Compute the bounds.
        li = cil - ri;
        ui = ciu + ri;

        % Obtain the sensitivity for heuristic.
        Si_ = max(abs(layeri.sensitivity),1e-6);
        sens = reshape(max(Si_,[],1),nk,[]);
        if size(sens,2) < bSz
            % Duplicate the sensitivity (there was neuron splitting involved).
            newSplits = bSz/size(Si,3);
            sens = repmat(sens,1,newSplits);
        end

        if q < qi
            % Pad constraints with zeros.
            At0 = cat(1,At0,zeros([qi - q p bSz],'like',At0));
            Atd = cat(1,Atd,zeros([qi - q p bSz],'like',Atd));
            % Update number of constraints.
            q = qi;
        end

        if q > size(Gi,2)
            % Pad generators with zeros.
            Gi = cat(2,Gi,zeros([nk q - size(Gi,2) bSz],'like',Gi));
        end
        if q > size(Go,2)
            % Pad generators with zeros.
            Go = cat(2,Go,zeros([nk q - size(Go,2) bSz],'like',Go));
        end
    
        % (i) ReLU(x) >= 0 
        % --> co + Go*\beta + dr >= 0 <--> -Go*\beta <= co
        Ati0 = -Go;
        bti0 = cou;
        % Append new constraints.
        At0 = cat(2,At0,permute(Ati0,[2 1 3]));
        bt0 = [bt0; bti0];
    
        % (ii) ReLU(x) >= x 
        % --> co + Go*\beta + dr >= ReLU(x) >= x = ci + Gi*\beta
        % <--> (Gi-Go)*\beta - dr <= co - ci
        % Compute difference of generator matrices.
        Atid = Gi - Go;
        btid = cou - cil;
        % Append new constraints.
        Atd = cat(2,Atd,permute(Atid,[2 1 3]));
        btd = [btd; btid];

        switch heuristic
            case 'least-unstable'
                % Least unstable neuron (normalize the un-stability).
                minBnd = 1./min(-li,ui);
                hi = minBnd.*sens;
            case 'most-sensitive-approx-error'
                hi = dr.*sens;
            case 'most-sensitive-input-radius'
                hi = ri.*sens;
        end

        % Only consider unstable neurons. 
        hi = (li < 0 & 0 < ui).*hi;
        % Prefer earlier layers.
        hi = hi.*1/(10^i);

        % Append heuristic and sort.
        [h,idx] = sort([h; hi(:,:)],1,'descend');
        % Only keep the constraints for the top neurons.
        numConstr_ = min(numConstr,size(h,1));
        h = h(1:numConstr_,:);
    
        % Obtain the indices for the relevant constraints.
        cIdx = sub2ind(size(At0,2:3), ...
            idx(1:numConstr_,:),repmat(1:bSz,numConstr_,1));
    
        % Select the relevant constraints.
        At0 = reshape(At0(:,cIdx),[q numConstr_ bSz]);
        bt0 = reshape(bt0(cIdx),[numConstr_ bSz]);
        Atd = reshape(Atd(:,cIdx),[q numConstr_ bSz]);
        btd = reshape(btd(cIdx),[numConstr_ bSz]);
        % Update number of constraints.
        p = size(At0,2);

        % Update indices.
        nrIdx.layerIdx = [nrIdx.layerIdx; repelem(i,nk,bSz)];
        nrIdx.layerIdx = reshape(nrIdx.layerIdx(idx(1:numConstr_,:)),...
            [numConstr_ bSz]);
        nrIdx.dimIdx = [nrIdx.dimIdx; repmat((1:nk)',1,bSz)];
        nrIdx.dimIdx = reshape(nrIdx.dimIdx(idx(1:numConstr_,:)),...
            [numConstr_ bSz]);
    end
    % Transpose constraint matrix.
    At = permute(cat(2,At0,Atd),[2 1 3]);
    bt = [bt0; btd];
    nrIdx.layerIdx = repmat(nrIdx.layerIdx,2,1);
    nrIdx.dimIdx = repmat(nrIdx.dimIdx,2,1);

    if size(At,3) < size(bc,2)
        % Duplicate the constraints (there was neuron splitting involved).
        newSplits = size(bc,2)/size(At,3);
        At = repmat(At,1,1,newSplits);
        bt = repmat(bt,1,newSplits);
        nrIdx.layerIdx = repmat(nrIdx.layerIdx,1,newSplits);
        nrIdx.dimIdx = repmat(nrIdx.dimIdx,1,newSplits);
    end

    % Identify which constraints we have to scale.
    scaleIdx = scaleInputSets(nrIdx.layerIdx);
    if any(scaleIdx,'all') && ~isempty(At)
        % Scale and offset the constraints with the current hypercube.
        [bt_scl,At_scl] = aux_scaleAndOffsetZonotope(bt,At,-bc,br);
        % Permute the constraints for logical indexing.
        At_ = permute(At,[2 1 3]);
        At_scl = permute(At_scl,[2 1 3]);
        % Replace the scaled constraints.
        At_(:,scaleIdx) = At_scl(:,scaleIdx);
        bt(scaleIdx) = bt_scl(scaleIdx);
        % Permute the constraints to correct dimensions.
        At = permute(At_,[2 1 3]);
    end
end

% Plotting ----------------------------------------------------------------

function cZeq = aux_2ConZonoWithEqConst(cZineq,apprErr)
    % Extract parameters of the constraint zonotope.
    c = double(gather(cZineq.c));
    G = double(gather(cZineq.G));
    r = double(gather(cZineq.r));
    A = double(gather(cZineq.A));
    b = double(gather(cZineq.b));

    % We convert the inequality constraints to equality constraints by 
    % adding a slack variable.

    % Obtain number of dimensions, generators, and batch size.
    [n,q,bSz] = size(G);
    % Obtain number of constraints.
    [p,~] = size(A);

    cZeq.c = c;
    % Add the radius to the generators.
    if any(r ~= 0,'all')
        G = cat(2,G,r.*eye(n));
        A = cat(2,A,zeros([p n bSz]));
    end
    % Add a slack variable.
    cZeq.G = cat(2,G,zeros([n p bSz]));
    % Compute scale for the slack variable.
    s = 1/2*(sum(abs(A),2) + permute(b,[1 3 2]));
    cZeq.A = cat(2,A,eye(p).*s);
    % Compensate for the slack variable.
    cZeq.b = b - s(:,:);
    % Set the approximation errors.
    cZeq.apprErr = double(gather(apprErr));
end

function [fig,hx0,hspec] = aux_initPlot(fig,plotDims,xs,ys,x0,r0,A,b,safeSet)
    % Plot the initial input set.
    subplot(1,2,1); hold on;
    title('Input Space')
    % Plot the initial input set.
    % plotPoints(xs,plotDims(1,:),'.k');
    hx0 = plot(interval(x0 - r0,x0 + r0),plotDims(1,:), ...
        'DisplayName','Input Set', ...
        'EdgeColor',CORAcolor('CORA:simulations'),'LineWidth',2);

    % Construct the halfspace specification.
    spec = polytope(A,b);

    % Plot the specification.
    subplot(1,2,2); hold on;
    title('Output Space')
    if safeSet
        safeSetStr = 'safe';
    else
        safeSetStr = 'unsafe';
    end

    plotPoints(ys,plotDims(2,:),'.k');
    hspec = plot(spec,plotDims(2,:),...
        'DisplayName',sprintf('Specification (%s)',safeSetStr), ...
        'FaceColor',CORAcolor(sprintf('CORA:%s',safeSetStr)),'FaceAlpha',0.2, ...
        'EdgeColor',CORAcolor(sprintf('CORA:%s',safeSetStr)),'LineWidth',2);
end

function [fig,hxi,hx,hxv,hy,hyv] = aux_plotInputAndOutputSets(fig, ...
    plotDims,x0,r0,res)
    % Obtain number of dimensions.
    [n,~] = size(x0);
    % Small interval to avoid plotting errors.
    pI = 1e-8*interval(-ones(n,1),ones(n,1));

    % Plot the input sets.
    subplot(1,2,1); hold on;
    % Plot the initial input set.
    hxi = plot(interval(x0 - r0,x0 + r0),plotDims(1,:), ...
        'DisplayName','Input Set', ...
        'EdgeColor',CORAcolor('CORA:simulations'),'LineWidth',2);
    % Store plot handles for potential deletion.
    hx = {};
    hxv = {};
    for j=1:size(res.Xs{end}.c,2)
        Xij = zonotope(res.Xs{end}.c(:,j),res.Xs{end}.G(:,:,j)) + pI;
        if res.Xs{end}.verified(j)
            hxv{end+1} = plot(Xij,plotDims(1,:), ...
                'DisplayName','Input Set (verified)', ...
                'FaceColor',CORAcolor('CORA:color2'),'FaceAlpha',0.5, ...
                'EdgeColor',CORAcolor('CORA:color2'),'LineWidth',2);
        else
            hx{end+1} = plot(Xij,plotDims(1,:), ...
                'DisplayName','Input Set', ...
                ... 'FaceColor',CORAcolor('CORA:reachSet'),'FaceAlpha',0.2, ...
                'EdgeColor',CORAcolor('CORA:reachSet'),'LineWidth',2);
        end
    end
    % Plot the output sets.
    subplot(1,2,2); hold on;
    % Store plot handles for potential deletion.
    hy = {};
    hyv = {};
    for j=1:size(res.Ys{end}.c,2)
        Yij = zonotope(res.Ys{end}.c(:,j),res.Ys{end}.G(:,:,j)) + pI;
        if res.Xs{end}.verified(j)
            hyv{end+1} = plot(Yij,plotDims(2,:),'DisplayName','Output Set', ...
                ...'FaceColor',CORAcolor('CORA:reachSet'),'FaceAlpha',0.2, ...
                'EdgeColor',CORAcolor('CORA:color2'),'LineWidth',2);
        else
            hy{end+1} = plot(Yij,plotDims(2,:),'DisplayName','Output Set', ...
                ...'FaceColor',CORAcolor('CORA:reachSet'),'FaceAlpha',0.2, ...
                'EdgeColor',CORAcolor('CORA:reachSet'),'LineWidth',2);
        end
    end
end

function [fig,hxs_,hys_] = aux_plotCounterExampleCandidates(fig, ...
    plotDims,res)
    % Plot inputs.
    subplot(1,2,1); hold on;
    hxs_ = plotPoints(res.xs_{end},plotDims(1,:),'or', ...
        'DisplayName','Counterexample Candidate');
    % Plot outputs.
    subplot(1,2,2); hold on;
    hys_ = plotPoints(res.ys_{end},plotDims(2,:),'or', ...
        'DisplayName','Counterexample Candidate');
end

function [fig,huy,huy_,hux,hx,hx_] = ...
    aux_plotUnsafeOutputAndNewInputSets(fig,plotDims,res,lis,uis, ...
        isContained,splitsPerUnsafeSet)
    % Obtain number of dimensions.
    [n,~] = size(lis);
    % Small interval to avoid plotting errors.
    pI = 1e-8*interval(-ones(n,1),ones(n,1));

    % Store plot handles for potential deletion.
    huy = {};
    huy_ = {};
    if isfield(res,'uYs')
        % Plot unsafe output constraint zonotope.
        subplot(1,2,2); hold on;
        for j=1:size(res.uYs{end}.c,2)
            % Plot with approximation error.
            % uYij_ = conZonotope( ...
            %     res.uYs{end}.c(:,j),res.uYs{end}.G(:,:,j),...
            %     res.uYs{end}.A(:,:,j),res.uYs{end}.b(:,j) ...
            %         + res.uYs{end}.apprErr(:,j)) + pI;
            % huy_{end+1} = plot(uYij_,plotDims(2,:),'--', ...
            %     'DisplayName','Output Set (unsafe, w. Approx. Err.)', ...
            %     'EdgeColor',CORAcolor('CORA:highlight1'),'LineWidth',1, ...
            %     'FaceColor',CORAcolor('CORA:reachSet'),'FaceAlpha',0.2 ...
            %     );
            % Plot without approximation error.
            uYij = conZonotope( ...
                res.uYs{end}.c(:,j),res.uYs{end}.G(:,:,j),...
                res.uYs{end}.A(:,:,j),res.uYs{end}.b(:,j)) + pI;
            huy{end+1} = plot(uYij,plotDims(2,:), ...
                'DisplayName','Output Set (unsafe)', ...
                'FaceColor',CORAcolor('CORA:highlight1'),'FaceAlpha',0.2, ...
                'EdgeColor',CORAcolor('CORA:highlight1'),'LineWidth',2);
        end
    end
    % Plot new input sets.
    subplot(1,2,1); hold on;
    % Store plot handles for potential deletion.
    hux = {};
    hx = {};
    hx_ = {};
    for j=1:size(lis,2)
        if isfield(res,'uXs') && mod(j-1,splitsPerUnsafeSet) == 0
            j_ = (j-1)/splitsPerUnsafeSet + 1;
            % Plot unsafe input constraint zonotope.
            uXij = conZonotope( ...
                res.uXs{end}.c(:,j_),res.uXs{end}.G(:,:,j_), ...
                res.uXs{end}.A(:,:,j_),res.uXs{end}.b(:,j_)) + pI;
            hux{end+1} = plot(uXij,plotDims(1,:), ...
                'DisplayName','Input Set (unsafe)', ...
                'EdgeColor',CORAcolor('CORA:highlight1'),'LineWidth',1, ...
                'FaceColor',CORAcolor('CORA:reachSet'),'FaceAlpha',0.2 ...
                );
        end
        % Obtain new input set.
        Xij = interval(lis(:,j),uis(:,j)) + pI;
        if isempty(isContained) || ~isContained(j)
            hx{end+1} = plot(Xij,plotDims(1,:), ...
                'DisplayName','Input Set', ...
                'EdgeColor',CORAcolor('CORA:simulations'),'LineWidth',2);
        else
            hx_{end+1} = plot(Xij,plotDims(1,:),'--', ...
                'DisplayName','Input Set', ...
                'EdgeColor',CORAcolor('CORA:simulations'),'LineWidth',1);
        end
    end
end

% ------------------------------ END OF CODE ------------------------------
