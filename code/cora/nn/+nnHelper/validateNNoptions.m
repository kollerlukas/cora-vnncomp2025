function options = validateNNoptions(options,varargin)
% validateNNoptions - checks input and sets default values for the
%    options.nn struct for the neuralNetwork/evaluate function.
%
% Syntax:
%    options = nnHelper.validateNNoptions(options,setTrainFields)
%
% Inputs:
%    options - struct (see neuralNetwork/evaluate)
%    setTrainFields - bool, decide if training parameters should be
%       validated (see neuralNetwork/train)
%
% Outputs:
%    options - updated options
%
% Other m-files required: none
% Subfunctions: none
% MAT-files required: none
%
% See also: neuralNetwork/evaluate

% Authors:       Tobias Ladner, Lukas Koller
% Written:       29-November-2022
% Last update:   16-February-2023 (poly_method)
%                01-March-2023 (backprop)
%                21-February-2024 (merged options.nn)
% Last revision: ---

% ------------------------------ BEGIN CODE -------------------------------

% default values
persistent defaultFields
if isempty(defaultFields)
    defaultFields = {
       'bound_approx', true;
       'poly_method', @aux_defaultPolyMethod;
       'num_generators', [];
       'max_gens_post', @(options) options.num_generators * 100;
       'add_approx_error_to_GI', false;
       'plot_multi_layer_approx_info', false;
       'reuse_bounds', false;
       'max_bounds', 5;
       'do_pre_order_reduction', true;
       'remove_GI', @(options) ~options.add_approx_error_to_GI;
       'force_approx_lin_at', Inf;
       'propagate_bounds', @(options) options.reuse_bounds;
       'sort_exponents', false;
       'maxpool_type', 'project';
       'order_reduction_sensitivity', false;
       'use_approx_error', true;
       'train', struct('backprop', false);
       'interval_center', false;
       'store_sensitivity',false;
       'use_dlconv',true; % Use deep learning toolbox convolution.
       % gnn
       'graph', graph();
       'idx_pert_edges', [];
       'invsqrt_order', 1;
       'invsqrt_use_approx_error', true;
       % evaluateZonotopeBatch
       'use_approx_error', true;
       'approx_error_order', 'sequential';
       'train', struct('backprop', false);
       % Zonotope Batch propagation with interval center
       'interval_center', false;
       'store_approx_error_gradients', false;
       % Batch norm
       'batch_norm_moving_stats', false;
       'batch_norm_calc_stats', false;
       % neuralNetwork/verify
       'falsification_method','center';
       'refinement_method','naive';
       'refinement_max_iter',5;
       'init_split',[1 1];
       'num_splits',2;
       'num_dimensions',1;
       'num_neuron_splits',0;
       'add_orth_neuron_splits',false;
       'exact_conzonotope_bounds',false;
       'batch_union_conzonotope_bounds',true;
       'max_verif_iter',inf;
       'num_relu_constraints',0;
       'polytope_bound_approx_max_iter',5;
       'verify_dequeue_type','half-half';
       'verify_enqueue_type','prepend';
    };
end
% default training parameter values
persistent defaultTrainFields
if isempty(defaultTrainFields) % TODO sort and add comments
    defaultTrainFields = {
       % general
       'use_gpu', aux_isGPUavailable();
       'optim', nnAdamOptimizer; % optimizer
       'max_epoch', 10; % maximum number of training epochs
       'mini_batch_size', 64; % batch size
       'loss','mse'; % loss type
       'lossFun', @(options) @(t,y) 0; % custom loss
       'lossDer', @(options) @(t,y) 0; % gradient of custom loss
       'backprop', true; % enable backpropagation
       'shuffle_data', 'never'; % shuffle the training data
       'early_stop', inf; % early stopping when val loss is not decreasing
       'lr_decay', 1; % factor for learning rate decay
       'lr_decay_epoch', []; % epochs for learning rate decay
       'val_freq', 50; % validation frequency
       'print_freq', 50; % verbose training output printing frequency
       'shuffle_data', 'never'; % shuffle training data
       'early_stop', inf; % number of epoch with non-decreasing validation loss for early stopping
       % projected gradient descent attack
       'pgd_iterations', 0;
       'pgd_stepsize', 0.01;
       'pgd_stepsize_decay', 1;
       'pgd_stepsize_decay_iter' [];
       % Robust training parameters
       'noise', 0; % perturbation radius
       'input_space_inf', 0; % bounds of the input space
       'input_space_sup', 1;
       'warm_up', 0; % number of 'warm-up' training epochs (noise=0)
       'ramp_up', 0; % noise=0 is linearly increase from 'warm-up' to 'ramp-up'
       'method', 'point'; % training method
       % Robust training parameters
       % IBP weighting
       'kappa', 1/2; % IBP weighting factor
       % TRADES or SABR weighting
       'lambda', 0; % SABR & TRADES weighting factor
       % Set training
       'volume_heuristic', 'interval';
       'tau', 0; % weighting factor
       'zonotope_weight_update', 'center'; % compute weight update
       'exact_backprop', true; % exact gradient computations of image enc
       'num_approx_err', inf; % maximum number of approximation errors per nonlinear layer
       'num_init_gens', inf; % maximum number of input generators
       'init_gens', 'l_inf'; % type of input generators
       'propagation_batch_size', inf;
    };
end

setTrainFields = setDefaultValues({false},varargin);

% check if any nn options are given
if ~isfield(options,'nn')
    options.nn = struct;
end
if setTrainFields && ~isfield(options.nn,'train')
    options.nn.train = struct;
end

% set default value of fields if required
options.nn = nnHelper.setDefaultFields(options.nn,defaultFields);

% set default training parameters
if setTrainFields
    options.nn.train = ...
        nnHelper.setDefaultFields(options.nn.train,defaultTrainFields);
end

% check fields
if CHECKS_ENABLED
    structName = inputname(1);
    
    % bound_approx
    aux_checkFieldClass(options.nn, 'bound_approx', ...
        {'logical', 'string'}, structName);
    if isa(options.nn.bound_approx, 'string')
        aux_checkFieldStr(options.nn, 'bound_approx', {'sample'}, structName)
        CORAwarning('CORA:nn',"Choosing Bound estimation '%s' does not lead to safe verification!", ...
            options.nn.bound_approx);
    end
    
    % poly_method
    validPolyMethod = {'regression', 'ridgeregression', ...
    'throw-catch', 'taylor', 'singh', 'bounds', 'center'};
    aux_checkFieldStr(options.nn, 'poly_method', validPolyMethod, structName);
    
    % num_generators
    aux_checkFieldClass(options.nn, 'num_generators', {'double'}, structName);
    
    % add_approx_error_to_GI
    aux_checkFieldClass(options.nn, 'add_approx_error_to_GI', ...
        {'logical'}, structName);
    
    % plot_multi_layer_approx_info
    aux_checkFieldClass(options.nn, 'plot_multi_layer_approx_info', ...
        {'logical'}, structName);
    
    % reuse_bounds
    aux_checkFieldClass(options.nn, 'reuse_bounds', {'logical'}, structName);
    
    % max_bounds
    aux_checkFieldClass(options.nn, 'max_bounds', {'double'}, structName);
    
    % reuse_bounds
    aux_checkFieldClass(options.nn, 'do_pre_order_reduction', ...
        {'logical'}, structName);
    
    % max_gens_post
    aux_checkFieldClass(options.nn, 'max_gens_post', {'double'}, structName);
    
    % remove_GI
    aux_checkFieldClass(options.nn, 'remove_GI', {'logical'}, structName);
    
    % force_approx_lin_at
    aux_checkFieldClass(options.nn, 'force_approx_lin_at', {'double'}, structName);
    
    % propagate_bounds
    aux_checkFieldClass(options.nn, 'propagate_bounds', {'logical'}, structName);
    
    % sort_exponents
    aux_checkFieldClass(options.nn, 'sort_exponents', {'logical'}, structName);
    
    % maxpool_type
    aux_checkFieldStr(options.nn, 'maxpool_type', {'project', 'regression'}, structName);
    
    % order_reduction_sensitivity
    aux_checkFieldClass(options.nn, 'order_reduction_sensitivity', {'logical'}, structName);

    % gnn ---
    
    % graph
    aux_checkFieldClass(options.nn, 'graph', {'graph'}, structName);

    % idx_pert_edges
    aux_checkFieldClass(options.nn, 'idx_pert_edges', {'double'}, structName);

    % invsqrt_order
    aux_checkFieldClass(options.nn, 'invsqrt_order', {'double'}, structName);

    % invsqrt_use_approx_error
    aux_checkFieldClass(options.nn, 'invsqrt_use_approx_error', {'logical'}, structName);

    % check training fields ---

    % Zonotope batch propagation.

    % use_approx_error
    aux_checkFieldClass(options.nn,'use_approx_error',{'logical'},structName);
    aux_checkFieldStr(options.nn,'approx_error_order',{'sequential', ...
        'random','length','sensitivity*length'},structName);

    if setTrainFields
        aux_checkFieldClass(options.nn.train,'optim', ...
            {'nnSGDOptimizer','nnAdamOptimizer'},structName);
        % aux_checkFieldClass(options.nn.train,'max_epoch', ...
        %     {'nonnegative'},structName);
        % aux_checkFieldClass(options.nn.train,'mini_batch_size', ...
        %     {'nonnegative'},structName);
        aux_checkFieldStr(options.nn.train,'loss', ...
            {'mse','softmax+log','custom'},structName);
        % aux_checkFieldClass(options.nn.train,'noise', ...
        %     {'nonnegative'},structName);

        % aux_checkFieldClass(options.nn.train,'warm_up', ...
        %     {'nonnegative'},structName);
        % aux_checkFieldClass(options.nn.train,'ramp_up', ...
        %     {'nonnegative'},structName);
        if options.nn.train.warm_up > options.nn.train.ramp_up
            throw(CORAerror('CORA:wrongFieldValue', ...
                sprintf(['options.nn.train.warm_up cannot be greater ' ...
                    'than options.nn.train.ramp_up'])));
        end

        aux_checkFieldStr(options.nn.train,'shuffle_data', ...
            {'never','every_epoch'},structName);

        % aux_checkFieldClass(options.nn.train,'lr_decay', ...
        %     {'nonnegative'},structName);
        % aux_checkFieldClass(options.nn.train,'lr_decay_epoch', ...
        %     {'nonnegative','vector'},structName);
        % aux_checkFieldClass(options.nn.train,'early_stop', ...
        %     {'nonnegative','vector'},structName);
        % aux_checkFieldClass(options.nn.train,'val_freq', ...
        %     {'nonnegative','vector'},structName);
        % aux_checkFieldClass(options.nn.train,'print_freq', ...
        %     {'nonnegative','vector'},structName);

        aux_checkFieldStr(options.nn.train,'method', ...
            {'point','set','madry','gowal','trades','sabr',...
            'rand','extreme','naive','grad'},structName);
        % 'set' training parameters
        aux_checkFieldStr(options.nn.train,'volume_heuristic', ...
            {'interval','f-radius'},structName);
        aux_checkFieldStr(options.nn.train,'zonotope_weight_update', ...
            {'center','sum'},structName);
        % aux_checkFieldClass(options.nn.train,'num_approx_err', ...
        %     {'nonnegative'},structName);
        if (options.nn.use_approx_error && (options.nn.train.num_approx_err == 0) && ~options.nn.interval_center) ...
                || (~options.nn.use_approx_error && options.nn.train.num_approx_err > 0)
            throw(CORAerror('CORA:wrongValue','options.nn.use_approx_error',...
                'options.nn.use_approx_error has to be true when options.nn.train.num_approx_err > 0'));
        end
        % aux_checkFieldClass(options.nn.train,'num_init_gens', ...
        %     {'nonnegative'},structName);
        aux_checkFieldStr(options.nn.train,'init_gens', ...
            {'l_inf','random','sensitivity','fgsm','fgsm1','fgsm2','fgsm3','patches'},structName);
        % Regression poly method is not supported for training
        aux_checkFieldStr(options.nn,'poly_method', ...
            {'bounds','singh','center'},structName);
        % 'madry' training parameters
        % aux_checkFieldClass(options.nn.train,'pgd_iterations', ...
        %     {'nonnegative'},structName);
        % aux_checkFieldClass(options.nn.train,'pgd_stepsize', ...
        %     {'nonnegative'},structName);
        % 'gowal' training parameters
        % aux_checkFieldClass(options.nn.train,'kappa', ...
        %     {'nonnegative'},structName);
        % 'trades' training parameters
        % aux_checkFieldClass(options.nn.train,'lambda', ...
        %     {'nonnegative'},structName);
        % 'sabr' training parameters
        % aux_checkFieldClass(options.nn.train,'pgd_stepsize_decay', ...
        %     {'nonnegative'},structName);
        % aux_checkFieldClass(options.nn.train,'pgd_stepsize_decay_iter', ...
        %     {'nonnegative','vector'},structName);
    end

    % Check neuralNetwork/verify fields.
    aux_checkFieldStr(options.nn,'falsification_method', ...
        {'center','fgsm','zonotack'},structName);
    aux_checkFieldStr(options.nn,'refinement_method', ...
        {'naive','zonotack','zonotack-layerwise'},structName);

    % aux_checkFieldClass(options.nn,'num_splits', ...
    %     {'nonnegative'},structName);
    % aux_checkFieldClass(options.nn,'num_dimensions', ...
    %     {'nonnegative'},structName);
    % aux_checkFieldClass(options.nn,'num_dimensions', ...
    %     {'max_verif_iter'},structName);
end

end


% Auxiliary functions -----------------------------------------------------

% see also inputArgsCheck, TODO integrate

function aux_checkFieldStr(optionsnn, field, admissibleValues, structName)
    % Check if field has an admissible value.
    if CHECKS_ENABLED
        fieldValue = optionsnn.(field);
        if ~(isa(fieldValue, 'string') || isa(fieldValue, 'char')) || ...
            ~ismember(fieldValue, admissibleValues)
            throw(CORAerror('CORA:wrongFieldValue', ...
                aux_getName(structName, field), admissibleValues))
        end
    end
end

function aux_checkFieldClass(optionsnn, field, admissibleClasses, structName)
    % Check if a field has the correct class.
    if CHECKS_ENABLED
        if ~ismember(class(optionsnn.(field)), admissibleClasses)
            throw(CORAerror('CORA:wrongFieldValue', ...
                aux_getName(structName, field), admissibleClasses))
        end
    end
end

function msg = aux_getName(structName, field)
    msg = sprintf("%s.nn.%s", structName, field);
end

function gpu_available = aux_isGPUavailable()
    try
        if ~isempty(which('gpuDeviceCount'))
            gpu_available = gpuDeviceCount('available') > 0;
        else
            gpu_available = false;
        end
    catch
        gpu_available = false;
    end
end

% ploy method has a different default for training
function poly_method = aux_defaultPolyMethod(options)
    if isfield(options,'train')
        poly_method = 'bounds';
    else
        poly_method = 'regression';
    end
end

% ------------------------------ END OF CODE ------------------------------
