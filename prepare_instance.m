function res = prepare_instance(benchName,modelPath,vnnlibPath)
  % Initialize error.
  e = [];

  fprintf('prepare_instance(%s,%s,%s)...\n',benchName,modelPath,vnnlibPath);
  try
      fprintf('--- Loading network...');
      % Load neural network.
      [nn,options,permuteDims] = aux_readNetworkAndOptions( ...
          benchName,modelPath,vnnlibPath,false);

      fprintf(' done\n');

      fprintf('--- GPU available: %s\n',string(options.nn.train.use_gpu));
   
      fprintf('--- Loading specification...');
      % Load specification.
      [X0,specs] = vnnlib2cora(vnnlibPath);
      fprintf(' done\n');

      fprintf('--- Storing MATLAB file...');
      % Create filename.
      instanceFilename = getInstanceFilename(benchName,modelPath,vnnlibPath);
      % Store network, options, and specification.
      save(instanceFilename,'nn','options','permuteDims','X0','specs');
      fprintf(' done\n');

      % Print the options.
      aux_printOptions(options);
  catch e
      % Print the error message. 
      printErrorMessage(e)
      return;
  end
  res = isempty(e);

  try
    % Clear variables.
    clearvars -except res
    % Reset GPU.
    reset(gpuDevice);
    % De-select GPU.
    gpuDevice([]);
  catch e
    % Print the error message. 
    printErrorMessage(e)
    return;
  end
end

% Auxiliary functions -----------------------------------------------------

function [nn,options,permuteDims] = aux_readNetworkAndOptions( ...
  benchName,modelPath,vnnlibPath,verbose)

  % Create evaluation options.
  options.nn = struct(...
      'use_approx_error',true,...
      'poly_method','bounds',... {'bounds','singh','center'}
      'train',struct(...
          'backprop',false,...
          'mini_batch_size',2^10 ...
      ) ...
  );
  % Set default training parameters
  options = nnHelper.validateNNoptions(options,true);
  % Disable the interval-center by default.
  options.nn.interval_center = false;
  % Use the moving statistics for the batch normalization.
  options.nn.batch_norm_moving_stats = true;

  % Specify falsification method: {'center','fgsm','zonotack'}.
  options.nn.falsification_method = 'zonotack';
  % Specify input set refinement method: {'naive','zonotack','zonotack-layerwise'}.
  options.nn.refinement_method = 'zonotack';
  % Set number of input generators.
  options.nn.train.num_init_gens = inf;
  % Set number of approximation error generators per layer.
  options.nn.approx_error_order = 'sensitivity*length';
  % Compute the exact bounds of the constraint zonotope.
  options.nn.exact_conzonotope_bounds = false;
  % Specify number of splits, dimensions, and neuron-splits.
  options.nn.num_splits = 2; 
  options.nn.num_dimensions = 1;
  options.nn.num_neuron_splits = 0;
  % Add relu tightening constraints.
  options.nn.num_relu_constraints = 0;
  options.nn.add_orth_neuron_splits = true;
  % Specify the number of iterations.
  options.nn.polytope_bound_approx_max_iter = 8;
  options.nn.refinement_max_iter = 8;

  % Default: do not permute the input dimensions. 
  permuteDims = false;

  % Obtain the model name.
  [~,modelName,~] = getInstanceFilename(benchName,modelPath,vnnlibPath);

  % VNN-COMP'24 Benchmarks ------------------------------------------------
  if strcmp(benchName,'test') ...
    || strcmp(modelName{1},'test_nano') % is called after each benchmark
      nn = neuralNetwork.readONNXNetwork(modelPath,verbose,'','', ...
          'dlnetwork',false);
      % Use the default parameters.
  elseif strcmp(benchName,'acasxu_2023') || strcmp(benchName,'acas_xu')
      % acasxu ----------------------------------------------------------
      nn = neuralNetwork.readONNXNetwork(modelPath,verbose,'BSSC');
      % Specify an initial split (num pieces, num dimensions).
      % options.nn.init_split = [10 5];
      % Specify number of splits, dimensions, and neuron-splits.
      options.nn.num_splits = 2; 
      options.nn.num_dimensions = 1;
      options.nn.num_neuron_splits = 0;
      % Add relu tightening constraints.
      options.nn.num_relu_constraints = 0;

      % options.nn.interval_center = true;
      % options.nn.train.num_init_gens = 5;
      % options.nn.train.num_approx_err = 0;
      % options.nn.train.mini_batch_size = 2^5;

      % options.nn.max_verif_iter = 100;
      % options.nn.verify_dequeue_type = 'front';
      % options.nn.verify_enqueue_type = 'append';
  elseif strcmp(benchName,'cersyve')
      nn = neuralNetwork.readONNXNetwork(modelPath,verbose,'BC', ...
          '','dagnetwork',true);
      % Specify number of splits, dimensions, and neuron-splits.
      options.nn.num_splits = 2; 
      options.nn.num_dimensions = 1;
      options.nn.num_neuron_splits = 0;
  elseif strcmp(benchName,'cifar100_2024')
      nn = neuralNetwork.readONNXNetwork(modelPath,verbose,'BCSS', ...
          '','dagnetwork',true);
      % Bring input into the correct shape.
      permuteDims = true;
      % Requires less memory.
      % options.nn.falsification_method = 'fgsm';
      % Use interval-center.
      options.nn.interval_center = true;
      options.nn.train.num_init_gens = 500;
      options.nn.train.num_approx_err = 100;
      % Add relu tightening constraints.
      % options.nn.num_relu_constraints = 100;
      % Specify number of splits, dimensions, and neuron-splits.
      % options.nn.num_splits = 3; 
      % options.nn.num_dimensions = 1;
      % options.nn.num_neuron_splits = 3;
      % Save memory (reduce batch size & do not batch union constraints).
      % options.nn.train.mini_batch_size = 2^2;
      options.nn.batch_union_conzonotope_bounds = false;
  elseif strcmp(benchName,'collins_rul_cnn_2022')
      nn = neuralNetwork.readONNXNetwork(modelPath,verbose,'BCSS');
      % Bring input into the correct shape.
      permuteDims = true;
      % Use interval-center.
      options.nn.interval_center = true;
      options.nn.train.num_init_gens = inf;
      options.nn.train.num_approx_err = 100;
  elseif strcmp(benchName,'cora_2024')
      nn = neuralNetwork.readONNXNetwork(modelPath,verbose,'BC');
      % Requires less memory.
      % options.nn.falsification_method = 'fgsm';
      % Specify an initial split (num pieces, num dimensions).
      % options.nn.init_split = [2 10];
      % Use the default parameters.
      % options.nn.interval_center = true;
      % options.nn.train.num_init_gens = 500; % inf;
      % options.nn.train.num_approx_err = 100;
      % Add relu tightening constraints.
      % options.nn.num_relu_constraints = inf;
      % Reduce batch size.
      % options.nn.train.mini_batch_size = 2^5;
      % Specify number of splits, dimensions, and neuron-splits.
      options.nn.num_splits = 2; 
      options.nn.num_dimensions = 1;
      options.nn.num_neuron_splits = 0;
      % Save memory (do not batch union constraints).
      options.nn.batch_union_conzonotope_bounds = false;
  elseif strcmp(benchName,'dist_shift_2023')
      % dist_shift ------------------------------------------------------
      nn = neuralNetwork.readONNXNetwork(modelPath,verbose,'BC');
      % Use default values.
  elseif strcmp(benchName,'linearizenn_2024')
      % LinearizeNN -----------------------------------------------------
      nn = neuralNetwork.readONNXNetwork(modelPath,verbose,'BC', ...
          '','dagnetwork',true);
      % Specify number of splits, dimensions, and neuron-splits.
      options.nn.num_splits = 2; 
      options.nn.num_dimensions = 1;
      options.nn.num_neuron_splits = 1;
  elseif strcmp(benchName,'malbeware')
      nn = neuralNetwork.readONNXNetwork(modelPath,verbose,'BCSS');
      % Bring input into the correct shape.
      permuteDims = true;
  elseif strcmp(benchName,'metaroom_2023')
      % metaroom --------------------------------------------------------
      nn = neuralNetwork.readONNXNetwork(modelPath,verbose,'BCSS');
      % Bring input into the correct shape.
      permuteDims = true;
      % Use interval-center.
      options.nn.interval_center = true;
      options.nn.train.num_init_gens = 500;
      options.nn.train.num_approx_err = 100;
      % Reduce the batch size.
      options.nn.train.mini_batch_size = 2^5;
      % Add relu tightening constraints.
      % options.nn.num_relu_constraints = 100;
  elseif strcmp(benchName,'nn4sys')
      % nn4sys ----------------------------------------------------------
      if strcmp(modelName{1},'lindex') || strcmp(modelName{1},'lindex_deep')
        nn = neuralNetwork.readONNXNetwork(modelPath,verbose,'BC','BC');
      else
        % nn = neuralNetwork.readONNXNetwork(modelPath,verbose,'BCT','BC', ...
        %     'dagnetwork',true);
        throw(CORAerror('CORA:notSupported',...
          sprintf("Networks not supported '%s'!",modelName{1})));
      end
  elseif strcmp(benchName,'relusplitter')
      nn = neuralNetwork.readONNXNetwork(modelPath,verbose,'CSS');
  elseif strcmp(benchName,'safenlp_2024')
      % safeNLP ---------------------------------------------------------
      nn = neuralNetwork.readONNXNetwork(modelPath,verbose,'BC');
  elseif strcmp(benchName,'sat_relu')
      nn = neuralNetwork.readONNXNetwork(modelPath,verbose,'BC');
  elseif strcmp(benchName,'soundnessbench')
      nn = neuralNetwork.readONNXNetwork(modelPath,verbose,'BC');
      % Use interval-center.
      options.nn.interval_center = true;
      options.nn.train.num_init_gens = inf;
      options.nn.train.num_approx_err = 50;
      % Reduce the batch size.
      options.nn.train.mini_batch_size = 2^5;
      % Specify number of splits, dimensions, and neuron-splits.
      options.nn.num_splits = 2; 
      options.nn.num_dimensions = 1;
      options.nn.num_neuron_splits = 1;
  elseif strcmp(benchName,'tinyimagenet_2024')
      % tinyimagenet ----------------------------------------------------
      nn = neuralNetwork.readONNXNetwork(modelPath,verbose,'BCSS', ...
          '','dagnetwork',true);
      % Bring input into the correct shape.
      permuteDims = true;
      % Requires less memory.
      % options.nn.falsification_method = 'fgsm';
      % Use interval-center.
      options.nn.interval_center = true;
      options.nn.train.num_init_gens = 500;
      options.nn.train.num_approx_err = 10;
      % Add relu tightening constraints.
      % options.nn.num_relu_constraints = 10;
      % Specify number of splits, dimensions, and neuron-splits.
      % options.nn.num_splits = 2; 
      % options.nn.num_dimensions = 1;
      % options.nn.num_neuron_splits = 1;
      % Save memory (reduce batch size & do not batch union constraints).
      options.nn.train.mini_batch_size = 2^2;
      options.nn.batch_union_conzonotope_bounds = false;
  elseif strcmp(benchName,'tllverifybench_2023')
      % tllverifybench --------------------------------------------------
      nn = neuralNetwork.readONNXNetwork(modelPath,verbose,'BC');
      % Use default settings.
  else
      throw(CORAerror('CORA:notSupported',...
          sprintf("Unknown benchmark '%s'!",benchName)));
  end

end

function aux_printOptions(options)
    % Print parameters.
    table = CORAtableParameters('neuralNetwork/verify options');
    table.printHeader();
    % Zonotope propagation options.
    table.printContentRow('GPU',string(options.nn.train.use_gpu));
    table.printContentRow('Poly. Method',options.nn.poly_method);
    table.printContentRow('Batchsize', ...
        string(options.nn.train.mini_batch_size));
    table.printContentRow('Interval Center', ...
        string(options.nn.interval_center));
    table.printContentRow('Num. init. Generators', ...
        string(options.nn.train.num_init_gens));
    table.printContentRow('Num. approx. Error (per nonl. Layer)', ...
        string(options.nn.train.num_approx_err));
    table.printContentRow('approx. Error Heuristic', ...
        options.nn.approx_error_order);
    % Main algorithm options.
    table.printContentRow('Falsification Method', ...
        options.nn.falsification_method);
    table.printContentRow('Refinement Method', ...
        options.nn.refinement_method);
    % Details algorithm hyperparameters.
    table.printContentRow('Num. of Splits', ...
        string(options.nn.num_splits));
    table.printContentRow('Num. of Dimensions', ...
        string(options.nn.num_dimensions));
    table.printContentRow('Num. of Neuron-Splits', ...
        string(options.nn.num_neuron_splits));
    table.printContentRow('Num. of ReLU-Tightening Constraints', ...
        string(options.nn.num_relu_constraints));
    % Finish table.
    table.printFooter();
end