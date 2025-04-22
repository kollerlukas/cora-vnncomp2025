function res = prepare_instance(benchName,modelPath,vnnlibPath)
  fprintf('prepare_instance(%s,%s,%s)...\n',benchName,modelPath,vnnlibPath);
  try
      fprintf('--- Loading network...');
      % Load neural network.
      [nn,options,permuteDims] = aux_readNetworkAndOptions( ...
          benchName,modelPath,vnnlibPath,false);

      fprintf(' done\n');

      fprintf('--- GPU available: %d\n',options.nn.train.use_gpu);
   
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
      fprintf(newline);
      fprintf(e.message);
      fprintf(newline);
      % Some error
      res = 1;
      return;
  end
  res = 0;
end

% Auxiliary functions -----------------------------------------------------

function [nn,options,permuteDims] = aux_readNetworkAndOptions( ...
  benchName,modelPath,vnnlibPath,verbose)

  % Create evaluation options.
  options.nn = struct(...
      'use_approx_error',true,...
      'poly_method','bounds',...'bounds','singh'
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
  options.nn.refinement_method = 'zonotack-layerwise';
  % Set number of input generators.
  options.nn.train.num_init_gens = inf;
  % Set number of approximation error generators per layer.
  options.nn.train.num_approx_err = inf;
  options.nn.approx_error_order = 'sensitivity*length';
  % Compute the exact bounds of the constraint zonotope.
  options.nn.exact_conzonotope_bounds = false;
  % Specify number of splits, dimensions, and neuron-splits.
  options.nn.num_splits = 2; 
  options.nn.num_dimensions = 1;
  options.nn.num_neuron_splits = 2;

  % Default: do not permute the input dimensions. 
  permuteDims = false;

  % Obtain the model name.
  [~,modelName,~] = getInstanceFilename(benchName,modelPath,vnnlibPath);

  if strcmp(benchName,'test') ...
    || strcmp(modelName{1},'test_nano') % is called after each benchmark
      nn = neuralNetwork.readONNXNetwork(modelPath,verbose,'','', ...
          'dlnetwork',false);
      % Use the default parameters.
  elseif strcmp(benchName,'acasxu_2023')
      % acasxu ----------------------------------------------------------
      nn = neuralNetwork.readONNXNetwork(modelPath,verbose,'BSSC');
      % Specify an initial split (num pieces, num dimensions).
      % options.nn.init_split = [10 5];
      % Specify number of splits, dimensions, and neuron-splits.
      options.nn.num_splits = 5; 
      options.nn.num_dimensions = 1;
      options.nn.num_neuron_splits = 0;
      % Add relu tightening constraints.
      options.nn.num_relu_tighten_constraints = 0; % inf;
      % Increase batch size.
      % options.nn.train.mini_batch_size = 2^14;
  elseif strcmp(benchName,'cctsdb_yolo_2023')
      throw(CORAerror('CORA:notSupported',...
          sprintf("Benchmark '%s' not supported!",benchName)));
  elseif strcmp(benchName,'cgan_2023')
      % c_gan -----------------------------------------------------------
      % nn = neuralNetwork.readONNXNetwork(modelPath,verbose,'BC');
      % --- TODO: implement convTranspose
      throw(CORAerror('CORA:notSupported',...
          sprintf("Benchmark '%s' not supported!",benchName)));
  elseif strcmp(benchName,'cifar100')
      % vnncomp2024_cifar100_benchmark ----------------------------------
      nn = neuralNetwork.readONNXNetwork(modelPath,verbose,'BCSS', ...
          '','dagnetwork',true);
      % Bring input into the correct shape.
      permuteDims = true;
      % Requires less memory.
      options.nn.falsification_method = 'fgsm';
      % Use interval-center.
      options.nn.interval_center = true;
      options.nn.train.num_init_gens = 500;
      options.nn.train.num_approx_err = 10;
      % Add relu tightening constraints.
      options.nn.num_relu_tighten_constraints = 10;
      % Specify number of splits, dimensions, and neuron-splits.
      options.nn.num_splits = 5; 
      options.nn.num_dimensions = 1;
      options.nn.num_neuron_splits = 0;
      % Save memory (reduce batch size & do not batch union constraints).
      options.nn.train.mini_batch_size = 2;
      options.nn.batch_union_conzonotope_bounds = false;
  elseif strcmp(benchName,'collins_aerospace_benchmark')
      throw(CORAerror('CORA:notSupported',...
          sprintf("Benchmark '%s' not supported!",benchName)));
  elseif strcmp(benchName,'collins_rul_cnn_2023')
      % collins_rul_cnn -------------------------------------------------
      nn = neuralNetwork.readONNXNetwork(modelPath,verbose,'BCSS');
      % Bring input into the correct shape.
      permuteDims = true;
      % Use interval-center.
      options.nn.interval_center = true;
      options.nn.train.num_init_gens = inf;
      options.nn.train.num_approx_err = 100;
  elseif strcmp(benchName,'collins_yolo_robustness_2023')
      throw(CORAerror('CORA:notSupported',...
          sprintf("Benchmark '%s' not supported!",benchName)));
  elseif strcmp(benchName,'cora')
      nn = neuralNetwork.readONNXNetwork(modelPath,verbose,'BC');
      % Use the default parameters.
      options.nn.interval_center = false;
      options.nn.train.num_init_gens = inf;
      options.nn.train.num_approx_err = inf;
      % Add relu tightening constraints.
      % options.nn.num_relu_tighten_constraints = 100;
      % Reduce batch size.
      options.nn.train.mini_batch_size = 2^6;
      % Specify number of splits, dimensions, and neuron-splits.
      options.nn.num_splits = 2; 
      options.nn.num_dimensions = 1;
      options.nn.num_neuron_splits = 0;
      % Use fgsm falsification.
      options.nn.falsification_method = 'fgsm';
      % Save memory (do not batch union constraints).
      options.nn.batch_union_conzonotope_bounds = false;
  elseif strcmp(benchName,'dist_shift_2023')
      % dist_shift ------------------------------------------------------
      nn = neuralNetwork.readONNXNetwork(modelPath,verbose,'BC');
      % % Simpler methods suffice.
      % options.nn.falsification_method = 'fgsm';
      % options.nn.refinement_method = 'naive';
      % Specify number of splits, dimensions, and neuron-splits.
      options.nn.num_splits = 5; 
      options.nn.num_dimensions = 1;
      options.nn.num_neuron_splits = 0;
      % Add relu tightening constraints.
      options.nn.num_relu_tighten_constraints = 100;
  elseif strcmp(benchName,'linearizenn')
      % LinearizeNN -----------------------------------------------------
      % nn = neuralNetwork.readONNXNetwork(modelPath,verbose,'BC','BC');
      % --- TODO: weird networks (MatMul) and concat => No
      throw(CORAerror('CORA:notSupported',...
          sprintf("Benchmark '%s' not supported!",benchName)));
  elseif strcmp(benchName,'lsnc')
      throw(CORAerror('CORA:notSupported',...
          sprintf("Benchmark '%s' not supported!",benchName)));
  elseif strcmp(benchName,'metaroom_2023')
      % metaroom --------------------------------------------------------
      nn = neuralNetwork.readONNXNetwork(modelPath,verbose,'BCSS');
      % Bring input into the correct shape.
      permuteDims = true;
      % Use interval-center.
      options.nn.interval_center = true;
      options.nn.train.num_init_gens = 500;
      options.nn.train.num_approx_err = 100;
  elseif strcmp(benchName,'ml4acopf_2023')
      throw(CORAerror('CORA:notSupported',...
          sprintf("Benchmark '%s' not supported!",benchName)));
  elseif strcmp(benchName,'ml4acopf_2024')
      throw(CORAerror('CORA:notSupported',...
          sprintf("Benchmark '%s' not supported!",benchName)));
  elseif strcmp(benchName,'nn4sys_2023')
      % nn4sys ----------------------------------------------------------
      if ~strcmp(modelName{1},'lindex') && ...
              ~strcmp(modelName{1},'lindex_deep')
          % Skip this instance.
          throw(CORAerror('CORA:notSupported',...
              sprintf("Model '%s' of benchmark '%s' is not " + ...
              "supported!",modelPath,benchName)));
      end
      nn = neuralNetwork.readONNXNetwork(modelPath,verbose,'BC','BC');
      % Use the default parameters.
  elseif strcmp(benchName,'safenlp')
      % safeNLP ---------------------------------------------------------
      nn = neuralNetwork.readONNXNetwork(modelPath,verbose,'BC');
      % Increase batch size.
      options.nn.train.mini_batch_size = 2^10;
      % % Specify number of splits, dimensions, and neuron-splits.
      % options.nn.num_splits = 2; 
      % options.nn.num_dimensions = 2;
      % options.nn.num_neuron_splits = 2;
      % Add relu tightening constraints.
      options.nn.num_relu_tighten_constraints = inf;
  elseif strcmp(benchName,'tinyimagenet')
      % vnncomp2024_cifar100_benchmark ----------------------------------
      nn = neuralNetwork.readONNXNetwork(modelPath,verbose,'BCSS', ...
          '','dagnetwork',true);
      % Bring input into the correct shape.
      permuteDims = true;
      % Use interval-center.
      options.nn.interval_center = true;
      options.nn.train.num_init_gens = 100;
      options.nn.train.num_approx_err = 0;
  elseif strcmp(benchName,'tllverifybench_2023')
      % tllverifybench --------------------------------------------------
      nn = neuralNetwork.readONNXNetwork(modelPath,verbose,'BC');
      % Simpler methods suffice.
      options.nn.falsification_method = 'fgsm';
      options.nn.refinement_method = 'naive';
  elseif strcmp(benchName,'traffic_signs_recognition_2023')
      throw(CORAerror('CORA:notSupported',...
          sprintf("Benchmark '%s' not supported!",benchName)));
  elseif strcmp(benchName,'vggnet16_2023')
      throw(CORAerror('CORA:notSupported',...
          sprintf("Benchmark '%s' not supported!",benchName)));
  elseif strcmp(benchName,'vit_2023')
      throw(CORAerror('CORA:notSupported',...
          sprintf("Benchmark '%s' not supported!",benchName)));
  elseif strcmp(benchName,'yolo_2023')
      throw(CORAerror('CORA:notSupported',...
          sprintf("Benchmark '%s' not supported!",benchName)));
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
    table.printContentRow('Poly. Method',options.nn.poly_method);
    table.printContentRow('Batchsize', ...
        string(options.nn.train.mini_batch_size));
    table.printContentRow('Interval Center',options.nn.interval_center);
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
        string(options.nn.num_relu_tighten_constraints));
    % Finish table.
    table.printFooter();
end
