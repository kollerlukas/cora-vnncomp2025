% Specify benchmark name.
dirs = regexp(pwd,'([^/]+)','match');
benchName = dirs{end};
% Obtain all instances.
filename = 'instances.csv';
instances = readtable(filename,'Delimiter',',');
% Rename columns
instances.Properties.VariableNames = {'model','vnnlib','timeout'};

verbose = true;

benchmarkNames = {};
models = {};
vnnlibs = {};
prepTimes = {};
results = {};
verifTimes = {};

% Create results folder.
resultsPath = './cora-results'; 
mkdir(resultsPath);

for i=1:size(instances,1) 
    % Extract current instance.
    instance = instances(i,:);
    modelPath = instance.model{1};
    vnnlibPath = instance.vnnlib{1};
    timeout = instance.timeout;

    % Create instance filename.
    modelName = regexp(modelPath,'([^/]+)(?=\.onnx$)','match');

    if strcmp(benchName,'safenlp')
        if contains(modelPath,'medical')
            modelName{1} = ['medial_' modelName{1}];
        elseif contains(modelPath,'ruarobot')
            modelName{1} = ['ruarobot_' modelName{1}];
        end
    end

    vnnlibName = regexp(vnnlibPath,'([^/]+)(?=\.vnnlib$)','match');
    instanceFilename = sprintf('%s/%s_%s.counterexample',...
        resultsPath,modelName{1},vnnlibName{1});

    % Prepare the current instance.
    prepare_instance(benchName,modelPath,vnnlibPath);

    tic

    % Run the current instance.
    res = run_instance(benchName,modelPath,vnnlibPath,instanceFilename, ...
        timeout,verbose);

    instanceTime = toc;

    if strcmp(res,'unsat') || strcmp(res,'unknown')
        % There is no counterexample; delete the file.
        delete(instanceFilename);
    end

    % Store outputs.
    benchmarkNames = [benchmarkNames; benchName];
    models = [models; 'vnncomp2024_benchmarks/benchmarks/' benchName '/' modelPath];
    vnnlibs = [vnnlibs; 'vnncomp2024_benchmarks/benchmarks/' benchName '/' vnnlibPath];
    prepTimes = [prepTimes; 0];
    results = [results; res];
    verifTimes = [verifTimes; instanceTime];
end

% Generate results table.
resultsTable = table(benchmarkNames,models,vnnlibs,prepTimes,results, ...
    verifTimes);
% Write to file.
writetable(resultsTable,sprintf('%s/results.csv',resultsPath),...
    'WriteVariableNames',false);
