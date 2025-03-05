function run_benchmarks(dataPath,resultsPath)
% Run all benchmarks in the current directory.

% Restrict number of CPU threads.
maxNumCompThreads(4);

% Get the basepath.
basepath = pwd;

% Specify base directory.
benchmarksPath = sprintf('%s/vnncomp2024_benchmarks/benchmarks',dataPath);

% Get all directories.
benchmarkDirs = aux_findAllBenchmarks(benchmarksPath);

for i=1:length(benchmarkDirs)
    % Change directory to the current benchmark.
    cd(benchmarkDirs{i});
    % Get the benchmark name.
    [~,benchmarkName,~] = fileparts(benchmarkDirs{i});
    % Create a results directory.
    benchmarkResultsPath = sprintf('%s/cora/%s',resultsPath,benchmarkName); 
    mkdir(resultsPath);
    % Run all instances of the benchmark.
    run_instances(benchmarkName,benchmarkResultsPath);
    % Go back to main directory.
    cd(basepath);
end
end

% Auxiliary Functions. ----------------------------------------------------

function benchmarkDirs = aux_findAllBenchmarks(benchmarksPath)
    % Store names of the configurations.
    benchmarkDirs = {};
    % Find all directories.
    files = dir(benchmarksPath);
    for i=1:length(files)
        filename = files(i).name;
        if ~strcmp(filename,'.') && ~strcmp(filename,'..')
            benchmarkDirs{end+1} = sprintf('%s/%s',basePath,filename);
        end
    end
end
