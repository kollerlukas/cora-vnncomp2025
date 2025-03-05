% Run all benchmarks in the current directory.

% Restrict number of CPU threads.
maxNumCompThreads(4);

% Specify base directory.
basePath = './vnncomp2024_benchmarks/benchmarks';

% Get all directories.
benchmarkDirs = aux_findAllBenchmarks(basePath);

for i=1:length(benchmarkDirs)
    % Change directory to the current benchmark.
    cd(benchmarkDirs{i});
    % Run all instances of the benchmark.
    run_instances
    % Go back to main directory.
    cd ../;
end

% Auxiliary Functions. ----------------------------------------------------

function benchmarkDirs = aux_findAllBenchmarks(basePath)
    % Store names of the configurations.
    benchmarkDirs = {};
    % Find all directories.
    files = dir(basePath);
    for i=1:length(files)
        filename = files(i).name;
        if ~strcmp(filename,'.') && ~strcmp(filename,'..')
            benchmarkDirs{end+1} = sprintf('%s/%s',basePath,filename);
        end
    end
end