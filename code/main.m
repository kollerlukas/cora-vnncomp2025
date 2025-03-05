function completed = main(varargin)
% main - runs all scripts to repeat the evaluation section of the paper
%
% results and plots will be saved to ./results
%
% Syntax:
%    completed = main()
%    completed = main(evalname)
%
% Inputs:
%    evalname - str, results are stored in ./results/<evalname>
%                    defaults to current date and time
%
% Outputs:
%    completed - boolean

% ------------------------------ BEGIN CODE -------------------------------

% 1. SETTINGS (update as needed) ------------------------------------------

% general settings
PAPER_TITLE = 'CORA'; % !
VENUE_NAME = 'VNN-COMP 2025';   % !
aux_runStartup(PAPER_TITLE,VENUE_NAME);

% plotting settings
plotSettings = struct;
plotSettings.saveOpenFigures = true;
plotSettings.saveAsFig = true;
plotSettings.saveAsPng = true;
plotSettings.saveAsEps = false;

% also change 3. Scripts below

% 2. SETUP (nothing to change here) ---------------------------------------

% parse input
if nargin < 1
    evalname = datestr(datetime,'yymmdd-hhMMss');
else
    evalname = varargin{1};
end

% set up paths
basepath = '.';

% PATH                      VARIABLE        PURPOSE
% ./                        basepath        base path
% - ./code                  codepath        path to code
%   - ./cora                -               path to CORA
%   - ./scripts             -               path to auxiliary scripts
%   - ./main.m              -               main evaluation file (this file)
% - ./data                  datapath        path to data
% - ./results/<evalname>    resultspath     path to results
%   - ./evaluation          evalpath        path to store any evaluation results
%   - ./plots               plotspath       path to plots; any open figure after
%                                           each script will be stored there
%   - ./results.txt         -               logs all outputs to command window
%
[codepath,datapath,resultspath,evalpath,plotspath] = aux_setup(basepath,evalname);

% 3. RUN SCRIPTS (update as needed) ---------------------------------------

scripts = {; ...
    % list all scripts of evaluation here
    @() run_benchmarks(datapath,evalpath,resultspath);
};

% run scripts
aux_runScripts(scripts,plotspath,plotSettings)

% 4. WRAP UP (nothing to change here) -------------------------------------

aux_wrapup()
completed = true;

end

% Auxiliary functions -----------------------------------------------------

function aux_runStartup(PAPER_TITLE,VENUE_NAME)
    rng(1)
    warning off

    % show startup block
    disp(' ')
    aux_seperateLine()
    disp(' ')
    disp('Repeatability Package')
    fprintf("Paper: %s\n", PAPER_TITLE)
    fprintf("Venue: %s\n", VENUE_NAME)
    fprintf("Date: %s\n", datestr(datetime()))
    disp(' ')
    if ~isempty(which('CORAVERSION'))
        fprintf('CORA: %s\n', CORAVERSION)
    end
    fprintf('Matlab: %s\n', version)
    fprintf('System: %s\n', computer)
    fprintf('GPU available: %i', canUseGPU)
    disp(' ')
    aux_seperateLine()
    disp(' ')
    pause(2) % to make the startup block readable
end

function [codepath,datapath,resultspath,evalpath,plotspath] = aux_setup(basepath,evalname)
    % set up paths
    codepath = sprintf("%s/code", basepath);
    datapath = sprintf("%s/data", basepath);
    resultspath = sprintf("%s/results/%s", basepath, evalname);
    mkdir(resultspath)
    evalpath = sprintf("%s/evaluation", resultspath);
    mkdir(evalpath)
    plotspath = sprintf("%s/plots", resultspath);
    mkdir(plotspath)
    
    % for smooth images (only for eps)
    set(0, 'defaultFigureRenderer', 'painters')
    
    % set up diary
    resultstxt = sprintf("%s/results.txt", resultspath);
    delete(resultstxt)
    diary(resultstxt)
end

function aux_runScripts(scripts,plotspath,plotSettings)
    % run scripts

    % process input
    n = size(scripts, 1);
    fprintf("Running %d scripts.. \n", n);
    disp(" ")
    
    for i = 1:n
        % run script i
        aux_seperateLine()
        disp(' ')
        script = scripts{i, 1};
        name = scripts{i, 2};
    
        try
            % call script
            fprintf("Running '%s' ...\n", name)
            script();
            disp(" ")
            fprintf("'%s' was run successfully!\n", name)
    
            % save open figures
            if plotSettings.saveOpenFigures
                fprintf("Saving plots to '%s'..\n", plotspath)
                
                % find all open figures
                h = findobj('type', 'figure');
                m = length(h);
        
                % get unique name
                figure_name = sprintf('%s/%s', plotspath, name);

                % iterate through figures
                for j = 1:m
                    % save as desired extensions
                    if plotSettings.saveAsFig % .fig
                        savefig(sprintf("%s_%d.%s", figure_name, j, 'fig'));
                    end
                    if plotSettings.saveAsPng % .png
                        saveas(gcf, sprintf("%s_%d.%s", figure_name, j, 'png'));
                    end
                    if plotSettings.saveAsEps % .eps
                        saveas(gcf, sprintf("%s_%d.%s", figure_name, j, 'eps'), 'epsc');
                    end
                    % close figure
                    close(gcf)
                end
            end
    
        catch ME
            % error handling
            disp(" ")
            fprintf("An ERROR occured during execution of '%s':\n", name);
            disp(ME.getReport())
            disp("Continuing with next script..")
        end
    
        disp(" ")
    end
end

function aux_wrapup()
    % wrap up evaluation
    aux_seperateLine()
    disp(" ")
    disp("Completed!")
    fprintf("Date: %s\n", datestr(datetime()))
    diary off;
end

function aux_seperateLine()
    disp ------------------------------------------------------------------
end

% ------------------------------ END OF CODE ------------------------------

