function printErrorMessage(e)
    % Print the error message along with file and line.
    
    % Print the error message. 
    fprintf(newline);
    fprintf('Unexpected Error! --- %s\n',e.message);

    % Print the stack.
    for i=1:length(e.stack)
        % Get stack info from the first entry.
       [funcname,classname,linenr] = extractStackInfo(e.stack(i));
       % Print the error message. 
       fprintf(' --- %s/%s [%d]\n',classname,funcname,linenr);
    end 
    fprintf(newline);
    
end

% Auxiliary functions -----------------------------------------------------

function [funcname,classname,linenr] = extractStackInfo(stackEntry)
    % Get the function name.
    funcname = stackEntry.name;
    % Get the classname.
    [dir,filename,~] = fileparts(stackEntry.file);
    if contains(dir,'@')
        % The function is contained in a separate file.
        [~,classname_] = fileparts(dir);
        % Remove the '@'.
        classname_(1) = [];
        % Handle sub-functions.
        if ~strcmp(filename,funcname)
            % The error occurred in a sub-function.
            funcname = [filename '/' funcname];
        end
        % Set the classname to the name of the parent 
        % directory.
        classname = classname_;
    else
        % The class name is the filename.
        classname = filename;
    end
    % Get the line number.
    linenr = stackEntry.line;
end

