function CORAwarning(identifier, varargin)
% CORAwarning - prints a CORA warning into the command line.
%    Can be controlled via the CORA_WARNINGS_ENABLED macro
%
% Syntax:
%    CORAwarning(identifier, varargin)
%
% Inputs:
%    identifier - char
%    varargin - depends on identifier, see below
%
% Outputs:
%    -
%
% Other m-files required: none
% Subfunctions: none
% MAT-files required: none
%
% See also: CORA_WARNINGS_ENABLED, CORAerror

% Authors:       Tobias Ladner
% Written:       17-April-2024
% Last update:   07-October-2024 (MW, add CORA:interface)
% Last revision: ---

% ------------------------------ BEGIN CODE -------------------------------

% parse input
narginchk(2,Inf);
aux_checkIdentifier(identifier)

% check if CORA warning should be shown (global/specific)
if ~CORA_WARNINGS_ENABLED || ~CORA_WARNINGS_ENABLED(identifier)
    % do not show warning
    return
end

% build specific warning text
switch identifier
    case 'CORA:deprecated'
        % identifier, type, name, version, replacement, reason

        % read input
        narginchk(6,6);
        type = varargin{1};
        name = varargin{2};
        version = varargin{3};
        replacement = varargin{4};
        reason = varargin{5};

        % compose replacement and reason
        replacement = aux_formatString(replacement);
        reason = aux_formatString(reason);
        
        desc = sprintf('The %s ''%s'' is deprecated (since %s) and will be removed in a future release.\n  %s\n  %s', ...
            type, name, version, replacement, reason);

    case 'CORA:interface'
        % read input
        narginchk(3,3);
        name = varargin{1};
        version = varargin{2};
        
        desc = sprintf('The interface of the function ''%s'' has changed (since %s):\n  Please check the function header for details', ...
            name, version);

    otherwise
        % identifier, msg, args for sprintf
        desc = varargin{1};
        desc = aux_formatString(desc);
        desc = sprintf(desc,varargin{2:end});
end

% show warning (the \b's erase the default 'Warning:' text)
warning('\b\b\b\b\b\b\b\b\b<strong>CORA warning:</strong> %s', desc);

end


% Auxiliary functions -----------------------------------------------------

function aux_checkIdentifier(identifier)
    % test identifier 
    % (please also add new identifiers to test_cora_warnings_enabled)
    inputArgsCheck({{identifier,'str',{ ...
        'CORA:app', ...
        'CORA:contDynamics', ...
        'CORA:contSet', ...
        'CORA:converter', ...
        'CORA:discDynamics', ...
        'CORA:examples', ...
        'CORA:global', ...
        'CORA:hybridDynamics', ...
        'CORA:manual', ...
        'CORA:matrixSets', ...
        'CORA:models', ...
        'CORA:nn', ...
        'CORA:specification', ...
        'CORA:unitTests', ...
         ... % special warnings
        'CORA:plot', ...
        'CORA:solver', ...
        'CORA:deprecated', ...
        'CORA:redundant', ...
        'CORA:interface', ...
    }}})
end

function str = aux_formatString(str)
    % format line breaks
    str = replace(str,'\n','\n  ');
    str = compose(str);
    str = str{1};
end

% ------------------------------ END OF CODE ------------------------------
