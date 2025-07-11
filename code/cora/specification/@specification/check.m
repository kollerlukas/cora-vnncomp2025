function [res,indSpec,indObj] = check(spec,S,varargin)
% check - checks if a set satisfies the specification
%
% Syntax:
%    [res,indSpec,indObj] = check(spec,S)
%    [res,indSpec,indObj] = check(spec,S,time)
%
% Inputs:
%    spec - specification object
%    S - numeric, contSet, reachSet, or simResult object
%    time - (optional) interval (class: interval) for the set inputs,
%           numeric for points (scalar or match number of points)
%
% Outputs:
%    res - true/false whether set satisfies the specification
%    indSpec - index of the first specification that is violated
%    indObj - index of the first object S that is violated
%
% Other m-files required: none
% Subfunctions: none
% MAT-files required: none
%
% See also: specification

% Authors:       Niklas Kochdumper, Tobias Ladner
% Written:       29-May-2020             
% Last update:   22-March-2022 (TL, simResult, indObj)
%                24-May-2024 (TL, vectorized check for numeric input)
%                28-April-2025 (TL, reachSet/simResult and timed specifications)
% Last revision: ---

% ------------------------------ BEGIN CODE -------------------------------

% parse input arguments
narginchk(2,3);
time = setDefaultValues({[]}, varargin);

% init outputs
res = true; indSpec = []; indObj = [];

% start checking ----------------------------------------------------------

% split into temporal logic and other specifications
[spec,specLogic] = splitLogic(spec);

% check temporal logic ----------------------------------------------------

if ~isempty(specLogic)
    for i = 1:size(specLogic,1)
        res = aux_checkLogic(specLogic(i,1).set,S,time);
        if ~res
            indSpec = 1; return;
        end
    end
    if ~isempty(spec)
        res = check(spec,S,time);
    end
end

% check other specification -----------------------------------------------

% check S object
if isnumeric(S) % ---------------------------------------------------------
    
    % check numeric input

    % loop over all specifications
    for i = 1:size(spec,1)
        % init
        spec_i = spec(i);
        spec_i_time = spec_i.time;
        numPoints = size(S,2);

        if ~isempty(time) && ~isscalar(time) && numel(time) ~= numPoints
            throw(CORAerror('CORA:specialError','Given time has to be empty, scalar, or match number of points to check.'))
        end

        % find indices where given time overlaps with (timed) specification
        tol = 1e-9;
        if representsa_(spec_i_time,'emptySet',tol)
            % check all points
            idxTimed = true;
        else
            if representsa_(time,'emptySet',tol) 
                % time has to be given for timed specifications
                throw(CORAerror('CORA:specialError',...
                    'Timed specifications require a time interval.')); 
            else
                % check which points need testing
                idxTimed = contains(spec_i_time,time);
            end
        end

        if isscalar(idxTimed)
            % extend to match number of points
            idxTimed = true(1,numPoints) & idxTimed;
        end

        % different types of specifications
        resvec = true(size(idxTimed));
        switch spec_i.type

            case 'invariant'
                resvec(idxTimed) = contains(spec_i.set,S(:,idxTimed));

            case 'unsafeSet'
                resvec(idxTimed) = ~contains(spec_i.set,S(:,idxTimed));

            case 'safeSet'
                resvec(idxTimed) = contains(spec_i.set,S(:,idxTimed));

            case 'custom'
                resvec(idxTimed) = aux_checkCustom(spec_i.set,S(:,idxTimed));
        end

        % gather results
        res = all(resvec);

        % return if one point violates the specification
        if ~res
            indSpec = i; 
            indObj = find(res,1,'first');
            return;
        end
    end
    
elseif isa(S,'simResult') % -----------------------------------------------


    % loop over all specifications
    for i = 1:size(spec,1)
        spec_i = spec(i);

        % check if any simResult is present for timed specification
        if ~representsa_(spec_i.time,'emptySet',1e-8)
            % find simulation corresponding to the time
            S_timed = find(S,'time',spec_i.time);
            if isemptyobject(S_timed)
                % no simulation found, return false 
                % as behavior is undefined for that time
                indSpec = i;
                res = false;
                return
            end
        end

        % loop over all simulations
        for j = 1:length(S)
            S_j = S(j);
            % loop over all trajectories
            for k = 1:length(S_j.x)
                Si_x_k = S_j.x{k};
                Si_t_k = S_j.t{k};
    
                % check simulation points with respective time
                [res,~,l] = check(spec_i, Si_x_k', Si_t_k');
                if ~res
                    indSpec = i;
                    indObj = {j,k,l};
                    return; 
                end
            end
        end
    end
    
elseif isa(S,'reachSet') % ------------------------------------------------

    % loop over all specifications
    for i = 1:size(spec,1)
        spec_i = spec(i);

        % check if any reachSet is present for timed specification
        if ~representsa_(spec_i.time,'emptySet',1e-8)
            % find reachable set corresponding to the time
            S_timed = find(S,'time',spec_i.time);
            if isemptyobject(S_timed)
                % no reachable set found, return false 
                % as behavior is undefined for that time
                indSpec = i;
                res = false;
                return
            end
        end

        % loop over all reachable sets
        for k = 1:size(S,1)

            timePoint = S(k).timePoint;
            timeInterval = S(k).timeInterval;

            for j = 1:length(timeInterval.set)
                % where is the specification active?

                if representsa_(spec_i.time,'emptySet',eps) % entire time horizon
                    res = check(spec_i, timeInterval.set{j});

                elseif rad(spec_i.time) > 0 % part of time horizon
                    res = check(spec_i, ...
                        timeInterval.set{j}, ...
                        timeInterval.time{j});

                else % only active in one time point
                    % check if its on boundary of timeInterval
                    if contains(spec_i.time, timePoint.time{j})
                        % only start of time interval
                        res = check(spec_i, ...
                            timePoint.set{j}, ...
                            interval(timePoint.time{j}));

                    elseif contains(spec_i.time, timePoint.time{j+1})
                        % only end of time interval
                        res = check(spec_i, ...
                            timePoint.set{j+1}, ...
                            interval(timePoint.time{j+1}));

                    else % intermediate
                        res = check(spec_i, ...
                            timeInterval.set{j}, ...
                            timeInterval.time{j});
                    end
                end

                if ~res
                    indSpec = i; 
                    indObj = {k,j};
                    return; 
                end
            end
        end
    end

else % contSet ------------------------------------------------------------

    % loop over all specifications
    for i = 1:size(spec,1)
        spec_i = spec(i);

        % check if time frames overlap
        if representsa_(time,'emptySet',eps) && ~isemptyobject(spec_i.time)
            throw(CORAerror('CORA:specialError',...
                'Timed specifications require a time interval.')); 
        end

        if isemptyobject(spec_i.time) || isIntersecting_(spec_i.time,time,'exact',1e-8)

            % different types of specifications
            switch spec_i.type

                case 'invariant'
                    res = aux_checkInvariant(spec_i.set,S);

                case 'unsafeSet'
                    res = aux_checkUnsafeSet(spec_i.set,S);

                case 'safeSet'
                    res = aux_checkSafeSet(spec_i.set,S);

                case 'custom'
                    res = aux_checkCustom(spec_i.set,S);
            end

            % return as soon as one specification is violated
            if ~res
                indSpec = i; 
                indObj = 1;
                return;
            end
        end
    end
end

end


% Auxiliary functions -----------------------------------------------------

function res = aux_checkUnsafeSet(set,S)
% check if reachable set intersects the unsafe sets

    % check if cell array is given
    wasCell = iscell(S);
    if ~wasCell
        % convert for easier usage
        S = {S};
    end

    % S is cell, check each
    res = true;
    for i = 1:length(S)
        try
            res = ~isIntersecting_(set,S{i},'exact',1e-8);
        catch
            res = ~isIntersecting_(set,S{i},'approx',1e-8); 
        end
        % early exit
        if ~res
           return; 
        end
    end   
end

function res = aux_checkSafeSet(set,S)
% check if reachable set is inside the safe set

    if iscell(S)
        res = true;
        for i = 1:length(S)
           res = contains(set,S{i},'approx'); 
           if ~res
              return; 
           end
        end   
    else
        res = contains(set,S,'approx');
    end
end

function res = aux_checkCustom(func,S)
% check if the reachable set satisfies a user provided specification

    if iscell(S)
        res = false;
        for i = 1:length(S)
            res = func(S{i});
            if res
               return; 
            end
        end
    else
        res = func(S);
    end
end

function res = aux_checkInvariant(set,S)
% check if reachable set intersects the invariant

    if iscell(S)
        res = false;
        for i = 1:length(S)
            res = isIntersecting_(set,S{i},'approx',1e-8);
            if res
               return; 
            end
        end
    else
        res = isIntersecting_(set,S,'approx',1e-8);
    end
end

function res = aux_checkLogic(set,S,time)
% check if the reachable set satisfies a temporal logic specification

    if isnumeric(S)
        res = modelCheckTrace(set,S,time);
    else
        res = modelChecking(S,set);
    end
end

% ------------------------------ END OF CODE ------------------------------
