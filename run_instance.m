function res = run_instance(benchName,modelPath,vnnlibPath,resultsPath, ...
    timeout,verbose)
    fprintf('run_instance(%s,%s,%s,%s,%d,%d)...\n',benchName,modelPath, ...
        vnnlibPath,resultsPath,timeout,verbose);
    try
        fprintf('--- Loading MATLAB file...');
        % Create filename.
        [instanceFilename,modelName,~] = ...
            getInstanceFilename(benchName,modelPath,vnnlibPath);
        % Load stored network and specification.
        load(instanceFilename,'nn','options','permuteInputDims', ...
            'X0','specs');
        fprintf(' done\n');
        
        fprintf('--- Deleting MATLAB file...');
        % Delete file with stored networks and specification.
        delete(instanceFilename);
        fprintf(' done\n');

        % Obtain the model name.
        if permuteInputDims
            % if strcmp(benchName,'collins_rul_cnn_2023') && ~strcmp(modelName,'NN_rul_full_window_40')
            %     inSize = nn.layers{1}.inputSize;
            % else
            %     inSize = nn.layers{1}.inputSize([2 1 3]);
            % end
            inSize = nn.layers{1}.inputSize([2 1 3]);
        end

        fprintf('--- Running verification...');
        if verbose
            fprintf('\n\n');
        end
        % There can be multiple input set. 
        % Concatenate the sets to a batch.
        x = [];
        r = [];
        for j=1:length(X0)
            % Extract the i-th input set.
            xi = 1/2*(X0{j}.sup + X0{j}.inf);
            ri = 1/2*(X0{j}.sup - X0{j}.inf);
            if permuteInputDims
                xi = reshape(permute(reshape(xi,inSize),[2 1 3]),[],1);
                ri = reshape(permute(reshape(ri,inSize),[2 1 3]),[],1);
            end
            % Append the i-th input set to the batch.
            x = [x xi];
            r = [r ri];
        end

        % Extract specification.
        % Handle multiple specs.
        for i=1:length(specs)
            if isa(specs(i).set,'halfspace')
                A = specs(i).set.c';
                b = -specs(i).set.d;
            else
                A = specs(i).set.A;
                b = -specs(i).set.b;
            end
            safeSet = strcmp(specs(i).type,'safeSet');
            
            % Measure verification time.
            tic
            
            while true
                try
                    % Do verification.
                    [res_,x_,y_] = nn.verify(x,r,A,b,safeSet, ...
                        options,timeout,verbose,false);
                    break;
                catch e
                    if ismember(e.identifier, ...
                            {'parallel:gpu:array:pmaxsize', ...
                                'parallel:gpu:array:OOM', ...
                                'MATLAB:array:SizeLimitExceeded'}) ...
                            && options.nn.train.mini_batch_size > 1
                        options.nn.train.mini_batch_size = ...
                            floor(1/2*options.nn.train.mini_batch_size);
                        fprintf('--- OOM error: half batchSize %d...\n', ...
                            options.nn.train.mini_batch_size);
                    else
                        % Print the error message. 
                        fprintf(newline);
                        fprintf(e.message);
                        fprintf(newline);
                        % No result.
                        res_ = struct('str','unknown','time',-1);
                        break;
                    end
                end
            end
            fprintf(' done\n');
    
            fprintf('Writing results...\n');
            fprintf('--- opening results file ...');
            % Open results file.
            fid = fopen(resultsPath,'w');
            fprintf(' done\n');
    
            fprintf('--- writing file ...');
            % Write results.
            if strcmp(res_.str,'VERIFIED')
                res = 'unsat';
                % Write content.
                fprintf(fid,['unsat' newline]);
                fclose(fid);
            elseif strcmp(res_.str,'COUNTEREXAMPLE')
                res = 'sat';
                % Reorder input dimensions...
                if permuteInputDims
                  x_ = reshape(permute(reshape(x_,inSize([2 1 3])),[2 1 3]),[],1);
                end
                % Write content.
                fprintf(fid,['sat' newline '(']);
                % Write input values.
                for j=1:size(x_,1)
                    fprintf(fid,['(X_%d %f)' newline],j-1,x_(j));
                end
                % Write output values.
                for j=1:size(y_,1)
                    fprintf(fid,['(Y_%d %f)' newline],j-1,y_(j));
                end
                fprintf(fid,')');
                fclose(fid);
                % Found a counterexample. We do not need to check the
                % remaining specifications.
                break;
            else
                res = 'unknown';
                % We cannot verify an input set; we dont have to check the other
                % input sets.
                fprintf(fid,['unknown' newline]);
                fclose(fid);
                % We do not need to check the remaining specifications.
                break;
            end
            fprintf(' done\n');
        end
        

    catch e
        fprintf(e.message);
        % There is some issue with the parsing; e.g. acasxu prop_6.vnnlib
        res = 'unknown';
        fprintf(' done\n');

        % Open results file.
        fid = fopen(resultsPath,'w');
        fprintf(fid,['unknown' newline]);
        fclose(fid);
    end

    if verbose
        % Print result.
        fprintf('%s -- %s: %s\n',modelPath,vnnlibPath,res);
        time = toc;
        fprintf('--- Verification time: %.4f / %.4f [s]\n',time,timeout);
    end

end