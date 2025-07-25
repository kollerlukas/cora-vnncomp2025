classdef nnReLULayer < nnLeakyReLULayer
% nnReLULayer - class for ReLU layers
%
% Syntax:
%    obj = nnReLULayer(name)
%
% Inputs:
%    name - name of the layer, defaults to type
%
% Outputs:
%    obj - generated object
%
% References:
%    [1] Tran, H.-D., et al. "Star-Based Reachability Analysis of Deep
%        Neural Networks", 2019
%
% Other m-files required: none
% Subfunctions: none
% MAT-files required: none
%
% See also: neuralNetwork

% Authors:       Tobias Ladner, Sebastian Sigl, Lukas Koller
% Written:       28-March-2022
% Last update:   07-July-2022 (SS, update nnLeakyReLULayer)
%                04-March-2024 (LK, re-implemented evalNumeric & df_i for performance)
% Last revision: 10-August-2022 (renamed)

% ------------------------------ BEGIN CODE -------------------------------

methods
    % constructor
    function obj = nnReLULayer(name)
        if nargin < 1
            name = [];
        end
        % call super class constructor
        obj@nnLeakyReLULayer(0, name)
    end
end

% evaluate ----------------------------------------------------------------

methods (Access = {?nnLayer, ?neuralNetwork})

    % numeric
    function r = evaluateNumeric(obj, input, options)
        r = max(0, input);
    end
    
    % conZonotope
    function [c, G, C, d, l_, u_] = evaluateConZonotopeNeuron(obj, c, G, C, d, l_, u_, j, options)
        % enclose the ReLU activation function with a constrained zonotope based on
        % the results for star sets in [1]

        n = length(c);
        m = size(G, 2);
        M = eye(n);
        M(:, j) = zeros(n, 1);

        % get lower bound
        if options.nn.bound_approx
            c_ = c(j) + 0.5 * G(j, :) * (u_ - l_);
            G_ = 0.5 * G(j, :) * diag(u_-l_);
            l = c_ - sum(abs(G_));
        else
            problem.f = G(j, :);
            problem.Aineq = C;
            problem.bineq = d;
            problem.Aeq = []; problem.beq = [];
            problem.lb = []; problem.ub = [];
            [~, temp] = CORAlinprog(problem);
            l = c(j) + temp;
        end

        % compute output according to Sec. 3.2 in [1]
        if l < 0

            % compute upper bound
            if options.nn.bound_approx
                u = c_ + sum(abs(G_));
            else
                problem.f = -G(j, :);
                problem.Aineq = C;
                problem.bineq = d;
                problem.Aeq = []; problem.beq = [];
                problem.lb = []; problem.ub = [];
                [~, temp] = CORAlinprog(problem);
                u = c(j) - temp;
            end

            if u <= 0
                % l <= u <= 0 -> linear
                c = M * c;
                G = M * G;
            else
                % compute relaxation

                % constraints and offset
                C = [
                    C, zeros(size(C, 1), 1)
                    zeros(1, m), -1;
                    G(j, :), -1;
                    -u / (u - l) * G(j, :), 1;
                    ];
                d = [d; 0; -c(j); u / (u - l) * (c(j) - l)];

                % center and generators
                c = M * c;
                G = [M * G, unitvector(j,n)];

                % bounds
                l_ = [l_; 0];
                u_ = [u_; u];
            end
        end
    end
end

methods
    function buckets = getMergeBuckets(obj)
        buckets = 0;
    end
end

end

% ------------------------------ END OF CODE ------------------------------
