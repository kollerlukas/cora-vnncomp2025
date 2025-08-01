function [z] = computeFGSMAttack(obj,x,t,options,epsilon,varargin)
% computeFGSMAttack - Compute a Fast Gradient Sign (FGSM) adversarial attack.
%
% Syntax:
%    [z] = computeFGSMAttack(obj,x,y,options,epsilon)
%
% Inputs:
%    obj - neural network
%    x - input value
%    t - target value
%    options - evaluation parameters (see neuralNetwork.evaluate)
%    epsilon - perturbation radius
%    lossDer - derivative of loss function
%    idxLayer - indices of layers to be evaluated
%
% Outputs:
%    z - altered input
%    
% References:
%    [1] I. J. Goodfellow, J. Shlens, and C. Szegedy. "Explaining and 
%        Harnessing Adversarial Examples." In: 3rd International Conference
%        on Learning Representations ICLR. San Diego, CA, USA, 2015.
%
% Other m-files required: none
% Subfunctions: none
% MAT-files required: none
%
% See also: neuralNetwork

% Authors:       Lukas Koller
% Written:       21-June-2023
% Last update:   11-July-2023 (fixed bug in computation)
%                12-July-2023 (corrected sign in computation)
%                07-September-2023 (batch-wise computation)
% Last revision: ---    

% ------------------------------ BEGIN CODE -------------------------------

% parse input
narginchk(5,7)
[lossDer,idxLayer] = setDefaultValues(...
    {@(t,y) softmax(y) - t,1:length(obj.layers)},varargin);

% validate input
inputArgsCheck({ ...
    {obj,'att','neuralNetwork'}; ...
    {x,'att','numeric'}; ...
    {t,'att','numeric'}; ... 
    {options,'att','struct'}; ... 
    {epsilon,'att','numeric'};
})

% % Alternative, but less efficient implementation.
% % compute sensitivity and forward propagation
% [S,y] = obj.calcSensitivity(x,options,false,idxLayer);
% % compute gradient
% grad = reshape(pagemtimes(S,'transpose', ...
%     permute(lossDer(t,y),[1 3 2]),'none'),size(x));

% 1. Forward propagation, with backprop enabled.
options.nn.train.backprop = true;
y = obj.evaluate(x,options,idxLayer);
% 2. Backpropagation.
grad = obj.backprop(lossDer(t,y),options,idxLayer,false);

% compute modified input; linear estimate on worst-case perturbation
z = x + epsilon.*sign(grad);
end

% ------------------------------ END OF CODE ------------------------------
