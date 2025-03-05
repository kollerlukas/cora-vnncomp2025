function [instanceFilename,modelName,vnnlibName] = ...
    getInstanceFilename(benchName,modelPath,vnnlibPath)
  % Create filename.
  modelName = regexp(modelPath,'([^/]+)(?=\.onnx$)','match');
  vnnlibName = regexp(vnnlibPath,'([^/]+)(?=\.vnnlib$)','match');
  instanceFilename = [benchName '_' modelName{1} '_' ...
      vnnlibName{1} '.mat'];
end