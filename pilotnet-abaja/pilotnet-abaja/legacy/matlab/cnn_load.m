function net = cnn_load(onnxPath, backend)
%CNN_LOAD Load the exported PilotNet model.
%   backend = 'native' (default): plain-MATLAB forward pass using
%             pilotnet_weights.mat written next to the .onnx by
%             export_onnx.py. No toolbox, no Python; verified against ONNX.
%   backend = 'python': ONNX Runtime through MATLAB's Python
%             interface. Needs pyenv pointing at the Python that has
%             onnxruntime + numpy (the one you trained with), e.g.
%             pyenv('Version','C:\path\to\venv\Scripts\python.exe')
%   backend = 'matlab': Deep Learning Toolbox ONNX importer
%             (importNetworkFromONNX, R2023b+, or importONNXNetwork).
%   The network takes a 66x200x3 RGB image (0..255) and returns one value per
%   head: [FOLLOW LEFT RIGHT] for the conditional model.
if nargin < 2, backend = 'native'; end
net.backend = backend;
if isempty(regexp(onnxPath, '^([A-Za-z]:|[\\/])', 'once'))      % make absolute for Python
    onnxPath = fullfile(pwd, onnxPath);
end
if ~exist(onnxPath, 'file'), error('ONNX model not found: %s', onnxPath); end
switch backend
    case 'native'
        matPath = fullfile(fileparts(onnxPath), 'pilotnet_weights.mat');
        net = pilotnet_native_load(matPath);
    case 'python'
        pyroot = fullfile(fileparts(mfilename('fullpath')), '..', 'python');
        p = cellfun(@char, cell(py.sys.path), 'UniformOutput', false);
        if ~any(strcmp(p, pyroot))
            insert(py.sys.path, int32(0), pyroot);
        end
        mod = py.importlib.import_module('pilotnet_infer');
        net.obj = mod.PilotNetONNX(onnxPath);
    case 'matlab'
        if exist('importNetworkFromONNX', 'file')
            net.obj = importNetworkFromONNX(onnxPath);
            net.kind = 'dlnetwork';
        else
            net.obj = importONNXNetwork(onnxPath, 'OutputLayerType', 'regression');
            net.kind = 'dag';
        end
    otherwise
        error('unknown backend %s', backend);
end
end
