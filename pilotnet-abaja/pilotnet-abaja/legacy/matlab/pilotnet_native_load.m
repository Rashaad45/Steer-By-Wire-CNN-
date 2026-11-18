function net = pilotnet_native_load(matPath)
%PILOTNET_NATIVE_LOAD Load PilotNet weights exported by python/export_onnx.py
%   (pilotnet_weights.mat) and pre-compute im2col gather indices, so the
%   forward pass is a handful of matrix products in plain MATLAB - no
%   toolbox and no Python needed. Verified against ONNX Runtime by
%   abaja_render_check.m.
S = load(matPath);
net.backend = 'native';
net.nBranches = double(S.n_branches);
net.normW = double(S.norm_w);            % 3x3 RGB->YUV matrix
net.normB = double(S.norm_b(:));
H = 66; W = 200; C = 3;
for l = 1:5
    w = double(S.(sprintf('conv%d_w', l)));          % Cout x Cin x k x k
    b = double(S.(sprintf('conv%d_b', l)));
    s = double(S.(sprintf('conv%d_s', l)));
    [Cout, Cin, k, ~] = size(w);
    assert(Cin == C);
    Ho = floor((H - k)/s) + 1; Wo = floor((W - k)/s) + 1;
    % patch order: row offset u fastest, then column offset v, then channel c
    [u, v, c] = ndgrid(0:k-1, 0:k-1, 0:C-1);
    patchOff = u(:) + H*v(:) + H*W*c(:);                         % P x 1
    [i, j] = ndgrid(0:Ho-1, 0:Wo-1);                             % output i fastest
    posBase = (i(:)'*s + 1) + H*(j(:)'*s);                       % 1 x Q
    net.idx{l} = patchOff + posBase;                             % P x Q (implicit expansion)
    net.Wm{l} = reshape(permute(w, [1 3 4 2]), Cout, []);        % Cout x (k*k*Cin)
    net.b{l} = b(:);
    net.outSize{l} = [Ho Wo Cout];
    H = Ho; W = Wo; C = Cout;
end
for br = 1:net.nBranches
    for f = 1:4
        net.fcW{br,f} = double(S.(sprintf('head%d_fc%d_w', br-1, f)));
        net.fcB{br,f} = double(S.(sprintf('head%d_fc%d_b', br-1, f)));
        net.fcB{br,f} = net.fcB{br,f}(:);
    end
end
end
