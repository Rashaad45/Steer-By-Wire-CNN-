function y = pilotnet_native_forward(net, img)
%PILOTNET_NATIVE_FORWARD PilotNet inference in plain MATLAB.
%   img: 66x200x3 uint8 RGB. y: 1 x nBranches.
%   1) fixed RGB->YUV, x/127.5 - 1      2) 5 x [conv (im2col*GEMM) + ELU]
%   3) flatten in PyTorch (C,H,W) order  4) per-branch FC 100-50-10-1 (ELU)
X = double(img);
A = zeros(size(X));
for c = 1:3
    A(:,:,c) = (net.normW(c,1)*X(:,:,1) + net.normW(c,2)*X(:,:,2) + ...
                net.normW(c,3)*X(:,:,3) + net.normB(c)) / 127.5 - 1;
end
for l = 1:5
    Z = net.Wm{l} * A(net.idx{l}) + net.b{l};          % Cout x (Ho*Wo)
    sz = net.outSize{l};
    A = elu(reshape(Z.', sz(1), sz(2), sz(3)));
end
f = reshape(permute(A, [2 1 3]), [], 1);               % = torch flatten of (C,H,W)
y = zeros(1, net.nBranches);
for br = 1:net.nBranches
    h = f;
    for k = 1:4
        h = net.fcW{br,k} * h + net.fcB{br,k};
        if k < 4, h = elu(h); end
    end
    y(br) = h;
end
end

function x = elu(x)
m = x < 0;
x(m) = exp(x(m)) - 1;
end
