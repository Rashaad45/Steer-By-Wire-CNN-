function P = hermite_chain(ctrl, h0, h1, ds, k)
%HERMITE_CHAIN Cubic Hermite spline through control points, resampled at ds.
%   End tangents follow node headings h0,h1 [rad]; interior tangents are
%   Catmull-Rom. Mirrors python/sim/course.py:hermite_chain.
if nargin < 5, k = 200; end
n = size(ctrl,1);
m = zeros(n,2);
for i = 2:n-1
    m(i,:) = 0.5*(ctrl(i+1,:) - ctrl(i-1,:));
end
m(1,:) = [cos(h0) sin(h0)] * norm(ctrl(2,:) - ctrl(1,:));
m(n,:) = [cos(h1) sin(h1)] * norm(ctrl(n,:) - ctrl(n-1,:));

t = linspace(0,1,k+1)'; t = t(1:end-1);
h00 = 2*t.^3 - 3*t.^2 + 1;
h10 = t.^3 - 2*t.^2 + t;
h01 = -2*t.^3 + 3*t.^2;
h11 = t.^3 - t.^2;
dense = zeros((n-1)*k + 1, 2);
for i = 1:n-1
    seg = h00*ctrl(i,:) + h10*m(i,:) + h01*ctrl(i+1,:) + h11*m(i+1,:);
    dense((i-1)*k+1:i*k, :) = seg;
end
dense(end,:) = ctrl(n,:);

s = [0; cumsum(hypot(diff(dense(:,1)), diff(dense(:,2))))];
nOut = floor(s(end)/ds);
sNew = (0:nOut)' * ds;
if s(end) - nOut*ds > 1e-9
    sNew = [sNew; s(end)];
end
P = [interp1(s, dense(:,1), sNew, 'linear'), interp1(s, dense(:,2), sNew, 'linear')];
end
