function [kappa, i, cte] = pure_pursuit(cfg, route, x, y, psi, v, hint)
%PURE_PURSUIT Expert/baseline lateral controller (Coulter, CMU-RI-TR-92-01).
%   Goal point at arc length L_d = lookMin + lookGain*v ahead on the route.
%   In the vehicle frame the goal is (dx, dy); the arc through it has
%   curvature kappa = 2*dy / L_d^2  (= 2 sin(alpha) / L_d).
i = nearest_on_route(route, x, y, hint);
ld = cfg.lookMin + cfg.lookGain*v;
sT = route.s(i) + ld;
N = numel(route.s);
j = i;
while j < N && route.s(j) < sT
    j = j + 1;
end
px = route.pts(j,1); py = route.pts(j,2);
dx =  (px - x)*cos(psi) + (py - y)*sin(psi);
dy = -(px - x)*sin(psi) + (py - y)*cos(psi);
L2 = max(dx*dx + dy*dy, 1e-6);
kappa = min(max(2*dy/L2, -cfg.kappaMax), cfg.kappaMax);
cte = cross_track(route, i, x, y);
end
