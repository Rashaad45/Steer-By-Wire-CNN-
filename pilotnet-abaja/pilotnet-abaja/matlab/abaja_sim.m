function out = abaja_sim(action, varargin)
%ABAJA_SIM aBAJA end-to-end CNN steering simulator - the whole MATLAB side in ONE file.
%
%   Navigation (road graph, A*, FOLLOW/LEFT/RIGHT commands), camera renderer,
%   kinematic bicycle + pure-pursuit expert, dataset generator, closed-loop
%   scoring and a toolbox-free PilotNet forward pass. Bit-compatible with
%   pilotnet_all.py (same course, texture, renderer and labels).
%   Base MATLAB R2019b+ only (no toolboxes); also runs in GNU Octave 7+.
%
%   abaja_sim('demo')                          course -> A* -> render check -> expert -> CNN
%   abaja_sim('demo', onnxPath, backend, cycles)
%   abaja_sim('generate', outDir, cycles, seed)          e.g. abaja_sim('generate', '../datasets/sim_matlab', 6, 0)
%   res = abaja_sim('closedloop', policy, onnxPath, cycles, opts)
%         policy 'expert' | 'cnn'; onnxPath ignored for 'expert' (pass [])
%         opts: latencySteps (1), speed (5), startOffset (0), plot (true),
%               video ('' or file.mp4, MATLAB only), backend ('native')
%         res.trajectory = [t x y psi delta cte cmd kappa s] log
%   ok  = abaja_sim('check')                   MATLAB renders/CNN == Python reference
%   ok  = abaja_sim('check', onnxPath, backend)
%   route = abaja_sim('route')                 plot course + A* route S->A->S->B->S
%
%   CNN backends:  'native' (default, plain MATLAB, uses pilotnet_weights.mat
%   next to the .onnx), 'python' (ONNX Runtime via pyenv), 'matlab' (Deep
%   Learning Toolbox ONNX importer).
%   Train / export the model in Python first:  python pilotnet_all.py export --ckpt runs/sim/best.pt
if nargin < 1 || isempty(action), action = 'demo'; end
switch lower(action)
    case 'demo'
        main_abaja_demo(varargin{:});
        out = [];
    case 'generate'
        generate_dataset(varargin{:});
        out = [];
    case 'closedloop'
        policy = argn(varargin, 1, 'expert');
        onnxPath = argn(varargin, 2, default_onnx());
        cycles = argn(varargin, 3, 2);
        opts = argn(varargin, 4, struct());
        net = [];
        if ~strcmp(policy, 'expert')
            if isempty(onnxPath), onnxPath = default_onnx(); end
            net = cnn_load(onnxPath, getopt(opts, 'backend', 'native'));
        end
        [out, logT] = run_closed_loop(policy, net, cycles, opts);
        out.trajectory = logT;
    case 'check'
        onnxPath = argn(varargin, 1, default_onnx());
        backend = argn(varargin, 2, 'native');
        net = [];
        if ~isempty(onnxPath) && exist(onnxPath, 'file')
            net = cnn_load(onnxPath, backend);
        else
            fprintf('(no ONNX model at %s - checking renders only)\n', onnxPath);
        end
        out = abaja_render_check(net);
    case 'route'
        out = plot_route();
    otherwise
        error('abaja_sim: unknown action ''%s'' (demo | generate | closedloop | check | route)', action);
end
if nargout == 0 && isempty(out), clear out; end     % no "ans = []" after demo / generate
end

% =====================================================================
% helpers for the dispatcher
% =====================================================================
function v = argn(c, k, default)
if numel(c) >= k && ~isempty(c{k}), v = c{k}; else, v = default; end
end

function d = here_dir()
d = fileparts(mfilename('fullpath'));
end

function p = default_onnx()
p = fullfile(here_dir(), '..', 'runs', 'sim', 'pilotnet.onnx');
end

% =====================================================================
% main_abaja_demo  (was main_abaja_demo.m)
% =====================================================================
function main_abaja_demo(ONNX_PATH, BACKEND, CYCLES, GENERATE_MATLAB_DATASET)
% End-to-end CNN steering for an aBAJA autonomous vehicle (PilotNet)
%   1. Build the course road graph, A* route and command schedule
%   2. Check MATLAB renders exactly what Python trained on
%   3. Pure-pursuit expert run (upper bound with perfect localisation)
%   4. CNN-in-the-loop run (camera -> PilotNet -> steering), scored with the
%      NVIDIA autonomy metric, cross-track error and fork-command compliance
if nargin < 1 || isempty(ONNX_PATH), ONNX_PATH = default_onnx(); end
if nargin < 2 || isempty(BACKEND), BACKEND = 'native'; end   % 'native' | 'python' | 'matlab'
if nargin < 3 || isempty(CYCLES), CYCLES = 2; end            % each cycle = one LEFT lap + one RIGHT lap
if nargin < 4 || isempty(GENERATE_MATLAB_DATASET), GENERATE_MATLAB_DATASET = false; end
close all;
here = here_dir();

%% 1. Course, navigation graph, route
cfg = abaja_config();
course = build_course(cfg);
tex = build_texture(cfg, course);
route = build_route(cfg, course, {'S','A','S','B','S'});
draw_route(cfg, tex, route);
fprintf('Route %.1f m, forks at s = %s m\n', route.s(end), mat2str(round([route.forks.s],1)));

%% 2. Renderer / model equivalence with Python
net = [];
if exist(ONNX_PATH, 'file')
    net = cnn_load(ONNX_PATH, BACKEND);
end
if exist(fullfile(here, 'reference', 'reference.json'), 'file')
    abaja_render_check(net);
end
cam = camera_precompute(cfg);
figure('Color','w','Name','camera view at the fork');
i = find(route.s >= route.forks(1).s - 8, 1);
image(render_camera(cfg, cam, tex, route.pts(i,1), route.pts(i,2), route.heading(i))); axis image off;
title('Camera input 66x200 (8 m before the fork)');

%% 3. Expert baseline
resExpert = run_closed_loop('expert', [], CYCLES);

%% 4. CNN in the loop
if ~isempty(net)
    resCNN = run_closed_loop('cnn', net, CYCLES, struct('latencySteps', 1));
    % robustness: start 1.5 m off-centre and drive 20 % faster
    resCNNfast = run_closed_loop('cnn', net, CYCLES, struct('speed', 6.0, 'startOffset', 1.5));
    resCNNfast.policy = 'cnn 6m/s +1.5m';
    print_results([resExpert; resCNN; resCNNfast]);
else
    warning('No ONNX model at %s - train/export in Python first (see help abaja_sim).', ONNX_PATH);
end

%% 5. (optional) generate a training set from MATLAB instead of Python
if GENERATE_MATLAB_DATASET
    generate_dataset(fullfile(here, '..', 'datasets', 'sim_matlab'), 6, 0);
end
end

function print_results(R)
% results table without struct2table (works in Octave too)
fprintf('\n%-16s %6s %9s %9s %8s %8s %12s %6s\n', 'policy', 'interv', 'autonomy%', ...
    'cte_mean', 'cte_p95', 'cte_max', 'steer_rate', 'forks');
for k = 1:numel(R)
    r = R(k);
    fprintf('%-16s %6d %9.2f %9.4f %8.4f %8.4f %12.3f %3d/%d\n', r.policy, r.interventions, ...
        r.autonomy_pct, r.cte_mean_abs, r.cte_p95, r.cte_max, r.steer_rate_rms_deg_s, ...
        r.forks_correct, r.forks_total);
end
end

function route = plot_route()
cfg = abaja_config();
course = build_course(cfg);
tex = build_texture(cfg, course);
route = build_route(cfg, course, {'S','A','S','B','S'});
draw_route(cfg, tex, route);
fprintf('Route %.1f m, forks at s = %s m, commands %s\n', route.s(end), ...
    mat2str(round([route.forks.s],1)), mat2str([route.forks.cmd]));
end

function draw_route(cfg, tex, route)
figure('Color','w','Name','aBAJA course');
image('XData',[cfg.mapX0 cfg.mapX0+cfg.mapWm],'YData',[cfg.mapY0 cfg.mapY0+cfg.mapHm],'CData',tex);
axis xy equal tight; hold on;
cmap = [1 1 1; 0.2 0.6 1; 1 0.3 0.3];
for c = 0:2
    m = route.cmd == c;
    plot(route.pts(m,1), route.pts(m,2), '.', 'Color', cmap(c+1,:), 'MarkerSize', 4);
end
for k = 1:numel(cfg.nodeNames)
    text(cfg.nodeXYH(k,1)+1.5, cfg.nodeXYH(k,2)+1.5, cfg.nodeNames{k}, 'Color','w','FontWeight','bold');
end
title('Road graph, A* route (S\rightarrowA\rightarrowS\rightarrowB\rightarrowS); white FOLLOW, blue LEFT, red RIGHT');
end

% =====================================================================
% abaja_config  (was abaja_config.m) - must match pilotnet_all.py class C
% =====================================================================
function cfg = abaja_config()
% ---------------- course (BAJA-style off-road loop with one fork) ---------
cfg.nodeNames = {'S','J1','A','B','J2'};
cfg.nodeXYH   = [ 60  12  180;
                  20  45   90;
                  55  83    0;
                  58  58    0;
                 108  50  -90];
cfg.edges = { ...
    'S',  'J1', [36 13; 22 26];
    'J1', 'A',  [12 62; 22 80];      % left branch
    'A',  'J2', [88 84; 105 72];
    'J1', 'B',  [26 55; 40 59];      % right branch
    'B',  'J2', [80 57; 98 62];
    'J2', 'S',  [104 28; 88 14]};
cfg.trackWidth = 6.0;      % [m]
cfg.pathDs     = 0.25;     % centreline resampling [m]
% ---------------- map raster ---------------------------------------------
cfg.mapX0 = -15; cfg.mapY0 = -15;
cfg.mapWm = 150; cfg.mapHm = 120;
cfg.mapRes = 0.10;         % [m/cell]
% ---------------- vehicle (aBAJA-class ATV, kinematic bicycle) -----------
cfg.L         = 1.40;            % wheelbase [m]
cfg.maxSteer  = deg2rad(30);
cfg.steerTau  = 0.15;            % actuator lag [s]
cfg.steerRate = deg2rad(120);    % actuator slew [rad/s]
cfg.kappaMax  = tan(cfg.maxSteer) / cfg.L;
cfg.speed     = 5.0;             % [m/s]
cfg.dt        = 0.05;            % 20 Hz
% ---------------- camera ---------------------------------------------------
cfg.imgW = 200; cfg.imgH = 66;
cfg.ss = 2;                       % supersampling
cfg.hfov = deg2rad(100);
cfg.camHeight = 1.20;
cfg.camPitch  = deg2rad(15);
cfg.camX      = 1.00;
cfg.maxRange  = 60;
cfg.hazeDist  = 45;
cfg.sky  = [175 200 230];
cfg.haze = [190 200 210];
% ---------------- expert / navigation -------------------------------------
cfg.lookMin  = 3.0;
cfg.lookGain = 0.4;
cfg.cmdBefore = 18.0;
cfg.cmdAfter  = 8.0;
cfg.FOLLOW = 0; cfg.LEFT = 1; cfg.RIGHT = 2;     % same indices as Python
end

% =====================================================================
% abaja_render_check  (was abaja_render_check.m)
% =====================================================================
function ok = abaja_render_check(net)
% Renders the reference poses in reference/reference.json and compares them
% with the PNGs rendered by Python; with a network, also the CNN outputs.
here = here_dir();
ref = jsondecode(fileread(fullfile(here, 'reference', 'reference.json')));
cfg = abaja_config();
course = build_course(cfg);
tex = build_texture(cfg, course);
cam = camera_precompute(cfg);
ok = true;
for k = 1:numel(ref.poses)
    p = ref.poses(k);
    img = render_camera(cfg, cam, tex, p.x, p.y, p.psi);
    refImg = imread(fullfile(here, 'reference', p.file));
    d = abs(double(img) - double(refImg));
    fprintf('pose %d: mean |diff| = %.3f, max |diff| = %d', k, mean(d(:)), max(d(:)));
    good = mean(d(:)) < 0.5;
    if nargin > 0 && ~isempty(net)
        y = cnn_predict(net, refImg);
        dy = max(abs(y(:) - p.cnn(:)));
        fprintf(', CNN max diff = %.2e', dy);
        good = good && dy < 1e-3;
    end
    fprintf('  -> %s\n', ternary(good, 'OK', 'MISMATCH'));
    ok = ok && good;
end
route = build_route(cfg, course, {'S','A','S','B','S'});
fprintf('route length: MATLAB %.4f m, Python %.4f m\n', route.s(end), ref.route_length);
ok = ok && abs(route.s(end) - ref.route_length) < 1e-6;
end

function out = ternary(c, a, b)
if c, out = a; else, out = b; end
end

% =====================================================================
% navigation layer: build_course, hermite_chain, astar_graph, build_route
% =====================================================================
function course = build_course(cfg)
% Road graph of the aBAJA course: nodes + Hermite centreline edges.
nN = numel(cfg.nodeNames);
course.names = cfg.nodeNames;
course.xyh = [cfg.nodeXYH(:,1:2), deg2rad(cfg.nodeXYH(:,3))];
nE = size(cfg.edges,1);
course.edges = struct('from',cell(nE,1),'to',[],'pts',[],'length',[]);
course.outEdges = cell(nN,1);
for e = 1:nE
    a = node_index(course, cfg.edges{e,1});
    b = node_index(course, cfg.edges{e,2});
    ctrl = [course.xyh(a,1:2); cfg.edges{e,3}; course.xyh(b,1:2)];
    P = hermite_chain(ctrl, course.xyh(a,3), course.xyh(b,3), cfg.pathDs);
    course.edges(e).from = a;
    course.edges(e).to = b;
    course.edges(e).pts = P;
    course.edges(e).length = sum(hypot(diff(P(:,1)), diff(P(:,2))));
    course.outEdges{a}(end+1) = e;
end
end

function i = node_index(course, name)
i = find(strcmp(course.names, name), 1);
if isempty(i), error('unknown node %s', name); end
end

function P = hermite_chain(ctrl, h0, h1, ds, k)
% Cubic Hermite spline through control points, resampled at ds.
% End tangents follow node headings h0,h1 [rad]; interior tangents are Catmull-Rom.
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

function edgePath = astar_graph(course, startNode, goalNode)
% A* search on the directed road graph (Hart, Nilsson & Raphael 1968).
% g(n) = travelled centreline length, h(n) = straight-line distance to the goal.
nN = size(course.xyh,1);
g = inf(nN,1); g(startNode) = 0;
f = inf(nN,1); f(startNode) = astar_heur(course, startNode, goalNode);
cameFromNode = zeros(nN,1); cameFromEdge = zeros(nN,1);
open = false(nN,1); open(startNode) = true;
closed = false(nN,1);
while any(open)
    cand = find(open);
    [~,k] = min(f(cand));
    n = cand(k);
    if n == goalNode, break; end
    open(n) = false; closed(n) = true;
    for e = course.outEdges{n}
        m = course.edges(e).to;
        if closed(m), continue; end
        gNew = g(n) + course.edges(e).length;
        if gNew < g(m)
            g(m) = gNew; f(m) = gNew + astar_heur(course, m, goalNode);
            cameFromNode(m) = n; cameFromEdge(m) = e;
            open(m) = true;
        end
    end
end
if isinf(g(goalNode)), error('no route %d -> %d', startNode, goalNode); end
edgePath = [];
n = goalNode;
while n ~= startNode
    edgePath = [cameFromEdge(n), edgePath]; %#ok<AGROW>
    n = cameFromNode(n);
end
end

function h = astar_heur(course, n, goalNode)
h = hypot(course.xyh(goalNode,1) - course.xyh(n,1), course.xyh(goalNode,2) - course.xyh(n,2));
end

function route = build_route(cfg, course, mission)
% A* legs between mission checkpoints joined into one route polyline with a
% high-level command at every point (FOLLOW / LEFT / RIGHT around forks).
edgeSeq = [];
for k = 1:numel(mission)-1
    a = find(strcmp(course.names, mission{k}), 1);
    b = find(strcmp(course.names, mission{k+1}), 1);
    edgeSeq = [edgeSeq, astar_graph(course, a, b)]; %#ok<AGROW>
end
pts = zeros(0,2); sAcc = 0;
forks = struct('s',{},'edge',{},'node',{},'cmd',{},'others',{});
for k = 1:numel(edgeSeq)
    e = edgeSeq(k);
    E = course.edges(e);
    if k == 1, P = E.pts; else, P = E.pts(2:end,:); end      % drop duplicate node
    outs = course.outEdges{E.from};
    if numel(outs) > 1
        f = struct('s', sAcc, 'edge', e, 'node', E.from, ...
                   'cmd', fork_turn(cfg, course, E.from, e), 'others', outs(outs ~= e));
        forks(end+1) = f; %#ok<AGROW>
    end
    pts = [pts; P]; %#ok<AGROW>
    sAcc = sAcc + E.length;
end
route.pts = pts;
route.s = [0; cumsum(hypot(diff(pts(:,1)), diff(pts(:,2))))];
d = diff(pts);
hd = atan2(d(:,2), d(:,1));
route.heading = [hd; hd(end)];
route.cmd = repmat(cfg.FOLLOW, size(pts,1), 1);
for k = 1:numel(forks)
    m = route.s >= forks(k).s - cfg.cmdBefore & route.s <= forks(k).s + cfg.cmdAfter;
    route.cmd(m) = forks(k).cmd;
end
route.forks = forks;
end

function c = fork_turn(cfg, course, node, chosen)
% LEFT if the chosen edge, probed 10 m after the node, lies furthest to the
% left of the node heading; otherwise RIGHT.
probe = 10;
x0 = course.xyh(node,1); y0 = course.xyh(node,2); hd = course.xyh(node,3);
outs = course.outEdges{node};
off = zeros(size(outs));
for i = 1:numel(outs)
    P = course.edges(outs(i)).pts;
    j = min(round(probe/cfg.pathDs), size(P,1)-1) + 1;
    off(i) = -(P(j,1)-x0)*sin(hd) + (P(j,2)-y0)*cos(hd);
end
if off(outs == chosen) == max(off), c = cfg.LEFT; else, c = cfg.RIGHT; end
end

% =====================================================================
% camera simulator: build_texture, camera_precompute, render_camera
% =====================================================================
function [tex, D] = build_texture(cfg, course)
% Procedural ground texture (uint8 HxWx3). Same hash noise and arithmetic as Python.
% Row r <-> y = mapY0 + (r-0.5)*res, column c <-> x = mapX0 + (c-0.5)*res.
W = round(cfg.mapWm/cfg.mapRes);
H = round(cfg.mapHm/cfg.mapRes);
D = inf(H, W);
R = cfg.trackWidth/2 + 1.0;
P = vertcat(course.edges.pts);
for k = 1:size(P,1)
    px = P(k,1); py = P(k,2);
    c0 = max(1, floor((px - R - cfg.mapX0)/cfg.mapRes) + 1);
    c1 = min(W, ceil((px + R - cfg.mapX0)/cfg.mapRes) + 1);
    r0 = max(1, floor((py - R - cfg.mapY0)/cfg.mapRes) + 1);
    r1 = min(H, ceil((py + R - cfg.mapY0)/cfg.mapRes) + 1);
    xs = cfg.mapX0 + ((c0:c1) - 0.5)*cfg.mapRes;
    ys = cfg.mapY0 + ((r0:r1)' - 0.5)*cfg.mapRes;
    d = hypot(xs - px, ys - py);                     % implicit expansion (R2016b+)
    D(r0:r1, c0:c1) = min(D(r0:r1, c0:c1), d);
end
[X, Y] = meshgrid(cfg.mapX0 + ((1:W) - 0.5)*cfg.mapRes, cfg.mapY0 + ((1:H) - 0.5)*cfg.mapRes);
n1 = 0.5*value_noise(X,Y,4.0,1) + 0.3*value_noise(X,Y,1.0,2) + 0.2*value_noise(X,Y,0.3,3);
n2 = 0.5*value_noise(X,Y,4.0,4) + 0.3*value_noise(X,Y,1.0,5) + 0.2*value_noise(X,Y,0.3,6);
bush = value_noise(X,Y,6.0,7);

track = D <= cfg.trackWidth/2;
rut   = abs(D - 0.85) < 0.18;
berm  = D >= cfg.trackWidth/2 - 0.35 & D <= cfg.trackWidth/2;
fd = ones(H,W); fd(rut) = 0.80; fd(berm) = fd(berm) * 1.12;
fg = ones(H,W); fg(bush > 0.72) = 0.55;
dirtBase  = [150 112 78];
grassBase = [72 118 52];
tex = zeros(H, W, 3);
for ch = 1:3
    dirt  = dirtBase(ch)  * (0.78 + 0.44*n1) .* fd;
    grass = grassBase(ch) * (0.70 + 0.60*n2) .* fg;
    tex(:,:,ch) = grass;
    tmp = tex(:,:,ch); tmp(track) = dirt(track); tex(:,:,ch) = tmp;
end
tex = uint8(min(max(floor(tex + 0.5), 0), 255));
end

function v = value_noise(X, Y, scale, seed)
u = X/scale; w = Y/scale;
i = floor(u); j = floor(w);
fu = u - i; fv = w - j;
fu = fu.*fu.*(3 - 2*fu);
fv = fv.*fv.*(3 - 2*fv);
a = hash2(i,   j,   seed);
b = hash2(i+1, j,   seed);
c = hash2(i,   j+1, seed);
d = hash2(i+1, j+1, seed);
v = (a.*(1-fu) + b.*fu).*(1-fv) + (c.*(1-fu) + d.*fu).*fv;
end

function h = hash2(i, j, seed)
v = sin(i*12.9898 + j*78.233 + seed*37.719) * 43758.5453;
h = v - floor(v);
end

function cam = camera_precompute(cfg)
% Intersect every (super-sampled) pixel ray with the ground.
% Pinhole camera at height h, pitched down by theta, camX ahead of the rear axle.
% For pixel (u,v): x_c=(u-cx)/f (right), y_c=(v-cy)/f (down). Ray (forward, left, up):
%     d = [cos(th) - y_c sin(th),  -x_c,  -sin(th) - y_c cos(th)]
% Ground hit (z = 0) at t = h / (-d_up) when d_up < 0.
ss = cfg.ss;
W = cfg.imgW*ss; H = cfg.imgH*ss;
f = (cfg.imgW/2) / tan(cfg.hfov/2);
u = ((0:W-1) + 0.5)/ss;
v = ((0:H-1) + 0.5)/ss;
[U, V] = meshgrid(u, v);
xc = (U - cfg.imgW/2)/f;
yc = (V - cfg.imgH/2)/f;
th = cfg.camPitch;
dF = cos(th) - yc*sin(th);
dL = -xc;
dU = -sin(th) - yc*cos(th);
t = inf(size(dU));
t(dU < 0) = cfg.camHeight ./ -dU(dU < 0);
dist = t .* hypot(dF, dL);
cam.valid = isfinite(t) & dist <= cfg.maxRange;
cam.pf = zeros(size(t)); cam.pl = zeros(size(t));
cam.pf(cam.valid) = cfg.camX + t(cam.valid).*dF(cam.valid);
cam.pl(cam.valid) = t(cam.valid).*dL(cam.valid);
cam.haze = ones(size(t));
cam.haze(cam.valid) = 1 - exp(-dist(cam.valid)/cfg.hazeDist);
cam.ss = ss;
end

function img = render_camera(cfg, cam, tex, x, y, psi)
% 66x200x3 uint8 RGB view from rear-axle pose (x, y, psi).
c = cos(psi); s = sin(psi);
X = x + cam.pf*c - cam.pl*s;
Y = y + cam.pf*s + cam.pl*c;
[Hm, Wm, ~] = size(tex);
col = min(max(floor((X - cfg.mapX0)/cfg.mapRes) + 1, 1), Wm);
row = min(max(floor((Y - cfg.mapY0)/cfg.mapRes) + 1, 1), Hm);
idx = sub2ind([Hm Wm], row, col);
w = cam.haze;
Hs = size(X,1); Ws = size(X,2);
img = zeros(Hs, Ws, 3);
for ch = 1:3
    T = tex(:,:,ch);
    g = double(T(idx));
    v = g.*(1 - w) + cfg.haze(ch)*w;
    v(~cam.valid) = cfg.sky(ch);
    img(:,:,ch) = v;
end
ss = cam.ss;
img = reshape(img, ss, cfg.imgH, ss, cfg.imgW, 3);    % box-filter down-sample
img = reshape(mean(mean(img, 1), 3), cfg.imgH, cfg.imgW, 3);
img = uint8(min(max(floor(img + 0.5), 0), 255));
end

% =====================================================================
% vehicle + expert: bicycle_step, pure_pursuit, nearest_on_route, cross_track
% =====================================================================
function st = bicycle_step(cfg, st, deltaCmd, v)
% Kinematic bicycle (rear-axle reference) + steering actuator.
% st = [x y psi delta].  delta' = sat((delta_cmd - delta)/tau, rate)
% x' = v cos(psi),  y' = v sin(psi),  psi' = v tan(delta) / L. Semi-implicit Euler.
dt = cfg.dt;
deltaCmd = min(max(deltaCmd, -cfg.maxSteer), cfg.maxSteer);
rate = (deltaCmd - st(4)) / cfg.steerTau;
rate = min(max(rate, -cfg.steerRate), cfg.steerRate);
d = min(max(st(4) + rate*dt, -cfg.maxSteer), cfg.maxSteer);
x = st(1) + v*cos(st(3))*dt;
y = st(2) + v*sin(st(3))*dt;
psi = st(3) + v/cfg.L*tan(d)*dt;
psi = atan2(sin(psi), cos(psi));
st = [x y psi d];
end

function [kappa, i, cte] = pure_pursuit(cfg, route, x, y, psi, v, hint)
% Pure pursuit (Coulter, CMU-RI-TR-92-01). Goal point L_d = lookMin + lookGain*v
% ahead on the route; kappa = 2*dy / L_d^2 in the vehicle frame.
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

function i = nearest_on_route(route, x, y, hint)
% Closest route index in a window around hint (40 back, 240 ahead = 10 m / 60 m).
N = size(route.pts,1);
lo = max(1, hint - 40);
hi = min(N, hint + 239);
seg = route.pts(lo:hi,:);
[~, k] = min((seg(:,1) - x).^2 + (seg(:,2) - y).^2);
i = lo + k - 1;
end

function e = cross_track(route, i, x, y)
% Signed lateral error to the route at index i (+ = left of path).
h = route.heading(i);
e = -(x - route.pts(i,1))*sin(h) + (y - route.pts(i,2))*cos(h);
end

% =====================================================================
% generate_dataset  (was generate_dataset.m)
% =====================================================================
function generate_dataset(outDir, cycles, seed)
% Expert demonstrations with recovery data. Writes outDir/images/*.png and
% outDir/labels.csv in the format read by  pilotnet_all.py train --dataset sim.
% Same algorithm as  pilotnet_all.py gen  (random streams differ, the
% course/renderer/labels do not).
% Labels: target = kappa_expert / kappa_max (+ = left), command 0 FOLLOW / 1 LEFT / 2 RIGHT.
if nargin < 1 || isempty(outDir), outDir = fullfile(here_dir(), '..', 'datasets', 'sim_matlab'); end
if nargin < 2 || isempty(cycles), cycles = 6; end
if nargin < 3 || isempty(seed), seed = 0; end
recordEvery = 2; noiseSigma = 0.06; maxOffset = 1.8; maxYaw = 20; speedJitter = 1.0; forkBoost = 3;

rng(seed);
cfg = abaja_config();
course = build_course(cfg);
tex = build_texture(cfg, course);
cam = camera_precompute(cfg);
mission = [{'S'}, repmat({'A','S','B','S'}, 1, cycles)];
route = build_route(cfg, course, mission);
if ~exist(fullfile(outDir,'images'), 'dir'), mkdir(fullfile(outDir,'images')); end
fprintf('[generate] %d cycle(s), seed %d -> %s\n', cycles, seed, outDir);

st = [route.pts(1,:), route.heading(1), 0];
hint = 1; nImg = 0; ou = 0; step = 0; noiseOn = false; noiseTimer = 0;
lapLen = route.s(end) / (2*cycles);
v = cfg.speed;
rows = zeros(0, 10); names = {};
N = size(route.pts,1);
stepsPerLap = round(lapLen / v / cfg.dt);
while hint < N - 4          % same stop point as Python (0-based hint < N-5)
    [kappa, hint, cte] = pure_pursuit(cfg, route, st(1), st(2), st(3), v, hint);
    if mod(step, stepsPerLap) == 0
        v = cfg.speed + (2*rand - 1)*speedJitter;
    end
    noiseTimer = noiseTimer - cfg.dt;
    if noiseTimer <= 0
        noiseOn = ~noiseOn; noiseTimer = 2 + 4*rand;
    end
    theta = 1.5;
    ou = ou - theta*ou*cfg.dt + noiseSigma*sqrt(2*theta*cfg.dt)*randn;
    kExec = kappa + noiseOn*ou;

    cmd = route.cmd(hint);
    if mod(step, recordEvery) == 0 || cmd ~= cfg.FOLLOW
        lap = floor(route.s(hint)/lapLen);
        [nImg, names, rows] = save_sample(outDir, cfg, nImg, names, rows, ...
            render_camera(cfg, cam, tex, st(1), st(2), st(3)), kappa, cmd, v, cte, st(1), st(2), st(3), 0, lap);
        nPert = 1 + forkBoost*(cmd ~= cfg.FOLLOW);
        for p = 1:nPert
            dl = (2*rand - 1)*maxOffset;
            dpsi = deg2rad(min(max(randn*maxYaw/2, -maxYaw), maxYaw));
            xp = st(1) - dl*sin(st(3)); yp = st(2) + dl*cos(st(3)); pp = st(3) + dpsi;
            [kp, ip, ctep] = pure_pursuit(cfg, route, xp, yp, pp, v, hint);
            if abs(ctep) < cfg.trackWidth/2 - 0.3
                [nImg, names, rows] = save_sample(outDir, cfg, nImg, names, rows, ...
                    render_camera(cfg, cam, tex, xp, yp, pp), kp, route.cmd(ip), v, ctep, xp, yp, pp, 1, lap);
            end
        end
    end
    st = bicycle_step(cfg, st, atan(cfg.L*kExec), v);
    step = step + 1;
    if mod(step, 2000) == 0
        fprintf('  step %d  s=%.0f/%.0f m  images=%d\n', step, route.s(hint), route.s(end), nImg);
    end
end

fid = fopen(fullfile(outDir, 'labels.csv'), 'w');
fprintf(fid, 'filename,target,kappa,command,speed,cte,x,y,psi,aug,lap\n');
for k = 1:nImg
    fprintf(fid, '%s,%.8f,%.8f,%d,%.4f,%.5f,%.4f,%.4f,%.6f,%d,%d\n', names{k}, rows(k,1), rows(k,2), ...
        rows(k,3), rows(k,4), rows(k,5), rows(k,6), rows(k,7), rows(k,8), rows(k,9), rows(k,10));
end
fclose(fid);
fprintf('done: %d images -> %s\n', nImg, outDir);
end

function [nImg, names, rows] = save_sample(outDir, cfg, nImg, names, rows, img, k, c, v, e, x, y, psi, aug, lap)
fn = sprintf('%06d.png', nImg);
imwrite(img, fullfile(outDir, 'images', fn));
nImg = nImg + 1;
names{nImg} = fn;
rows(nImg,:) = [k/cfg.kappaMax, k, c, v, e, x, y, psi, aug, lap];
end

% =====================================================================
% run_closed_loop  (was run_closed_loop.m)
% =====================================================================
function [res, logT] = run_closed_loop(policy, net, cycles, opts)
% Drive the aBAJA course with the CNN (or the expert) in the loop.
%   opts (optional struct): latencySteps (1 = 50 ms), speed, startOffset,
%                           plot (true), video ('' or file.mp4)
% Per step: mission checkpoints -> A* route -> command FOLLOW/LEFT/RIGHT from
% route progress -> CNN head selected by the command -> curvature ->
% delta = atan(L*kappa) -> actuator -> bicycle. The route is used only to
% issue the command and to score the run; the CNN steers from the image alone.
% Intervention (NVIDIA protocol): vehicle centre more than half a track width
% from the route -> counted, vehicle re-centred, 6 s penalty in the autonomy metric.
if nargin < 3 || isempty(cycles), cycles = 2; end
if nargin < 4, opts = struct(); end
latencySteps = getopt(opts, 'latencySteps', 1);
cfg = abaja_config();
v = getopt(opts, 'speed', cfg.speed);
doPlot = getopt(opts, 'plot', true);
videoFile = getopt(opts, 'video', '');
if ~isempty(videoFile) && ~exist('VideoWriter', 'class') && ~exist('VideoWriter', 'file')
    warning('VideoWriter is not available (GNU Octave) - no video written.');
    videoFile = '';
end

course = build_course(cfg);
tex = build_texture(cfg, course);
cam = camera_precompute(cfg);
route = build_route(cfg, course, [{'S'}, repmat({'A','S','B','S'}, 1, cycles)]);
st = [route.pts(1,1), route.pts(1,2) + getopt(opts, 'startOffset', 0), route.heading(1), 0];
hint = 1; t = 0;
queue = zeros(1, latencySteps);
N = size(route.pts,1);
logT = zeros(0, 9);
interv = zeros(0, 3);
forkRes = nan(1, numel(route.forks));
if ~isempty(videoFile)
    vw = VideoWriter(videoFile, 'MPEG-4'); vw.FrameRate = 20; open(vw);
end
fprintf('[closedloop] policy=%s cycles=%d speed=%.2f\n', policy, cycles, v);
while hint < N - 4          % same stop point as Python (0-based hint < N-5)
    x = st(1); y = st(2); psi = st(3);
    hint = nearest_on_route(route, x, y, hint);
    cte = cross_track(route, hint, x, y);
    cmd = route.cmd(hint);
    % --- did we take the commanded branch? (checked 15 m after each fork)
    for k = 1:numel(route.forks)
        f = route.forks(k);
        if isnan(forkRes(k)) && route.s(hint) >= f.s + 15
            dOk = min(hypot(course.edges(f.edge).pts(:,1) - x, course.edges(f.edge).pts(:,2) - y));
            dBad = inf;
            for o = f.others
                dBad = min(dBad, min(hypot(course.edges(o).pts(:,1) - x, course.edges(o).pts(:,2) - y)));
            end
            forkRes(k) = dOk <= dBad;
        end
    end
    % --- intervention
    if abs(cte) > cfg.trackWidth/2
        interv(end+1,:) = [t, x, y]; %#ok<AGROW>
        for k = 1:numel(route.forks)
            if isnan(forkRes(k)) && route.s(hint) >= route.forks(k).s - 5 && route.s(hint) <= route.forks(k).s + 25
                forkRes(k) = 0;
            end
        end
        st = [route.pts(hint,:), route.heading(hint), 0];
        queue = zeros(1, latencySteps);
        continue
    end
    % --- policy
    if strcmp(policy, 'expert')
        kappa = pure_pursuit(cfg, route, x, y, psi, v, hint);
    else
        img = render_camera(cfg, cam, tex, x, y, psi);
        heads = cnn_predict(net, img);
        if numel(heads) > 1, h = heads(cmd + 1); else, h = heads(1); end
        kappa = min(max(h, -1), 1) * cfg.kappaMax;
        if ~isempty(videoFile)
            writeVideo(vw, imresize_nn(img, 3));
        end
    end
    queue(end+1) = atan(cfg.L*kappa); %#ok<AGROW>
    dApplied = queue(1); queue(1) = [];
    logT(end+1,:) = [t, x, y, psi, st(4), cte, cmd, kappa, route.s(hint)]; %#ok<AGROW>
    st = bicycle_step(cfg, st, dApplied, v);
    t = t + cfg.dt;
end
if ~isempty(videoFile), close(vw); end

cteAbs = abs(logT(:,6));
res.policy = policy;
res.distance_m = route.s(end);
res.elapsed_s = t;
res.interventions = size(interv,1);
res.autonomy_pct = max(0, 1 - res.interventions*6/t)*100;      % Bojarski et al. 2016
res.cte_mean_abs = mean(cteAbs);
res.cte_rms = sqrt(mean(logT(:,6).^2));
res.cte_p95 = prctile_simple(cteAbs, 95);
res.cte_max = max(cteAbs);
res.steer_rate_rms_deg_s = rad2deg(sqrt(mean((diff(logT(:,5))/cfg.dt).^2)));
res.forks_total = numel(route.forks);
res.forks_correct = sum(forkRes == 1);
disp(res);

if doPlot
    figure('Name', ['closed loop: ' policy], 'Color', 'w', 'Position', [80 80 1300 480]);
    subplot(1,2,1);
    image('XData', [cfg.mapX0 cfg.mapX0+cfg.mapWm], 'YData', [cfg.mapY0 cfg.mapY0+cfg.mapHm], 'CData', tex);
    axis xy equal tight; hold on;
    plot(route.pts(:,1), route.pts(:,2), 'w--', 'LineWidth', 0.7);
    plot(logT(:,2), logT(:,3), 'r-', 'LineWidth', 1.0);
    if ~isempty(interv), plot(interv(:,2), interv(:,3), 'yx', 'MarkerSize', 10, 'LineWidth', 2); end
    title(sprintf('%s: autonomy %.1f %%, forks %d/%d', policy, res.autonomy_pct, res.forks_correct, res.forks_total));
    xlabel('x [m]'); ylabel('y [m]'); legend({'A* route', 'trajectory'}, 'Location', 'southeast');
    subplot(1,2,2);
    plot(logT(:,9), logT(:,6)); hold on; grid on;
    sl = [min(logT(:,9)) max(logT(:,9))];                 % yline() is not in Octave
    plot(sl, [1 1]*cfg.trackWidth/2, 'r:'); plot(sl, -[1 1]*cfg.trackWidth/2, 'r:');
    xlabel('route distance s [m]'); ylabel('cross-track error [m]');
    title(sprintf('|CTE| mean %.2f m, p95 %.2f m', res.cte_mean_abs, res.cte_p95));
end
end

function v = getopt(s, name, default)
if isfield(s, name), v = s.(name); else, v = default; end
end

function p = prctile_simple(x, q)
% percentile without the Statistics Toolbox (linear interpolation, same as numpy default)
x = sort(x(:)); n = numel(x);
r = (q/100)*(n - 1) + 1;
lo = floor(r); hi = min(lo + 1, n);
p = x(lo) + (r - lo)*(x(hi) - x(lo));
end

function out = imresize_nn(img, k)
% nearest-neighbour upscale without the Image Processing Toolbox
out = img(ceil((1:size(img,1)*k)/k), ceil((1:size(img,2)*k)/k), :);
end

% =====================================================================
% CNN: cnn_load, cnn_predict, pilotnet_native_load, pilotnet_native_forward
% =====================================================================
function net = cnn_load(onnxPath, backend)
% Load the exported PilotNet model.
%   'native' (default): plain-MATLAB forward pass using pilotnet_weights.mat
%             written next to the .onnx by  pilotnet_all.py export. No toolbox, no Python.
%   'python': ONNX Runtime through MATLAB's Python interface. Needs pyenv
%             pointing at a Python with onnxruntime + numpy, e.g.
%             pyenv('Version','C:\Program Files\Python311\python.exe')
%   'matlab': Deep Learning Toolbox ONNX importer (importNetworkFromONNX, R2023b+,
%             or importONNXNetwork).
% The network takes a 66x200x3 RGB image (0..255) and returns one value per
% head: [FOLLOW LEFT RIGHT] for the conditional model.
if nargin < 2 || isempty(backend), backend = 'native'; end
if isempty(regexp(onnxPath, '^([A-Za-z]:|[\\/])', 'once'))      % make absolute for Python
    onnxPath = fullfile(pwd, onnxPath);
end
if ~exist(onnxPath, 'file'), error('ONNX model not found: %s', onnxPath); end
switch backend
    case 'native'
        matPath = fullfile(fileparts(onnxPath), 'pilotnet_weights.mat');
        net = pilotnet_native_load(matPath);
    case 'python'
        net.backend = backend;
        if exist('pyrun', 'builtin') || exist('pyrun', 'file')       % R2021b+: 1 thread, as in Python
            net.obj = pyrun(['import onnxruntime as ort' newline 'so = ort.SessionOptions()' newline ...
                'so.intra_op_num_threads = 1' newline ...
                's = ort.InferenceSession(p, so, providers=["CPUExecutionProvider"])'], 's', 'p', onnxPath);
        else                    % R2019b-R2021a: SessionOptions (pybind11) cannot be passed from MATLAB
            ort = py.importlib.import_module('onnxruntime');
            net.obj = ort.InferenceSession(onnxPath, pyargs('providers', py.list({'CPUExecutionProvider'})));
        end
        net.np = py.importlib.import_module('numpy');
    case 'matlab'
        net.backend = backend;
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

function y = cnn_predict(net, img)
% Run PilotNet on one 66x200x3 uint8 RGB image -> row vector of heads.
switch net.backend
    case 'native'
        y = pilotnet_native_forward(net, img);
    case 'python'
        % H x W x 3 (column-major) -> row-major (H,W,C) bytes -> float32 (1,3,H,W) for ONNX Runtime
        buf = reshape(permute(img, [3 2 1]), 1, []);
        np = net.np;
        a = np.frombuffer(py.bytes(buf), pyargs('dtype', 'uint8'));
        a = a.reshape(py.tuple({int32(size(img,1)), int32(size(img,2)), int32(3)}));
        a = np.expand_dims(a.astype('float32').transpose(py.tuple({int32(2), int32(0), int32(1)})), int32(0));
        r = cell(net.obj.run(py.None, py.dict(pyargs('image', a))));
        y = cellfun(@double, cell(r{1}.flatten().tolist()));
    case 'matlab'
        if strcmp(net.kind, 'dlnetwork')
            X = dlarray(single(img), 'SSCB');
            y = double(extractdata(predict(net.obj, X)));
        else
            y = double(predict(net.obj, single(img)));
        end
        y = reshape(y, 1, []);
end
end

function net = pilotnet_native_load(matPath)
% Load PilotNet weights (pilotnet_weights.mat) and pre-compute im2col gather
% indices, so the forward pass is a handful of matrix products in plain MATLAB.
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

function y = pilotnet_native_forward(net, img)
% PilotNet inference in plain MATLAB. img: 66x200x3 uint8 RGB. y: 1 x nBranches.
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
