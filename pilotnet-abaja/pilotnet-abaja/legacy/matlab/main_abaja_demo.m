%% End-to-end CNN steering for an aBAJA autonomous vehicle (PilotNet)
% MATLAB navigation + closed-loop simulation driver.
%
% Pipeline
%   1. Build the course road graph, A* route and command schedule
%   2. Check MATLAB renders exactly what Python trained on
%   3. Pure-pursuit expert run (upper bound with perfect localisation)
%   4. CNN-in-the-loop run (camera -> PilotNet -> steering), scored with the
%      NVIDIA autonomy metric, cross-track error and fork-command compliance
%
% Before step 4, train and export the model in Python:
%   python -m sim.gen_dataset --out ../data/sim         (or generate_dataset.m)
%   python train.py --dataset sim --root ../data/sim --branches 3 --balance --out runs/sim
%   python export_onnx.py --ckpt runs/sim/best.pt
% (BACKEND 'python' also needs: pyenv('Version', '<path to that python>'))
clear; clc; close all;
cd(fileparts(mfilename('fullpath')));

ONNX_PATH = fullfile('..', 'python', 'runs', 'sim', 'pilotnet.onnx');   % <- edit if needed
BACKEND   = 'native';      % 'native' (plain MATLAB, default) | 'python' (ONNX Runtime via pyenv) | 'matlab' (DL Toolbox)
CYCLES    = 2;             % each cycle = one lap on the LEFT branch + one on the RIGHT
GENERATE_MATLAB_DATASET = false;

%% 1. Course, navigation graph, route
cfg = abaja_config();
course = build_course(cfg);
tex = build_texture(cfg, course);
route = build_route(cfg, course, {'S','A','S','B','S'});
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
fprintf('Route %.1f m, forks at s = %s m\n', route.s(end), mat2str(round([route.forks.s],1)));

%% 2. Renderer / model equivalence with Python
net = [];
if exist(ONNX_PATH, 'file')
    net = cnn_load(ONNX_PATH, BACKEND);
end
if exist(fullfile('reference','reference.json'), 'file')
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
    T = struct2table([resExpert; resCNN; resCNNfast]);
    disp(T(:, {'policy','interventions','autonomy_pct','cte_mean_abs','cte_p95','cte_max', ...
               'steer_rate_rms_deg_s','forks_correct','forks_total'}));
else
    warning('No ONNX model at %s - train/export in Python first (see header).', ONNX_PATH);
end

%% 5. (optional) generate a training set from MATLAB instead of Python
if GENERATE_MATLAB_DATASET
    generate_dataset(fullfile('..','data','sim_matlab'), 6, 0);
end
