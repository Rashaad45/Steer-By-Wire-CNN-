function [res, logT] = run_closed_loop(policy, net, cycles, opts)
%RUN_CLOSED_LOOP Drive the aBAJA course with the CNN (or the expert) in the loop.
%   [res, log] = run_closed_loop('cnn', net, 2)
%   [res, log] = run_closed_loop('expert', [], 2)
%   opts (optional struct): latencySteps (1 = 50 ms), speed, startOffset,
%                           plot (true), video ('' or file.mp4)
%
%   Navigation stack per step:
%     mission checkpoints -> A* route on the road graph -> command
%     FOLLOW/LEFT/RIGHT from route progress -> CNN head selected by the
%     command -> curvature -> delta = atan(L*kappa) -> actuator -> bicycle.
%   The route is used to issue the command and to score the run; the CNN
%   steers from the camera image alone.
%   Intervention (NVIDIA protocol): vehicle centre more than half a track
%   width from the route -> counted, vehicle re-centred, 6 s penalty in the
%   autonomy metric.
if nargin < 3 || isempty(cycles), cycles = 2; end
if nargin < 4, opts = struct(); end
latencySteps = getopt(opts, 'latencySteps', 1);
cfg = abaja_config();
v = getopt(opts, 'speed', cfg.speed);
doPlot = getopt(opts, 'plot', true);
videoFile = getopt(opts, 'video', '');

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
    yline(cfg.trackWidth/2, 'r:'); yline(-cfg.trackWidth/2, 'r:');
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
