function generate_dataset(outDir, cycles, seed)
%GENERATE_DATASET Expert demonstrations with recovery data (MATLAB version).
%   generate_dataset('../data/sim_matlab', 6, 0)
%   Writes outDir/images/*.png and outDir/labels.csv in the format read by
%   python/train.py --dataset sim. Same algorithm as python/sim/gen_dataset.py
%   (random streams differ, the course/renderer/labels do not).
%   Labels: target = kappa_expert / kappa_max  (+ = left),
%           command = 0 FOLLOW / 1 LEFT / 2 RIGHT.
if nargin < 1, outDir = fullfile('..','data','sim_matlab'); end
if nargin < 2, cycles = 6; end
if nargin < 3, seed = 0; end
recordEvery = 2; noiseSigma = 0.06; maxOffset = 1.8; maxYaw = 20; speedJitter = 1.0; forkBoost = 3;

rng(seed);
cfg = abaja_config();
course = build_course(cfg);
tex = build_texture(cfg, course);
cam = camera_precompute(cfg);
mission = [{'S'}, repmat({'A','S','B','S'}, 1, cycles)];
route = build_route(cfg, course, mission);
if ~exist(fullfile(outDir,'images'), 'dir'), mkdir(fullfile(outDir,'images')); end

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
        save_sample(render_camera(cfg, cam, tex, st(1), st(2), st(3)), kappa, cmd, cte, st(1), st(2), st(3), 0);
        nPert = 1 + forkBoost*(cmd ~= cfg.FOLLOW);
        for p = 1:nPert
            dl = (2*rand - 1)*maxOffset;
            dpsi = deg2rad(min(max(randn*maxYaw/2, -maxYaw), maxYaw));
            xp = st(1) - dl*sin(st(3)); yp = st(2) + dl*cos(st(3)); pp = st(3) + dpsi;
            [kp, ip, ctep] = pure_pursuit(cfg, route, xp, yp, pp, v, hint);
            if abs(ctep) < cfg.trackWidth/2 - 0.3
                save_sample(render_camera(cfg, cam, tex, xp, yp, pp), kp, route.cmd(ip), ctep, xp, yp, pp, 1);
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

    function save_sample(img, k, c, e, x, y, psi, aug)
        fn = sprintf('%06d.png', nImg);
        imwrite(img, fullfile(outDir, 'images', fn));
        nImg = nImg + 1;
        names{nImg} = fn; %#ok<AGROW>
        rows(nImg,:) = [k/cfg.kappaMax, k, c, v, e, x, y, psi, aug, floor(route.s(hint)/lapLen)];
    end
end
