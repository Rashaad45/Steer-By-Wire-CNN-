function cfg = abaja_config()
%ABAJA_CONFIG Simulator constants. Must match python/sim/config.py exactly.
%   The course, texture and camera are deterministic, so MATLAB and Python
%   render the same image for the same pose (checked by abaja_render_check.m).

% ---------------- course (BAJA-style off-road loop with one fork) ---------
% node: name, x [m], y [m], heading [deg]
cfg.nodeNames = {'S','J1','A','B','J2'};
cfg.nodeXYH   = [ 60  12  180;
                  20  45   90;
                  55  83    0;
                  58  58    0;
                 108  50  -90];
% edges: {from, to, [intermediate control points]}
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
