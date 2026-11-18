function cam = camera_precompute(cfg)
%CAMERA_PRECOMPUTE Intersect every (super-sampled) pixel ray with the ground.
%   Pinhole camera at height h, pitched down by theta, CAM_X ahead of the
%   rear axle. For pixel (u,v): x_c=(u-cx)/f (right), y_c=(v-cy)/f (down).
%   Ray in vehicle frame (forward, left, up):
%       d = [cos(th) - y_c sin(th),  -x_c,  -sin(th) - y_c cos(th)]
%   Ground hit (z = 0) at t = h / (-d_up) when d_up < 0.
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
