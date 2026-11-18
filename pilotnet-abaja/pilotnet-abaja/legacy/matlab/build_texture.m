function [tex, D] = build_texture(cfg, course)
%BUILD_TEXTURE Procedural ground texture of the course (uint8 HxWx3).
%   Row r <-> y = mapY0 + (r-0.5)*res, column c <-> x = mapX0 + (c-0.5)*res.
%   Same hash noise and arithmetic as python/sim/course.py:build_texture.
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
