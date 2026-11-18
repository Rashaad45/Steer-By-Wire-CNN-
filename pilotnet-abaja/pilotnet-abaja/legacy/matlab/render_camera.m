function img = render_camera(cfg, cam, tex, x, y, psi)
%RENDER_CAMERA 66x200x3 uint8 RGB view from rear-axle pose (x, y, psi).
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
