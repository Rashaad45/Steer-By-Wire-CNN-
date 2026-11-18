function ok = abaja_render_check(net)
%ABAJA_RENDER_CHECK Verify MATLAB reproduces the Python simulator exactly.
%   Renders the reference poses stored in reference/reference.json and
%   compares them with the PNGs rendered by Python. If a loaded network is
%   passed, also compares the CNN outputs with the Python ONNX outputs.
%   Expected: mean |diff| < 0.5 grey level, max |diff| <= a few levels
%   (floating-point rounding of the hash noise), CNN diff < 1e-3.
here = fileparts(mfilename('fullpath'));
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
