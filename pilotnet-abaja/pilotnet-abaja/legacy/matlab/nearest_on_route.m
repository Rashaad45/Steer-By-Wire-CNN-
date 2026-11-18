function i = nearest_on_route(route, x, y, hint)
%NEAREST_ON_ROUTE Closest route index searched in a window around hint.
%   The window (40 back, 240 ahead = 10 m / 60 m) keeps the projection on the
%   planned branch where two branches run close together.
N = size(route.pts,1);
lo = max(1, hint - 40);
hi = min(N, hint + 239);
seg = route.pts(lo:hi,:);
[~, k] = min((seg(:,1) - x).^2 + (seg(:,2) - y).^2);
i = lo + k - 1;
end
