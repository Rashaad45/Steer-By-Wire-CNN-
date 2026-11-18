function e = cross_track(route, i, x, y)
%CROSS_TRACK Signed lateral error to the route at index i (+ = left of path).
h = route.heading(i);
e = -(x - route.pts(i,1))*sin(h) + (y - route.pts(i,2))*cos(h);
end
