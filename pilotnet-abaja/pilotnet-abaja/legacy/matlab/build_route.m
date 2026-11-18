function route = build_route(cfg, course, mission)
%BUILD_ROUTE Navigation layer: A* legs between mission checkpoints, joined
%into one route polyline with a high-level command at every point
%(FOLLOW / LEFT / RIGHT around forks, Codevilla et al. 2018).
%   mission: cell array of node names, e.g. {'S','A','S','B','S'}.
edgeSeq = [];
for k = 1:numel(mission)-1
    a = find(strcmp(course.names, mission{k}), 1);
    b = find(strcmp(course.names, mission{k+1}), 1);
    edgeSeq = [edgeSeq, astar_graph(course, a, b)]; %#ok<AGROW>
end
pts = zeros(0,2); sAcc = 0;
forks = struct('s',{},'edge',{},'node',{},'cmd',{},'others',{});
for k = 1:numel(edgeSeq)
    e = edgeSeq(k);
    E = course.edges(e);
    if k == 1, P = E.pts; else, P = E.pts(2:end,:); end      % drop duplicate node
    outs = course.outEdges{E.from};
    if numel(outs) > 1
        f.s = sAcc; f.edge = e; f.node = E.from;
        f.cmd = fork_turn(cfg, course, E.from, e);
        f.others = outs(outs ~= e);
        forks(end+1) = f; %#ok<AGROW>
    end
    pts = [pts; P]; %#ok<AGROW>
    sAcc = sAcc + E.length;
end
route.pts = pts;
route.s = [0; cumsum(hypot(diff(pts(:,1)), diff(pts(:,2))))];
d = diff(pts);
hd = atan2(d(:,2), d(:,1));
route.heading = [hd; hd(end)];
route.cmd = repmat(cfg.FOLLOW, size(pts,1), 1);
for k = 1:numel(forks)
    m = route.s >= forks(k).s - cfg.cmdBefore & route.s <= forks(k).s + cfg.cmdAfter;
    route.cmd(m) = forks(k).cmd;
end
route.forks = forks;
end

function c = fork_turn(cfg, course, node, chosen)
% LEFT if the chosen edge, probed 10 m after the node, lies furthest to the
% left of the node heading; otherwise RIGHT.
probe = 10;
x0 = course.xyh(node,1); y0 = course.xyh(node,2); hd = course.xyh(node,3);
outs = course.outEdges{node};
off = zeros(size(outs));
for i = 1:numel(outs)
    P = course.edges(outs(i)).pts;
    j = min(round(probe/cfg.pathDs), size(P,1)-1) + 1;
    off(i) = -(P(j,1)-x0)*sin(hd) + (P(j,2)-y0)*cos(hd);
end
if off(outs == chosen) == max(off), c = cfg.LEFT; else, c = cfg.RIGHT; end
end
