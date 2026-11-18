function course = build_course(cfg)
%BUILD_COURSE Road graph of the aBAJA course: nodes + Hermite centreline edges.
nN = numel(cfg.nodeNames);
course.names = cfg.nodeNames;
course.xyh = [cfg.nodeXYH(:,1:2), deg2rad(cfg.nodeXYH(:,3))];
nE = size(cfg.edges,1);
course.edges = struct('from',cell(nE,1),'to',[],'pts',[],'length',[]);
course.outEdges = cell(nN,1);
for e = 1:nE
    a = node_index(course, cfg.edges{e,1});
    b = node_index(course, cfg.edges{e,2});
    ctrl = [course.xyh(a,1:2); cfg.edges{e,3}; course.xyh(b,1:2)];
    P = hermite_chain(ctrl, course.xyh(a,3), course.xyh(b,3), cfg.pathDs);
    course.edges(e).from = a;
    course.edges(e).to = b;
    course.edges(e).pts = P;
    course.edges(e).length = sum(hypot(diff(P(:,1)), diff(P(:,2))));
    course.outEdges{a}(end+1) = e;
end
end

function i = node_index(course, name)
i = find(strcmp(course.names, name), 1);
if isempty(i), error('unknown node %s', name); end
end
