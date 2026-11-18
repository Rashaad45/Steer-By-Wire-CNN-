function edgePath = astar_graph(course, startNode, goalNode)
%ASTAR_GRAPH A* search on the directed road graph (Hart, Nilsson & Raphael 1968).
%   g(n) = travelled centreline length, h(n) = straight-line distance to the
%   goal (admissible: no edge is shorter than the chord between its nodes).
%   Returns the edge indices from startNode to goalNode (node indices).
nN = size(course.xyh,1);
g = inf(nN,1); g(startNode) = 0;
f = inf(nN,1); f(startNode) = heur(startNode);
cameFromNode = zeros(nN,1); cameFromEdge = zeros(nN,1);
open = false(nN,1); open(startNode) = true;
closed = false(nN,1);
while any(open)
    cand = find(open);
    [~,k] = min(f(cand));
    n = cand(k);
    if n == goalNode, break; end
    open(n) = false; closed(n) = true;
    for e = course.outEdges{n}
        m = course.edges(e).to;
        if closed(m), continue; end
        gNew = g(n) + course.edges(e).length;
        if gNew < g(m)
            g(m) = gNew; f(m) = gNew + heur(m);
            cameFromNode(m) = n; cameFromEdge(m) = e;
            open(m) = true;
        end
    end
end
if isinf(g(goalNode)), error('no route %d -> %d', startNode, goalNode); end
edgePath = [];
n = goalNode;
while n ~= startNode
    edgePath = [cameFromEdge(n), edgePath]; %#ok<AGROW>
    n = cameFromNode(n);
end

    function h = heur(n)
        h = hypot(course.xyh(goalNode,1) - course.xyh(n,1), course.xyh(goalNode,2) - course.xyh(n,2));
    end
end
