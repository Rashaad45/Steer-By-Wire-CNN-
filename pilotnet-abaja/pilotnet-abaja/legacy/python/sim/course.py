"""Course geometry, navigation graph (A*), route/command generation and the
procedural ground texture. Mirrors matlab/build_course.m, astar_graph.m,
build_route.m and build_texture.m.
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field

import numpy as np

from . import config as C


# --------------------------------------------------------------------------
# Hermite-spline centrelines
# --------------------------------------------------------------------------
def hermite_chain(pts: np.ndarray, h0: float, h1: float, ds: float = C.PATH_DS,
                  k: int = 200) -> np.ndarray:
    """Cubic Hermite spline through pts (n x 2).

    End tangents follow the node headings h0, h1 [rad] (so every edge leaving
    or entering a node has the same heading there -> C1-continuous routes);
    interior tangents are Catmull-Rom. Result is resampled at uniform arc
    length ds and includes both end points.
    """
    pts = np.asarray(pts, float)
    n = len(pts)
    m = np.zeros_like(pts)
    for i in range(1, n - 1):
        m[i] = 0.5 * (pts[i + 1] - pts[i - 1])
    m[0] = np.array([math.cos(h0), math.sin(h0)]) * np.linalg.norm(pts[1] - pts[0])
    m[-1] = np.array([math.cos(h1), math.sin(h1)]) * np.linalg.norm(pts[-1] - pts[-2])

    t = np.linspace(0.0, 1.0, k + 1)[:-1]            # k samples, last excluded
    h00 = 2 * t**3 - 3 * t**2 + 1
    h10 = t**3 - 2 * t**2 + t
    h01 = -2 * t**3 + 3 * t**2
    h11 = t**3 - t**2
    dense = []
    for i in range(n - 1):
        seg = (np.outer(h00, pts[i]) + np.outer(h10, m[i]) +
               np.outer(h01, pts[i + 1]) + np.outer(h11, m[i + 1]))
        dense.append(seg)
    dense.append(pts[-1:])
    dense = np.vstack(dense)

    seg_len = np.hypot(*np.diff(dense, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(seg_len)])
    n_out = int(math.floor(s[-1] / ds))
    s_new = np.concatenate([np.arange(n_out + 1) * ds, [s[-1]]]) \
        if s[-1] - n_out * ds > 1e-9 else np.arange(n_out + 1) * ds
    x = np.interp(s_new, s, dense[:, 0])
    y = np.interp(s_new, s, dense[:, 1])
    return np.column_stack([x, y])


def polyline_s(p: np.ndarray) -> np.ndarray:
    return np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(p, axis=0).T))])


@dataclass
class Course:
    nodes: dict
    edges: list                       # list of dict(from,to,pts,length)
    out_edges: dict = field(default_factory=dict)

    def edge(self, a: str, b: str) -> dict:
        for e in self.edges:
            if e["from"] == a and e["to"] == b:
                return e
        raise KeyError((a, b))


def build_course() -> Course:
    nodes = {k: (v[0], v[1], math.radians(v[2])) for k, v in C.NODES.items()}
    edges = []
    for a, b, mid in C.EDGES:
        pa, pb = nodes[a], nodes[b]
        ctrl = np.array([pa[:2], *mid, pb[:2]], float)
        pts = hermite_chain(ctrl, pa[2], pb[2])
        edges.append(dict(**{"from": a, "to": b}, pts=pts, length=float(polyline_s(pts)[-1])))
    out = {}
    for i, e in enumerate(edges):
        out.setdefault(e["from"], []).append(i)
    return Course(nodes, edges, out)


# --------------------------------------------------------------------------
# Navigation: A* on the directed road graph
# --------------------------------------------------------------------------
def astar(course: Course, start: str, goal: str) -> list[int]:
    """Return list of edge indices from start to goal (A*, Euclidean heuristic)."""
    def h(n):
        x0, y0, _ = course.nodes[n]
        x1, y1, _ = course.nodes[goal]
        return math.hypot(x1 - x0, y1 - y0)

    g = {start: 0.0}
    came = {}
    pq = [(h(start), 0, start)]
    tie = 1
    closed = set()
    while pq:
        _, _, n = heapq.heappop(pq)
        if n == goal:
            break
        if n in closed:
            continue
        closed.add(n)
        for ei in course.out_edges.get(n, []):
            e = course.edges[ei]
            m = e["to"]
            cand = g[n] + e["length"]
            if cand < g.get(m, math.inf):
                g[m] = cand
                came[m] = (n, ei)
                heapq.heappush(pq, (cand + h(m), tie, m))
                tie += 1
    if goal not in g:
        raise RuntimeError(f"no route {start}->{goal}")
    path, n = [], goal
    while n != start:
        n, ei = came[n]
        path.append(ei)
    return path[::-1]


def fork_turn(course: Course, node: str, chosen_edge: int, probe: float = 10.0) -> int:
    """LEFT/RIGHT for the chosen outgoing edge of a fork node.

    Each outgoing edge is probed `probe` metres from the node; the one with the
    largest lateral offset to the left of the node heading is LEFT.
    """
    x0, y0, hd = course.nodes[node]
    off = {}
    for ei in course.out_edges[node]:
        p = course.edges[ei]["pts"]
        j = min(int(round(probe / C.PATH_DS)), len(p) - 1)
        dx, dy = p[j, 0] - x0, p[j, 1] - y0
        off[ei] = -dx * math.sin(hd) + dy * math.cos(hd)
    return C.LEFT if off[chosen_edge] == max(off.values()) else C.RIGHT


@dataclass
class Route:
    pts: np.ndarray          # (N,2)
    s: np.ndarray            # (N,)
    heading: np.ndarray      # (N,)
    cmd: np.ndarray          # (N,) int
    forks: list              # list of dict(s, cmd, edge, other_edges)


def build_route(course: Course, mission: list[str]) -> Route:
    """Concatenate A* legs between consecutive mission nodes and attach
    high-level navigation commands around every fork."""
    edge_seq = []
    for a, b in zip(mission[:-1], mission[1:]):
        edge_seq += astar(course, a, b)
    pts, forks, s_acc = [], [], 0.0
    for k, ei in enumerate(edge_seq):
        e = course.edges[ei]
        p = e["pts"] if k == 0 else e["pts"][1:]          # drop duplicate node
        if len(course.out_edges[e["from"]]) > 1:
            forks.append(dict(s=s_acc, edge=ei, node=e["from"],
                              cmd=fork_turn(course, e["from"], ei),
                              others=[o for o in course.out_edges[e["from"]] if o != ei]))
        pts.append(p)
        s_acc += e["length"]
    pts = np.vstack(pts)
    s = polyline_s(pts)   # equals s_acc at every node (segments are unchanged)
    d = np.diff(pts, axis=0)
    hd = np.arctan2(d[:, 1], d[:, 0])
    hd = np.concatenate([hd, hd[-1:]])
    cmd = np.full(len(pts), C.FOLLOW, int)
    for f in forks:
        m = (s >= f["s"] - C.CMD_BEFORE) & (s <= f["s"] + C.CMD_AFTER)
        cmd[m] = f["cmd"]
    return Route(pts, s, hd, cmd, forks)


def nearest_on_route(route: Route, x: float, y: float, hint: int,
                     back: int = 40, fwd: int = 240) -> int:
    lo, hi = max(0, hint - back), min(len(route.pts), hint + fwd)
    seg = route.pts[lo:hi]
    return lo + int(np.argmin((seg[:, 0] - x) ** 2 + (seg[:, 1] - y) ** 2))


def cross_track(route: Route, i: int, x: float, y: float) -> float:
    """Signed lateral error, + = vehicle left of the path."""
    px, py = route.pts[i]
    h = route.heading[i]
    return -(x - px) * math.sin(h) + (y - py) * math.cos(h)


# --------------------------------------------------------------------------
# Procedural texture (identical in MATLAB: same hash, same arithmetic)
# --------------------------------------------------------------------------
def _hash(i, j, seed):
    v = np.sin(i * 12.9898 + j * 78.233 + seed * 37.719) * 43758.5453
    return v - np.floor(v)


def value_noise(X, Y, scale, seed):
    u, v = X / scale, Y / scale
    i, j = np.floor(u), np.floor(v)
    fu, fv = u - i, v - j
    fu = fu * fu * (3 - 2 * fu)
    fv = fv * fv * (3 - 2 * fv)
    a = _hash(i, j, seed)
    b = _hash(i + 1, j, seed)
    c = _hash(i, j + 1, seed)
    d = _hash(i + 1, j + 1, seed)
    return (a * (1 - fu) + b * fu) * (1 - fv) + (c * (1 - fu) + d * fu) * fv


def distance_field(course: Course) -> np.ndarray:
    """Min distance [m] from each map cell centre to any centreline sample."""
    W = int(round(C.MAP_W_M / C.MAP_RES))
    H = int(round(C.MAP_H_M / C.MAP_RES))
    D = np.full((H, W), np.inf)
    R = C.TRACK_WIDTH / 2 + 1.0
    pts = np.vstack([e["pts"] for e in course.edges])
    for px, py in pts:
        c0 = max(0, int(math.floor((px - R - C.MAP_X0) / C.MAP_RES)))
        c1 = min(W - 1, int(math.ceil((px + R - C.MAP_X0) / C.MAP_RES)))
        r0 = max(0, int(math.floor((py - R - C.MAP_Y0) / C.MAP_RES)))
        r1 = min(H - 1, int(math.ceil((py + R - C.MAP_Y0) / C.MAP_RES)))
        xs = C.MAP_X0 + (np.arange(c0, c1 + 1) + 0.5) * C.MAP_RES
        ys = C.MAP_Y0 + (np.arange(r0, r1 + 1) + 0.5) * C.MAP_RES
        d = np.hypot(xs[None, :] - px, ys[:, None] - py)
        blk = D[r0:r1 + 1, c0:c1 + 1]
        np.minimum(blk, d, out=blk)
    return D


def build_texture(course: Course, seed_offset: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Return (tex uint8 HxWx3, D distance field). seed_offset != 0 gives an unseen
    ground appearance (same geometry) for domain-shift tests."""
    D = distance_field(course)
    H, W = D.shape
    X = C.MAP_X0 + (np.arange(W) + 0.5) * C.MAP_RES
    Y = C.MAP_Y0 + (np.arange(H) + 0.5) * C.MAP_RES
    X, Y = np.meshgrid(X, Y)
    n1 = 0.5 * value_noise(X, Y, 4.0, 1 + seed_offset) + 0.3 * value_noise(X, Y, 1.0, 2 + seed_offset) + 0.2 * value_noise(X, Y, 0.3, 3 + seed_offset)
    n2 = 0.5 * value_noise(X, Y, 4.0, 4 + seed_offset) + 0.3 * value_noise(X, Y, 1.0, 5 + seed_offset) + 0.2 * value_noise(X, Y, 0.3, 6 + seed_offset)
    bush = value_noise(X, Y, 6.0, 7 + seed_offset)

    dirt = np.array([150.0, 112.0, 78.0]) * (0.78 + 0.44 * n1)[..., None]
    rut = np.abs(D - 0.85) < 0.18
    dirt[rut] *= 0.80
    berm = (D >= C.TRACK_WIDTH / 2 - 0.35) & (D <= C.TRACK_WIDTH / 2)
    dirt[berm] *= 1.12
    grass = np.array([72.0, 118.0, 52.0]) * (0.70 + 0.60 * n2)[..., None]
    grass[bush > 0.72] *= 0.55
    track = D <= C.TRACK_WIDTH / 2
    tex = np.where(track[..., None], dirt, grass)
    tex = np.clip(np.floor(tex + 0.5), 0, 255).astype(np.uint8)
    return tex, D
