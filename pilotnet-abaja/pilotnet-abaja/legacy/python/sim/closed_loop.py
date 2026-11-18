"""Closed-loop evaluation in the aBAJA simulator (Python reference of
matlab/run_closed_loop.m).

Navigation stack:  mission checkpoints -> A* route on the road graph ->
high-level command (FOLLOW/LEFT/RIGHT) from route progress -> the CNN
(selected head) outputs curvature -> steering actuator -> kinematic bicycle.
The route is used only to issue commands and to score the run; the CNN steers
from pixels alone.

python -m sim.closed_loop --onnx runs/sim/pilotnet.onnx --cycles 2 --out runs/sim/closed_loop
python -m sim.closed_loop --policy expert ...      # pure-pursuit upper bound
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os

import numpy as np

from . import config as C
from .course import build_course, build_route, build_texture, cross_track, nearest_on_route
from .render import camera_precompute, render
from .vehicle import bicycle_step, kappa_to_delta, pure_pursuit


def point_to_polyline(p, x, y):
    return float(np.min(np.hypot(p[:, 0] - x, p[:, 1] - y)))


def run(policy, cycles=2, latency_steps=1, speed=C.SPEED, seed=0, start_offset=0.0,
        texture_seed=0, gain=1.0):
    course = build_course()
    tex, _ = build_texture(course, texture_seed)
    cam = camera_precompute()
    route = build_route(course, ["S"] + ["A", "S", "B", "S"] * cycles)
    st = [route.pts[0, 0], route.pts[0, 1] + start_offset, route.heading[0], 0.0]
    hint, t, step = 0, 0.0, 0
    queue = [0.0] * latency_steps            # compute latency (commands applied late)
    log, interventions = [], []
    fork_res = [None] * len(route.forks)
    v = speed
    while hint < len(route.pts) - 5:
        x, y, psi, d = st
        hint = nearest_on_route(route, x, y, hint)
        cte = cross_track(route, hint, x, y)
        cmd = int(route.cmd[hint])
        # ---- fork bookkeeping (did the vehicle take the commanded branch?)
        for k, f in enumerate(route.forks):
            if fork_res[k] is None and route.s[hint] >= f["s"] + 15.0:
                d_ok = point_to_polyline(course.edges[f["edge"]]["pts"], x, y)
                d_bad = min(point_to_polyline(course.edges[o]["pts"], x, y) for o in f["others"])
                fork_res[k] = bool(d_ok <= d_bad)
        # ---- intervention: vehicle centre left the track
        if abs(cte) > C.TRACK_WIDTH / 2:
            interventions.append(dict(t=t, s=float(route.s[hint]), x=x, y=y))
            for k, f in enumerate(route.forks):
                if fork_res[k] is None and f["s"] - 5 <= route.s[hint] <= f["s"] + 25:
                    fork_res[k] = False
            st = [route.pts[hint, 0], route.pts[hint, 1], route.heading[hint], 0.0]
            queue = [0.0] * latency_steps
            continue
        # ---- policy
        if policy == "expert":
            kappa, _, _ = pure_pursuit(route, x, y, psi, v, hint)
            heads = None
        else:
            img = render(cam, tex, x, y, psi)
            if gain != 1.0:                      # lighting shift (darker / brighter)
                img = np.clip(img * gain, 0, 255).astype(np.uint8)
            heads = policy(img)
            head = heads[cmd] if len(heads) > 1 else heads[0]
            kappa = float(np.clip(head, -1, 1)) * C.KAPPA_MAX
        queue.append(kappa_to_delta(kappa))
        st = bicycle_step(st, queue.pop(0), v)
        log.append([t, x, y, psi, d, cte, cmd, kappa, float(route.s[hint])])
        t += C.DT
        step += 1
    log = np.array(log)
    cte_abs = np.abs(log[:, 5])
    from pilotnet.metrics import autonomy, smoothness
    res = dict(
        distance_m=float(route.s[-1]), elapsed_s=float(t), interventions=len(interventions),
        autonomy_pct=autonomy(len(interventions), t),
        cte_mean_abs=float(cte_abs.mean()), cte_rms=float(np.sqrt(np.mean(log[:, 5] ** 2))),
        cte_p95=float(np.percentile(cte_abs, 95)), cte_max=float(cte_abs.max()),
        steer_rate_rms_deg_s=math.degrees(smoothness(log[:, 4], C.DT)),
        forks_total=len(route.forks), forks_correct=int(sum(bool(r) for r in fork_res)),
        latency_steps=latency_steps, speed=speed)
    return res, log, interventions, route, course, tex


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", choices=["cnn", "expert"], default="cnn")
    ap.add_argument("--onnx", default=None)
    ap.add_argument("--cycles", type=int, default=2)
    ap.add_argument("--latency-steps", type=int, default=1, help="1 step = 50 ms")
    ap.add_argument("--speed", type=float, default=C.SPEED)
    ap.add_argument("--start-offset", type=float, default=0.0, help="initial lateral offset [m]")
    ap.add_argument("--texture-seed", type=int, default=0, help="!=0: unseen ground texture")
    ap.add_argument("--gain", type=float, default=1.0, help="image brightness gain (lighting shift)")
    ap.add_argument("--tag", default="", help="suffix for output files")
    ap.add_argument("--out", default="runs/closed_loop")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)

    if a.policy == "cnn":
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from pilotnet_infer import PilotNetONNX
        net = PilotNetONNX(a.onnx)
        pol = net.predict
    else:
        pol = "expert"
    res, log, iv, route, course, tex = run(pol, a.cycles, a.latency_steps, a.speed,
                                           start_offset=a.start_offset, texture_seed=a.texture_seed,
                                           gain=a.gain)
    res.update(policy=a.policy, texture_seed=a.texture_seed, gain=a.gain)
    tag = a.policy + (f"_{a.tag}" if a.tag else "")
    print(json.dumps(res, indent=2))
    with open(os.path.join(a.out, f"metrics_{tag}.json"), "w") as f:
        json.dump(res, f, indent=2)
    with open(os.path.join(a.out, f"trajectory_{tag}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "x", "y", "psi", "delta", "cte", "cmd", "kappa_cmd", "s"])
        w.writerows(log.tolist())

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw=dict(width_ratios=[1.1, 1.4]))
    ax[0].imshow(tex, origin="lower", extent=[C.MAP_X0, C.MAP_X0 + C.MAP_W_M, C.MAP_Y0, C.MAP_Y0 + C.MAP_H_M])
    ax[0].plot(route.pts[:, 0], route.pts[:, 1], "w--", lw=.7, label="A* route")
    ax[0].plot(log[:, 1], log[:, 2], "r-", lw=.9, label=f"{a.policy} trajectory")
    for e in iv:
        ax[0].plot(e["x"], e["y"], "yx", ms=9, mew=2)
    ax[0].legend(loc="lower right", fontsize=8)
    ax[0].set_title(f"autonomy {res['autonomy_pct']:.1f} %  |  forks {res['forks_correct']}/{res['forks_total']}")
    ax[1].plot(log[:, 8], log[:, 5], lw=.8)
    ax[1].axhline(C.TRACK_WIDTH / 2, color="r", ls=":")
    ax[1].axhline(-C.TRACK_WIDTH / 2, color="r", ls=":")
    ax[1].set_xlabel("route distance s [m]")
    ax[1].set_ylabel("cross-track error [m]")
    ax[1].set_title(f"|CTE| mean {res['cte_mean_abs']:.2f} m, p95 {res['cte_p95']:.2f} m")
    plt.tight_layout()
    plt.savefig(os.path.join(a.out, f"closed_loop_{tag}.png"), dpi=130)


if __name__ == "__main__":
    main()
