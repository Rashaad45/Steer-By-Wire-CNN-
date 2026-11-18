"""Write reference renders (and optional CNN outputs) that MATLAB checks against
in matlab/abaja_render_check.m.

python make_reference.py --onnx runs/sim/pilotnet.onnx
"""
from __future__ import annotations

import argparse
import json
import os

import cv2
import numpy as np

from sim.course import build_course, build_route, build_texture
from sim.render import camera_precompute, render


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default=None)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "matlab", "reference"))
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    course = build_course()
    tex, _ = build_texture(course)
    cam = camera_precompute()
    route = build_route(course, ["S", "A", "S", "B", "S"])
    net = None
    if a.onnx:
        from pilotnet_infer import PilotNetONNX
        net = PilotNetONNX(a.onnx)
    poses = []
    fork_s = route.forks[0]["s"]
    picks = [10.0, fork_s - 8.0, fork_s + 20.0, 250.0, 420.0]
    for k, s in enumerate(picks):
        i = int(np.searchsorted(route.s, s))
        x, y, psi = float(route.pts[i, 0]), float(route.pts[i, 1]), float(route.heading[i])
        if k == 3:                         # an off-centre, rotated pose
            x, y, psi = x - 1.2 * np.sin(psi), y + 1.2 * np.cos(psi), psi + 0.2
        img = render(cam, tex, x, y, psi)
        fn = f"ref_{k}.png"
        cv2.imwrite(os.path.join(a.out, fn), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        p = dict(file=fn, x=x, y=y, psi=psi)
        if net is not None:
            p["cnn"] = [float(v) for v in net.predict(img)]
        poses.append(p)
    with open(os.path.join(a.out, "reference.json"), "w") as f:
        json.dump(dict(poses=poses, route_length=float(route.s[-1])), f, indent=2)
    print(f"wrote {len(poses)} reference poses to {os.path.abspath(a.out)}")


if __name__ == "__main__":
    main()
