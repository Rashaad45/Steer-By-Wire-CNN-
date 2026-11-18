"""Generate a labelled camera dataset in the aBAJA simulator (Python reference
of matlab/generate_dataset.m).

The expert is pure pursuit on the A*-planned route. Two mechanisms produce the
off-centre "recovery" data a behaviour-cloned policy needs (Pomerleau 1989;
Bojarski 2016; Codevilla 2018):
  1. Noise injection: an Ornstein-Uhlenbeck perturbation is added to the
     *executed* command for part of the drive, while the *recorded* label is
     the expert's clean command at the reached pose.
  2. Pose perturbation: for each recorded frame an extra image is rendered
     from a laterally shifted / rotated pose and labelled with the expert's
     command at that pose (the simulator equivalent of NVIDIA's viewpoint
     transformation, but exact because the scene is re-rendered).

Usage:
    python -m sim.gen_dataset --out data/sim --cycles 6 --seed 0
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os

import cv2
import numpy as np

from . import config as C
from .course import build_course, build_route, build_texture
from .render import camera_precompute, render
from .vehicle import bicycle_step, kappa_to_delta, pure_pursuit


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/sim")
    ap.add_argument("--cycles", type=int, default=6, help="each cycle = left lap + right lap")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--record-every", type=int, default=2, help="20 Hz sim -> 10 Hz frames")
    ap.add_argument("--noise-sigma", type=float, default=0.06, help="OU noise on kappa [1/m]")
    ap.add_argument("--max-offset", type=float, default=1.8, help="pose perturbation [m]")
    ap.add_argument("--max-yaw-deg", type=float, default=20.0)
    ap.add_argument("--speed-jitter", type=float, default=1.0, help="+- m/s per lap")
    ap.add_argument("--fork-boost", type=int, default=3,
                    help="extra perturbed samples per frame while a LEFT/RIGHT command is active")
    a = ap.parse_args(argv)

    rng = np.random.default_rng(a.seed)
    os.makedirs(os.path.join(a.out, "images"), exist_ok=True)
    course = build_course()
    tex, _ = build_texture(course)
    cam = camera_precompute()
    mission = ["S"] + ["A", "S", "B", "S"] * a.cycles
    route = build_route(course, mission)

    state = [route.pts[0, 0], route.pts[0, 1], route.heading[0], 0.0]
    hint, n_img, ou, step = 0, 0, 0.0, 0
    noise_on, noise_timer = False, 0.0
    rows = []
    lap_len = route.s[-1] / (2 * a.cycles)
    v = C.SPEED

    def save(img, kappa, cmd, cte, x, y, psi, aug):
        nonlocal n_img
        fn = f"{n_img:06d}.png"
        cv2.imwrite(os.path.join(a.out, "images", fn), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        rows.append([fn, kappa / C.KAPPA_MAX, kappa, int(cmd), v, cte, x, y, psi, aug,
                     int(route.s[hint] // lap_len)])
        n_img += 1

    while hint < len(route.pts) - 5:
        x, y, psi, _ = state
        kappa, hint, cte = pure_pursuit(route, x, y, psi, v, hint)
        lap = int(route.s[hint] // lap_len)
        if step % int(round(lap_len / v / C.DT)) == 0:      # new speed each lap
            v = C.SPEED + rng.uniform(-a.speed_jitter, a.speed_jitter)

        # noise schedule: alternate ~2-6 s windows with / without noise
        noise_timer -= C.DT
        if noise_timer <= 0:
            noise_on = not noise_on
            noise_timer = rng.uniform(2.0, 6.0)
        theta = 1.5
        ou += -theta * ou * C.DT + a.noise_sigma * math.sqrt(2 * theta * C.DT) * rng.standard_normal()
        k_exec = kappa + (ou if noise_on else 0.0)

        cmd = route.cmd[hint]
        if step % a.record_every == 0 or cmd != C.FOLLOW:     # 20 Hz near forks
            save(render(cam, tex, x, y, psi), kappa, cmd, cte, x, y, psi, 0)
            n_pert = 1 + (a.fork_boost if cmd != C.FOLLOW else 0)
            for _ in range(n_pert):                            # pose perturbations
                dl = rng.uniform(-a.max_offset, a.max_offset)
                dpsi = math.radians(float(np.clip(rng.normal(0, a.max_yaw_deg / 2),
                                                  -a.max_yaw_deg, a.max_yaw_deg)))
                xp, yp, pp = x - dl * math.sin(psi), y + dl * math.cos(psi), psi + dpsi
                kp, ip, ctep = pure_pursuit(route, xp, yp, pp, v, hint)
                if abs(ctep) < C.TRACK_WIDTH / 2 - 0.3:
                    save(render(cam, tex, xp, yp, pp), kp, route.cmd[ip], ctep, xp, yp, pp, 1)

        state = bicycle_step(state, kappa_to_delta(k_exec), v)
        step += 1
        if step % 2000 == 0:
            print(f"  step {step}  s={route.s[hint]:.0f}/{route.s[-1]:.0f} m  images={n_img}  lap={lap}")

    with open(os.path.join(a.out, "labels.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "target", "kappa", "command", "speed", "cte", "x", "y", "psi", "aug", "lap"])
        w.writerows(rows)
    meta = dict(source="python-sim", kappa_max=C.KAPPA_MAX, wheelbase=C.WHEELBASE,
                target="kappa / kappa_max, + = left", commands=["FOLLOW", "LEFT", "RIGHT"],
                img_hw=[C.IMG_H, C.IMG_W], n_images=n_img, cycles=a.cycles, seed=a.seed)
    with open(os.path.join(a.out, "dataset_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    tgt = np.array([r[1] for r in rows])
    cmds = np.array([r[3] for r in rows])
    print(f"done: {n_img} images -> {a.out}; target mean={tgt.mean():.3f} std={tgt.std():.3f}; "
          f"commands F/L/R = {np.bincount(cmds, minlength=3).tolist()}")


if __name__ == "__main__":
    main()
