"""Pinhole ground-plane camera renderer (flat-world ray casting).

Mirrors matlab/camera_precompute.m and render_camera.m. Each pixel ray is
intersected with the ground plane z = 0 once (the camera is rigid), giving a
fixed offset (pf, pl) in the vehicle frame. Per frame we only rotate/translate
these offsets into the world and look up the texture -> ~1 ms per frame.
"""
from __future__ import annotations

import math

import numpy as np

from . import config as C

SKY = np.array([175.0, 200.0, 230.0])
HAZE = np.array([190.0, 200.0, 210.0])


def camera_precompute():
    ss = C.SUPERSAMPLE
    W, H = C.IMG_W * ss, C.IMG_H * ss
    f = (C.IMG_W / 2) / math.tan(C.CAM_HFOV / 2)            # focal [final px]
    u = (np.arange(W) + 0.5) / ss                             # final-px coords
    v = (np.arange(H) + 0.5) / ss
    U, V = np.meshgrid(u, v)
    xc = (U - C.IMG_W / 2) / f                                # right
    yc = (V - C.IMG_H / 2) / f                                # down
    th = C.CAM_PITCH
    d_f = math.cos(th) - yc * math.sin(th)                    # forward
    d_l = -xc                                                 # left
    d_u = -math.sin(th) - yc * math.cos(th)                   # up
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(d_u < 0, C.CAM_HEIGHT / -d_u, np.inf)
    dist = t * np.hypot(d_f, d_l)
    valid = np.isfinite(t) & (dist <= C.MAX_RANGE)
    pf = np.where(valid, C.CAM_X + t * d_f, 0.0)
    pl = np.where(valid, t * d_l, 0.0)
    haze = np.where(valid, 1.0 - np.exp(-dist / C.HAZE_DIST), 1.0)
    return dict(pf=pf, pl=pl, valid=valid, haze=haze, ss=ss)


def render(cam: dict, tex: np.ndarray, x: float, y: float, psi: float) -> np.ndarray:
    """Return a (66, 200, 3) uint8 RGB image from rear-axle pose (x, y, psi)."""
    c, s = math.cos(psi), math.sin(psi)
    X = x + cam["pf"] * c - cam["pl"] * s
    Y = y + cam["pf"] * s + cam["pl"] * c
    Hm, Wm = tex.shape[:2]
    col = np.clip(np.floor((X - C.MAP_X0) / C.MAP_RES).astype(int), 0, Wm - 1)
    row = np.clip(np.floor((Y - C.MAP_Y0) / C.MAP_RES).astype(int), 0, Hm - 1)
    g = tex[row, col].astype(float)
    w = cam["haze"][..., None]
    img = g * (1 - w) + HAZE * w
    img[~cam["valid"]] = SKY
    ss = cam["ss"]
    img = img.reshape(C.IMG_H, ss, C.IMG_W, ss, 3).mean(axis=(1, 3))
    return np.clip(np.floor(img + 0.5), 0, 255).astype(np.uint8)
