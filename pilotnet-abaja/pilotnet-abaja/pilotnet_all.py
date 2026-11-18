#!/usr/bin/env python
"""pilotnet_all.py - end-to-end CNN steering for an aBAJA autonomous vehicle, in ONE file.

PilotNet (Bojarski et al., 2016) with optional command-conditioned heads
FOLLOW/LEFT/RIGHT (Codevilla et al., 2018), dataset readers, augmentation,
metrics, VisualBackProp, the aBAJA simulator (course, A*, texture, renderer,
kinematic bicycle, pure pursuit), dataset generator, training, evaluation,
ONNX + .mat export, MATLAB reference writer, closed-loop simulation, ONNX
inference class, unit tests, dataset download and conversion.

This file is a verbatim merge of the former python/ package (now in legacy/):
the maths, constants, rendering, labels and splits are unchanged.

Usage (PowerShell, from anywhere; relative paths are resolved from the folder
holding this file):
    python pilotnet_all.py selftest
    python pilotnet_all.py download   --what sully --dest datasets
    python pilotnet_all.py gen        --out datasets/sim --cycles 6 --seed 0
    python pilotnet_all.py train      --dataset sim --root datasets/sim --branches 3 --balance --cache --out runs/sim
    python pilotnet_all.py evaluate   --ckpt runs/sim/best.pt --root datasets/sim
    python pilotnet_all.py export     --ckpt runs/sim/best.pt
    python pilotnet_all.py reference  --onnx runs/sim/pilotnet.onnx
    python pilotnet_all.py closedloop --onnx runs/sim/pilotnet.onnx --out runs/sim/results
    python pilotnet_all.py convert    --source tartandrive --src <path> --dst datasets/tartandrive_csv
Run any subcommand with -h for all flags.
"""
from __future__ import annotations

import argparse
import csv
import glob
import heapq
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
import warnings
import zipfile
from dataclasses import dataclass, field

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

ROOT = os.path.dirname(os.path.abspath(__file__))     # project root


def _abs(p):
    """Resolve a relative path against the project root (not the shell's cwd)."""
    if p is None or p == "" or os.path.isabs(p):
        return p
    return os.path.normpath(os.path.join(ROOT, p))


# ============================================================================
# 1. Simulator configuration  (was sim/config.py; mirrored in matlab/abaja_sim.m)
# ============================================================================
class C:
    """Shared simulator configuration.

    Every value here is duplicated in matlab/abaja_sim.m (abaja_config). Keep
    the two in sync: the renderer, course and texture are designed to produce
    identical images in Python and MATLAB.
    """
    # ---------------- course (BAJA-style off-road loop with one fork) ----------
    # node: name -> (x [m], y [m], heading [deg])
    NODES = {
        "S":  (60.0, 12.0, 180.0),
        "J1": (20.0, 45.0, 90.0),
        "A":  (55.0, 83.0, 0.0),
        "B":  (58.0, 58.0, 0.0),
        "J2": (108.0, 50.0, -90.0),
    }
    # directed edges: (from, to, [intermediate control points])
    EDGES = [
        ("S",  "J1", [(36.0, 13.0), (22.0, 26.0)]),
        ("J1", "A",  [(12.0, 62.0), (22.0, 80.0)]),          # left branch
        ("A",  "J2", [(88.0, 84.0), (105.0, 72.0)]),
        ("J1", "B",  [(26.0, 55.0), (40.0, 59.0)]),          # right branch
        ("B",  "J2", [(80.0, 57.0), (98.0, 62.0)]),
        ("J2", "S",  [(104.0, 28.0), (88.0, 14.0)]),
    ]
    TRACK_WIDTH = 6.0          # [m]
    PATH_DS = 0.25             # centreline resampling step [m]

    # ---------------- map raster -------------------------------------------------
    MAP_X0, MAP_Y0 = -15.0, -15.0
    MAP_W_M, MAP_H_M = 150.0, 120.0
    MAP_RES = 0.10             # [m / cell]

    # ---------------- vehicle (aBAJA-class ATV, kinematic bicycle) ---------------
    WHEELBASE = 1.40           # [m]
    MAX_STEER = math.radians(30.0)
    STEER_TAU = 0.15           # first-order steering actuator lag [s]
    STEER_RATE = math.radians(120.0)   # actuator slew limit [rad/s]
    KAPPA_MAX = math.tan(MAX_STEER) / WHEELBASE   # 0.412 1/m -> label normalisation
    SPEED = 5.0                # nominal speed [m/s]
    DT = 0.05                  # simulation step [s] (20 Hz)

    # ---------------- camera -----------------------------------------------------
    IMG_W, IMG_H = 200, 66     # PilotNet input
    SUPERSAMPLE = 2            # render at 2x then box-filter down (anti-aliasing)
    CAM_HFOV = math.radians(100.0)
    CAM_HEIGHT = 1.20          # above ground [m]
    CAM_PITCH = math.radians(15.0)   # pitched down
    CAM_X = 1.00               # ahead of rear axle [m]
    MAX_RANGE = 60.0           # rays beyond this are sky/haze [m]
    HAZE_DIST = 45.0           # exponential haze scale [m]

    # ---------------- expert / navigation ---------------------------------------
    LOOKAHEAD_MIN = 3.0
    LOOKAHEAD_GAIN = 0.4       # Ld = LOOKAHEAD_MIN + LOOKAHEAD_GAIN * v
    CMD_BEFORE = 18.0          # issue LEFT/RIGHT this many metres before the fork
    CMD_AFTER = 8.0            # ... and keep it this long after the fork
    # command indices
    FOLLOW, LEFT, RIGHT = 0, 1, 2


# ============================================================================
# 2. Model  (was pilotnet/model.py)
# ============================================================================
# PilotNet (Bojarski et al., 2016) with optional command-conditioned branches
# (Codevilla et al., 2018).
#
# Input  : RGB image, float32, range 0..255, shape (N, 3, 66, 200).
#          Colour conversion (RGB -> YUV) and normalisation are *inside* the
#          network, so the exported ONNX model takes raw pixels.
# Output : (N, n_branches) normalised steering targets. With n_branches == 1 the
#          network is exactly PilotNet. With n_branches == 3 the heads are
#          [FOLLOW, LEFT, RIGHT]; the navigation system picks one head at run time.
#
# Layer sizes (66x200 input):
#     conv1 5x5/2  24 -> 31x98
#     conv2 5x5/2  36 -> 14x47
#     conv3 5x5/2  48 ->  5x22
#     conv4 3x3/1  64 ->  3x20
#     conv5 3x3/1  64 ->  1x18   -> flatten 1152
#     fc 100 -> 50 -> 10 -> 1
#     ~252 k parameters for one branch, matching the "about 250 thousand
#     parameters" reported by NVIDIA. (The paper's figure labels the flatten
#     layer "1164"; 64*1*18 = 1152, and a 1164-unit dense layer would add
#     ~1.3 M parameters, contradicting the stated total, so we use 1152.)
COMMANDS = ("FOLLOW", "LEFT", "RIGHT")


class RGB2YUVNorm(nn.Module):
    """Fixed RGB[0,255] -> YUV (BT.601 analogue) -> roughly [-1, 1].

    Y = 0.299 R + 0.587 G + 0.114 B
    U = -0.14713 R - 0.28886 G + 0.436 B + 128
    V = 0.615 R - 0.51499 G - 0.10001 B + 128
    out = YUV / 127.5 - 1
    Implemented as a frozen 1x1 convolution so it exports cleanly to ONNX.
    """

    def __init__(self) -> None:
        super().__init__()
        m = torch.tensor([[0.299, 0.587, 0.114],
                          [-0.14713, -0.28886, 0.436],
                          [0.615, -0.51499, -0.10001]], dtype=torch.float32)
        b = torch.tensor([0.0, 128.0, 128.0], dtype=torch.float32)
        self.register_buffer("weight", m.view(3, 3, 1, 1))
        self.register_buffer("bias", b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.weight, self.bias) / 127.5 - 1.0


def _act(name: str) -> nn.Module:
    return {"elu": nn.ELU(), "relu": nn.ReLU(), "gelu": nn.GELU()}[name]


class PilotNet(nn.Module):
    def __init__(self, n_branches: int = 1, dropout: float = 0.2,
                 act: str = "elu", input_hw: tuple[int, int] = (66, 200),
                 batchnorm: bool = False) -> None:
        super().__init__()
        self.n_branches = n_branches
        self.input_hw = input_hw
        self.norm = RGB2YUVNorm()

        def conv(cin, cout, k, s):
            layers = [nn.Conv2d(cin, cout, k, s)]
            if batchnorm:
                layers.append(nn.BatchNorm2d(cout))
            layers.append(_act(act))
            return nn.Sequential(*layers)

        self.convs = nn.ModuleList([
            conv(3, 24, 5, 2), conv(24, 36, 5, 2), conv(36, 48, 5, 2),
            conv(48, 64, 3, 1), conv(64, 64, 3, 1),
        ])
        with torch.no_grad():
            f = torch.zeros(1, 3, *input_hw)
            for c in self.convs:
                f = c(f)
        self.flat_dim = int(f.numel())

        def head():
            return nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(self.flat_dim, 100), _act(act), nn.Dropout(dropout),
                nn.Linear(100, 50), _act(act),
                nn.Linear(50, 10), _act(act),
                nn.Linear(10, 1))

        self.heads = nn.ModuleList([head() for _ in range(n_branches)])

    def trunk(self, x: torch.Tensor, keep: bool = False):
        x = self.norm(x)
        feats = []
        for c in self.convs:
            x = c(x)
            if keep:
                feats.append(x)
        return (x, feats) if keep else x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.trunk(x).flatten(1)
        return torch.cat([h(f) for h in self.heads], dim=1)


def select_branch(out: torch.Tensor, cmd: torch.Tensor | None) -> torch.Tensor:
    """Pick each sample's head given its command index (0=FOLLOW,1=LEFT,2=RIGHT)."""
    if out.shape[1] == 1 or cmd is None:
        return out[:, 0]
    return out.gather(1, cmd.long().view(-1, 1)).squeeze(1)


def count_parameters(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


# ============================================================================
# 3. Augmentation  (was pilotnet/augment.py)
# ============================================================================
# Training-time augmentation on uint8 RGB images of shape (66, 200, 3).
# Geometric augmentations change the label consistently:
#   * horizontal flip  -> target = -target, LEFT <-> RIGHT command swap
#   * horizontal shift -> target += sign * shift_px * k  (recovery heuristic used
#     for the Udacity data; the simulator does not need it because it renders
#     physically perturbed poses with exact expert labels).
# Photometric augmentations leave the label unchanged.
LEFT, RIGHT = COMMANDS.index("LEFT"), COMMANDS.index("RIGHT")


def flip(img, target, cmd):
    cmd = {LEFT: RIGHT, RIGHT: LEFT}.get(cmd, cmd)
    return np.ascontiguousarray(img[:, ::-1]), -target, cmd


def brightness_saturation(img, rng):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 2] *= rng.uniform(0.6, 1.3)
    hsv[..., 1] *= rng.uniform(0.7, 1.3)
    hsv[..., 0] = (hsv[..., 0] + rng.uniform(-5, 5)) % 180
    return cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB)


def random_shadow(img, rng):
    h, w = img.shape[:2]
    x1, x2 = rng.uniform(0, w, size=2)
    poly = np.array([[x1, 0], [x2, h], [w if rng.random() < .5 else 0, h],
                     [w if rng.random() < .5 else 0, 0]], np.int32)
    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [poly], 1)
    out = img.astype(np.float32)
    out[mask == 1] *= rng.uniform(0.45, 0.8)
    return out.clip(0, 255).astype(np.uint8)


def shift(img, dx, dy):
    h, w = img.shape[:2]
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)


def augment(img, target, cmd, rng, *, allow_flip=True, shift_k=0.0, right_positive=True):
    """Return augmented (img, target, cmd). shift_k = label change per pixel of
    horizontal shift (0 disables label-changing shifts)."""
    if allow_flip and rng.random() < 0.5:
        img, target, cmd = flip(img, target, cmd)
    if shift_k > 0 and rng.random() < 0.5:
        dx = rng.uniform(-25, 25)
        img = shift(img, dx, 0)
        # content moved right (dx>0) -> vehicle is left of the road -> steer right
        target += (1 if right_positive else -1) * dx * shift_k
    if rng.random() < 0.3:                       # pitch from bumpy terrain
        img = shift(img, 0, rng.uniform(-4, 4))
    if rng.random() < 0.8:
        img = brightness_saturation(img, rng)
    if rng.random() < 0.3:
        img = random_shadow(img, rng)
    if rng.random() < 0.1:
        img = cv2.GaussianBlur(img, (3, 3), 0)
    if rng.random() < 0.2:
        img = np.clip(img + rng.normal(0, 6, img.shape), 0, 255).astype(np.uint8)
    return img, target, cmd


# ============================================================================
# 4. Dataset readers  (was pilotnet/data.py)
# ============================================================================
# sully   : SullyChen/driving-datasets. data.txt lines "filename.jpg angle"
#           (2017 set) or "filename.jpg angle,date time" (2018 set).
#           angle = steering-wheel angle in degrees. Target = radians(angle).
# udacity : Udacity self-driving-car-sim recordings, driving_log.csv with
#           center,left,right,steering,throttle,brake,speed. Steering is already
#           normalised to [-1, 1] (x25 deg), + = right. Side cameras are used
#           with a +-0.2 correction.
# sim     : this project's simulator (MATLAB or Python) and every `convert`
#           output: labels.csv with filename,target,command,...
#           Simulator: target = kappa / kappa_max, + = left.
# Splits are made over *contiguous blocks* of frames, never per-frame.
IMG_H, IMG_W = 66, 200


@dataclass
class Sample:
    path: str
    target: float
    command: int = 0


@dataclass
class SourceInfo:
    name: str
    crop: tuple[float, float]      # keep rows [top_frac, bottom_frac)
    right_positive: bool           # True if target > 0 means steer right
    shift_k: float                 # label change per px of horizontal shift (0 = off)
    has_commands: bool = False


def read_sully(root: str):
    txt = os.path.join(root, "data.txt")
    out = []
    with open(txt) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            ang = float(parts[1].split(",")[0])
            out.append(Sample(os.path.join(root, parts[0]), math.radians(ang)))
    # 455x256 frames: the reference implementation keeps the bottom 150 rows
    # (drops sky / dashboard-free horizon). Sign convention is not documented,
    # so label-changing shifts are disabled.
    return out, SourceInfo("sully", (106 / 256, 1.0), True, 0.0)


def read_udacity(root: str, side_correction: float = 0.2, use_side: bool = True):
    csv_path = os.path.join(root, "driving_log.csv")
    img_dir = os.path.join(root, "IMG")

    def resolve(p):
        p = p.strip().replace("\\", "/")
        return os.path.join(img_dir, os.path.basename(p))

    out = []
    with open(csv_path) as f:
        for row in csv.reader(f):
            if not row or row[0].strip().lower() == "center":
                continue
            steer = float(row[3])
            out.append(Sample(resolve(row[0]), steer))
            if use_side:
                out.append(Sample(resolve(row[1]), steer + side_correction))   # left cam -> steer right
                out.append(Sample(resolve(row[2]), steer - side_correction))
    # 320x160 frames: drop 60 px of sky and 25 px of bonnet
    return out, SourceInfo("udacity", (60 / 160, 135 / 160), True, 0.004)


def read_sim(root: str):
    out = []
    with open(os.path.join(root, "labels.csv")) as f:
        for r in csv.DictReader(f):
            out.append(Sample(os.path.join(root, "images", r["filename"]),
                              float(r["target"]), int(r.get("command", 0) or 0)))
    return out, SourceInfo("sim", (0.0, 1.0), False, 0.0, has_commands=True)


READERS = {"sully": read_sully, "udacity": read_udacity, "sim": read_sim}


def block_split(n: int, block: int, val_frac: float, test_frac: float, seed: int = 0):
    """Assign contiguous blocks of `block` frames to train/val/test."""
    n_blocks = math.ceil(n / block)
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_blocks)
    n_test = max(1, round(test_frac * n_blocks)) if test_frac > 0 else 0
    n_val = max(1, round(val_frac * n_blocks)) if val_frac > 0 else 0
    test_b, val_b = set(order[:n_test]), set(order[n_test:n_test + n_val])
    idx = {"train": [], "val": [], "test": []}
    for i in range(n):
        b = i // block
        idx["test" if b in test_b else "val" if b in val_b else "train"].append(i)
    return idx


def load_image(path: str, crop: tuple[float, float]) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h = img.shape[0]
    img = img[int(round(crop[0] * h)):int(round(crop[1] * h))]
    if img.shape[:2] != (IMG_H, IMG_W):
        img = cv2.resize(img, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
    return img


class SteeringDataset(Dataset):
    def __init__(self, samples, info: SourceInfo, train: bool, seed: int = 0, cache: bool = False):
        self.s, self.info, self.train = samples, info, train
        self.seed = seed
        self.cache = {} if cache else None
        self._rng = None      # created lazily inside each DataLoader worker

    def __len__(self):
        return len(self.s)

    def __getitem__(self, i):
        smp = self.s[i]
        if self.cache is not None and i in self.cache:
            img = self.cache[i]
        else:
            img = load_image(smp.path, self.info.crop)
            if self.cache is not None:
                self.cache[i] = img
        target, cmd = smp.target, smp.command
        if self.train:
            if self._rng is None:   # torch gives every worker a distinct, seed-derived initial_seed
                self._rng = np.random.default_rng((self.seed, torch.initial_seed() % 2**32))
            img, target, cmd = augment(img, target, cmd, self._rng, shift_k=self.info.shift_k,
                                       right_positive=self.info.right_positive)
        x = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))).float()
        return x, torch.tensor(target, dtype=torch.float32), torch.tensor(cmd, dtype=torch.long)


def balance_weights(samples, n_bins: int = 21, alpha: float = 0.5, by_command: bool = False):
    """Per-sample weights ~ 1/freq^alpha over steering bins (and commands).
    Counteracts the straight-driving bias of driving logs."""
    t = np.array([s.target for s in samples])
    lim = np.percentile(np.abs(t), 99) + 1e-6
    bins = np.clip(((t / lim + 1) / 2 * n_bins).astype(int), 0, n_bins - 1)
    cnt = np.bincount(bins, minlength=n_bins).astype(float)
    w = (1.0 / cnt[bins]) ** alpha
    if by_command:
        c = np.array([s.command for s in samples])
        cc = np.bincount(c, minlength=3).astype(float)
        w *= (1.0 / np.maximum(cc[c], 1)) ** alpha
    return w / w.mean()


# ============================================================================
# 5. Metrics  (was pilotnet/metrics.py)
# ============================================================================
def regression_metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    e = y_pred - y_true
    ss_res = float(np.sum(e ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2)) + 1e-12
    return dict(
        n=int(len(e)),
        rmse=float(np.sqrt(np.mean(e ** 2))),
        mae=float(np.mean(np.abs(e))),
        max_abs_err=float(np.max(np.abs(e))),
        r2=float(1 - ss_res / ss_tot),
        # baseline: always predict the training mean ~ 0 -> how much better than "drive straight"
        rmse_zero_baseline=float(np.sqrt(np.mean(y_true ** 2))),
    )


def smoothness(y, dt: float) -> float:
    """RMS rate of change of a steering signal (units/s). Lower = smoother."""
    y = np.asarray(y, float)
    return float(np.sqrt(np.mean((np.diff(y) / dt) ** 2))) if len(y) > 1 else 0.0


def autonomy(n_interventions: int, elapsed_s: float, penalty_s: float = 6.0) -> float:
    """NVIDIA autonomy metric (Bojarski et al., 2016):
    (1 - interventions * 6 s / elapsed time) * 100."""
    return float(max(0.0, 1.0 - n_interventions * penalty_s / max(elapsed_s, 1e-9)) * 100.0)


# ============================================================================
# 6. VisualBackProp  (was pilotnet/vbp.py; Bojarski et al., ICRA 2018)
# ============================================================================
# 1. Average every conv layer's feature maps over channels.
# 2. Start at the deepest averaged map; repeatedly up-scale it to the previous
#    layer's size with a transposed convolution whose kernel is all ones, and
#    multiply point-wise with that layer's averaged map.
# 3. Up-scale once more to the input resolution and normalise to [0, 1].
@torch.no_grad()
def visual_backprop(model, x: torch.Tensor) -> torch.Tensor:
    """x: (N,3,H,W) float RGB 0..255. Returns masks (N,H,W) in [0,1]."""
    model.eval()
    _, feats = model.trunk(x, keep=True)
    convs = [c[0] for c in model.convs]                        # nn.Conv2d of each block
    sizes = [x.shape[-2:]] + [f.shape[-2:] for f in feats]
    m = feats[-1].mean(1, keepdim=True)
    for li in range(len(feats) - 1, -1, -1):
        conv = convs[li]
        k, s = conv.kernel_size, conv.stride
        tgt = sizes[li]
        out_h = (m.shape[-2] - 1) * s[0] + k[0]
        out_w = (m.shape[-1] - 1) * s[1] + k[1]
        op = (tgt[0] - out_h, tgt[1] - out_w)
        w = torch.ones(1, 1, *k, device=x.device)
        m = F.conv_transpose2d(m, w, stride=s, output_padding=op)
        if li > 0:
            m = m * feats[li - 1].mean(1, keepdim=True)
    m = m[:, 0]
    mn = m.flatten(1).min(1)[0].view(-1, 1, 1)
    mx = m.flatten(1).max(1)[0].view(-1, 1, 1)
    return (m - mn) / (mx - mn + 1e-8)


# ============================================================================
# 7. Course, A*, route, texture  (was sim/course.py)
# ============================================================================
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


# Procedural texture (identical in MATLAB: same hash, same arithmetic)
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


# ============================================================================
# 8. Camera renderer  (was sim/render.py)
# ============================================================================
# Pinhole ground-plane camera (flat-world ray casting). Each pixel ray is
# intersected with the ground plane z = 0 once (the camera is rigid), giving a
# fixed offset (pf, pl) in the vehicle frame. Per frame we only rotate/translate
# these offsets into the world and look up the texture -> ~1 ms per frame.
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


# ============================================================================
# 9. Vehicle + expert  (was sim/vehicle.py)
# ============================================================================
def bicycle_step(state, delta_cmd: float, v: float, dt: float = C.DT):
    """state = [x, y, psi, delta] (rear-axle reference). Returns new state.

    Steering actuator: first-order lag (tau) with slew-rate limit, then the
    kinematic bicycle  x' = v cos(psi), y' = v sin(psi), psi' = v tan(delta)/L.
    Semi-implicit Euler: the updated steering angle is used for the pose.
    """
    x, y, psi, d = state
    delta_cmd = max(-C.MAX_STEER, min(C.MAX_STEER, delta_cmd))
    rate = (delta_cmd - d) / C.STEER_TAU
    rate = max(-C.STEER_RATE, min(C.STEER_RATE, rate))
    d = max(-C.MAX_STEER, min(C.MAX_STEER, d + rate * dt))
    x += v * math.cos(psi) * dt
    y += v * math.sin(psi) * dt
    psi += v / C.WHEELBASE * math.tan(d) * dt
    psi = math.atan2(math.sin(psi), math.cos(psi))
    return [x, y, psi, d]


def pure_pursuit(route: Route, x: float, y: float, psi: float, v: float, hint: int):
    """Pure pursuit (Coulter, 1992) on the planned route.

    Returns (kappa, nearest_index, cte). kappa = 2*y_L / L_d^2 where y_L is the
    lateral coordinate of the look-ahead point in the vehicle frame.
    """
    i = nearest_on_route(route, x, y, hint)
    ld = C.LOOKAHEAD_MIN + C.LOOKAHEAD_GAIN * v
    s_t = route.s[i] + ld
    j = i
    n = len(route.s)
    while j < n - 1 and route.s[j] < s_t:
        j += 1
    px, py = route.pts[j]
    dx = (px - x) * math.cos(psi) + (py - y) * math.sin(psi)
    dy = -(px - x) * math.sin(psi) + (py - y) * math.cos(psi)
    L2 = max(dx * dx + dy * dy, 1e-6)
    kappa = 2.0 * dy / L2
    kappa = max(-C.KAPPA_MAX, min(C.KAPPA_MAX, kappa))
    return kappa, i, cross_track(route, i, x, y)


def kappa_to_delta(kappa: float) -> float:
    return math.atan(C.WHEELBASE * kappa)


# ============================================================================
# 10. ONNX inference  (was pilotnet_infer.py)
# ============================================================================
class PilotNetONNX:
    def __init__(self, onnx_path: str, threads: int = 1):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = int(threads)
        self.sess = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])

    def predict(self, img_rgb: np.ndarray) -> np.ndarray:
        """img_rgb: (66,200,3) uint8 RGB -> (n_branches,) float."""
        x = img_rgb.astype(np.float32).transpose(2, 0, 1)[None]
        return self.sess.run(None, {"image": x})[0][0]

    def predict_bytes(self, buf, h: int = 66, w: int = 200):
        img = np.frombuffer(bytes(buf), dtype=np.uint8).reshape(int(h), int(w), 3)
        return tuple(float(v) for v in self.predict(img))


# ============================================================================
# 11. Dataset generator  (was sim/gen_dataset.py; mirrored by abaja_sim('generate'))
# ============================================================================
# The expert is pure pursuit on the A*-planned route. Two mechanisms produce the
# off-centre "recovery" data a behaviour-cloned policy needs:
#   1. Noise injection: an Ornstein-Uhlenbeck perturbation is added to the
#      *executed* command for part of the drive, while the *recorded* label is
#      the expert's clean command at the reached pose.
#   2. Pose perturbation: for each recorded frame an extra image is rendered
#      from a laterally shifted / rotated pose and labelled with the expert's
#      command at that pose.
def add_gen_args(ap):
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


def cmd_gen(a):
    print(f"[gen] {a.cycles} cycle(s), seed {a.seed} -> {a.out}")
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


# ============================================================================
# 12. Training  (was train.py)
# ============================================================================
def add_train_args(ap):
    ap.add_argument("--dataset", choices=list(READERS), required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="runs/exp")
    ap.add_argument("--branches", type=int, default=1, help="1 = PilotNet, 3 = FOLLOW/LEFT/RIGHT heads")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--loss", choices=["mse", "huber"], default="mse")
    ap.add_argument("--balance", action="store_true", help="re-weight rare steering angles / commands")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--block", type=int, default=500, help="frames per contiguous split block")
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--cache", action="store_true", help="keep decoded images in RAM")
    ap.add_argument("--max-samples", type=int, default=0, help="debug: use only the first N samples")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


@torch.no_grad()
def run_eval(model, loader, device):
    model.eval()
    ys, ps = [], []
    for x, y, c in loader:
        out = model(x.to(device, non_blocking=True))
        ps.append(select_branch(out, c.to(device)).cpu())
        ys.append(y)
    y, p = torch.cat(ys).numpy(), torch.cat(ps).numpy()
    return float(np.sqrt(np.mean((p - y) ** 2))), float(np.mean(np.abs(p - y)))


def cmd_train(a):
    set_seed(a.seed)
    device = torch.device("cuda" if (a.device == "auto" and torch.cuda.is_available()) else
                          ("cpu" if a.device == "auto" else a.device))
    os.makedirs(a.out, exist_ok=True)

    samples, info = READERS[a.dataset](a.root)
    if a.max_samples:
        samples = samples[:a.max_samples]
    if a.branches > 1 and not info.has_commands:
        raise SystemExit("--branches > 1 needs a dataset with navigation commands (sim)")
    split = block_split(len(samples), a.block, a.val_frac, a.test_frac, a.seed)
    with open(os.path.join(a.out, "split.json"), "w") as f:
        json.dump(split, f)
    tr = [samples[i] for i in split["train"]]
    va = [samples[i] for i in split["val"]]
    print(f"[data] {a.dataset}: {len(samples)} samples -> train {len(tr)} / val {len(va)} / "
          f"test {len(split['test'])}  (block split, block={a.block})")

    ds_tr = SteeringDataset(tr, info, train=True, seed=a.seed, cache=a.cache)
    ds_va = SteeringDataset(va, info, train=False, cache=a.cache)
    sampler = None
    if a.balance:
        w = balance_weights(tr, by_command=a.branches > 1)
        sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), len(tr), replacement=True)
    # persistent workers keep each worker's image cache and RNG alive across epochs
    kw = dict(num_workers=a.workers, pin_memory=device.type == "cuda",
              persistent_workers=a.workers > 0)
    dl_tr = DataLoader(ds_tr, a.bs, shuffle=sampler is None, sampler=sampler, drop_last=True, **kw)
    dl_va = DataLoader(ds_va, 256, shuffle=False, **kw)

    model = PilotNet(n_branches=a.branches, dropout=a.dropout).to(device)
    print(f"[model] PilotNet branches={a.branches} params={count_parameters(model):,} device={device}")
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * len(dl_tr),
                                                pct_start=0.15)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    loss_fn = F.mse_loss if a.loss == "mse" else (lambda p, y: F.huber_loss(p, y, delta=0.1))

    best, bad, hist = float("inf"), 0, []
    cfg = dict(vars(a), n_branches=a.branches, source=info.name, crop=info.crop,
               right_positive=info.right_positive)
    for ep in range(1, a.epochs + 1):
        model.train()
        t0, run, n = time.time(), 0.0, 0
        for x, y, c in dl_tr:
            x, y, c = x.to(device, non_blocking=True), y.to(device), c.to(device)
            with torch.autocast(device.type, enabled=use_amp):
                p = select_branch(model(x), c if a.branches > 1 else None)
                loss = loss_fn(p.float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            run += loss.item() * len(y)
            n += len(y)
        va_rmse, va_mae = run_eval(model, dl_va, device)
        hist.append(dict(epoch=ep, train_loss=run / n, val_rmse=va_rmse, val_mae=va_mae,
                         lr=sched.get_last_lr()[0], sec=time.time() - t0))
        flag = ""
        if va_rmse < best - 1e-5:
            best, bad, flag = va_rmse, 0, "  *best*"
            torch.save(dict(state_dict=model.state_dict(), cfg=cfg, epoch=ep, val_rmse=va_rmse),
                       os.path.join(a.out, "best.pt"))
        else:
            bad += 1
        print(f"ep {ep:3d}  train {run / n:.5f}  val RMSE {va_rmse:.4f}  MAE {va_mae:.4f}  "
              f"({time.time() - t0:.0f}s){flag}", flush=True)
        if bad >= a.patience:
            print(f"[early stop] no val improvement for {a.patience} epochs")
            break

    with open(os.path.join(a.out, "history.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(hist[0]))
        w.writeheader()
        w.writerows(hist)
    print(f"[done] best val RMSE {best:.4f} -> {os.path.join(a.out, 'best.pt')}")


# ============================================================================
# 13. Export ONNX + MATLAB weights  (was export_onnx.py)
# ============================================================================
# ONNX graph: input  "image"  float32 (N, 3, 66, 200), RGB 0..255
#             output "steer"  float32 (N, n_branches)
# The RGB->YUV conversion and normalisation are inside the graph.
def load_model(ckpt_path: str, device="cpu") -> tuple[PilotNet, dict]:
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ck["cfg"]
    m = PilotNet(n_branches=cfg["n_branches"], dropout=cfg.get("dropout", 0.2))
    m.load_state_dict(ck["state_dict"])
    return m.eval().to(device), cfg


def add_export_args(ap):
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--opset", type=int, default=17)


def cmd_export(a):
    out = a.out or os.path.join(os.path.dirname(a.ckpt), "pilotnet.onnx")

    model, cfg = load_model(a.ckpt)
    dummy = torch.rand(1, 3, 66, 200) * 255
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    torch.onnx.export(model, dummy, out, input_names=["image"], output_names=["steer"],
                      dynamic_axes={"image": {0: "N"}, "steer": {0: "N"}},
                      opset_version=a.opset, dynamo=False)

    import onnxruntime as ort
    sess = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
    x = (np.random.default_rng(0).random((8, 3, 66, 200)) * 255).astype(np.float32)
    y_ort = sess.run(None, {"image": x})[0]
    with torch.no_grad():
        y_pt = model(torch.from_numpy(x)).numpy()
    err = float(np.abs(y_ort - y_pt).max())
    assert err < 1e-4, f"ONNX mismatch {err}"

    meta = dict(input="image", output="steer", input_layout="NCHW", input_range="RGB 0..255 float32",
                input_hw=[66, 200], n_branches=cfg["n_branches"],
                branch_names=list(COMMANDS[:cfg["n_branches"]]) if cfg["n_branches"] > 1 else ["STEER"],
                source=cfg["source"], target_definition=(
                    "kappa / kappa_max, + = left" if cfg["source"] == "sim" else
                    "dataset steering units (see README)"),
                max_abs_diff_vs_pytorch=err)
    with open(os.path.splitext(out)[0] + "_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"exported {out}  (max |onnx - torch| = {err:.2e})")
    mat = export_matlab_weights(model, cfg, os.path.join(os.path.dirname(out), "pilotnet_weights.mat"))
    print(f"exported {mat}  (for matlab/abaja_sim.m, native backend)")


def export_matlab_weights(model, cfg, path):
    """Weights for the toolbox-free MATLAB forward pass (ELU, no batch-norm)."""
    from scipy.io import savemat
    assert cfg.get("act", "elu") == "elu" and not cfg.get("batchnorm", False)
    sd = {k: v.detach().cpu().numpy().astype(np.float64) for k, v in model.state_dict().items()}
    d = dict(n_branches=float(cfg["n_branches"]),
             norm_w=sd["norm.weight"].reshape(3, 3), norm_b=sd["norm.bias"])
    for l in range(5):
        conv = model.convs[l][0]
        d[f"conv{l + 1}_w"] = sd[f"convs.{l}.0.weight"]
        d[f"conv{l + 1}_b"] = sd[f"convs.{l}.0.bias"]
        d[f"conv{l + 1}_s"] = float(conv.stride[0])
    for b in range(cfg["n_branches"]):
        for f, li in enumerate((1, 4, 6, 8), start=1):
            d[f"head{b}_fc{f}_w"] = sd[f"heads.{b}.{li}.weight"]
            d[f"head{b}_fc{f}_b"] = sd[f"heads.{b}.{li}.bias"]
    savemat(path, d)
    return path


# ============================================================================
# 14. Open-loop evaluation  (was evaluate.py)
# ============================================================================
def add_evaluate_args(ap):
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--vbp", type=int, default=8, help="number of VisualBackProp examples")
    ap.add_argument("--fps", type=float, default=10.0, help="frame rate for smoothness metric")
    ap.add_argument("--workers", type=int, default=2, help="DataLoader workers (0 = main process)")


def cmd_evaluate(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = a.out or os.path.join(os.path.dirname(a.ckpt), "eval")
    os.makedirs(out, exist_ok=True)

    model, cfg = load_model(a.ckpt)
    samples, info = READERS[cfg["dataset"]](a.root)
    if cfg.get("max_samples"):
        samples = samples[:cfg["max_samples"]]
    with open(os.path.join(os.path.dirname(a.ckpt), "split.json")) as f:
        split = json.load(f)
    idx = list(range(len(samples))) if a.split == "all" else split[a.split]
    sel = [samples[i] for i in idx]
    print(f"[eval] {cfg['dataset']} split={a.split}: {len(sel)} frames")
    dl = DataLoader(SteeringDataset(sel, info, train=False), 256, shuffle=False, num_workers=a.workers)

    ys, ps, cs, all_heads = [], [], [], []
    with torch.no_grad():
        for x, y, c in dl:
            o = model(x)
            ps.append(select_branch(o, c if cfg["n_branches"] > 1 else None))
            ys.append(y)
            cs.append(c)
            all_heads.append(o)
    y, p, c = torch.cat(ys).numpy(), torch.cat(ps).numpy(), torch.cat(cs).numpy()

    res = dict(split=a.split, overall=regression_metrics(y, p),
               smoothness_true=smoothness(y, 1 / a.fps), smoothness_pred=smoothness(p, 1 / a.fps))
    if cfg["n_branches"] > 1:
        res["per_command"] = {COMMANDS[k]: regression_metrics(y[c == k], p[c == k])
                              for k in range(cfg["n_branches"]) if np.any(c == k)}

    # inference latency, batch 1, CPU (what an on-vehicle loop sees)
    xb = torch.rand(1, 3, 66, 200) * 255
    for _ in range(10):
        model(xb)
    t0 = time.perf_counter()
    for _ in range(200):
        with torch.no_grad():
            model(xb)
    res["latency_ms_cpu_batch1"] = (time.perf_counter() - t0) / 200 * 1000
    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump(res, f, indent=2)
    with open(os.path.join(out, "predictions.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "path", "target", "pred", "command"])
        for k, i in enumerate(idx):
            w.writerow([i, samples[i].path, y[k], p[k], c[k]])

    # ---- plots
    n = min(len(y), 1500)
    plt.figure(figsize=(11, 3.4))
    plt.plot(y[:n], lw=1.2, label="ground truth")
    plt.plot(p[:n], lw=1.0, label="PilotNet")
    plt.xlabel("test frame (contiguous blocks)")
    plt.ylabel("normalised steering")
    plt.legend()
    plt.title(f"Test RMSE {res['overall']['rmse']:.4f}, MAE {res['overall']['mae']:.4f}")
    plt.tight_layout()
    plt.savefig(os.path.join(out, "timeseries.png"), dpi=130)
    plt.close()

    lim = float(np.percentile(np.abs(np.concatenate([y, p])), 99.5)) * 1.05
    plt.figure(figsize=(4.4, 4.2))
    plt.scatter(y, p, s=3, alpha=.35)
    plt.plot([-lim, lim], [-lim, lim], "k--", lw=.8)
    plt.xlim(-lim, lim)
    plt.ylim(-lim, lim)
    plt.xlabel("ground truth")
    plt.ylabel("prediction")
    plt.title(f"R² = {res['overall']['r2']:.3f}")
    plt.tight_layout()
    plt.savefig(os.path.join(out, "scatter.png"), dpi=130)
    plt.close()

    if a.vbp > 0:
        pick = np.linspace(0, len(sel) - 1, a.vbp).astype(int)
        ds = SteeringDataset(sel, info, train=False)
        xs = torch.stack([ds[i][0] for i in pick])
        masks = visual_backprop(model, xs).numpy()
        fig, ax = plt.subplots(len(pick), 2, figsize=(8, 1.35 * len(pick)))
        for r, i in enumerate(pick):
            img = xs[r].permute(1, 2, 0).numpy().astype(np.uint8)
            ax[r, 0].imshow(img)
            ax[r, 1].imshow(img)
            ax[r, 1].imshow(masks[r], cmap="inferno", alpha=.6)
            ax[r, 0].set_ylabel(f"y={y[i]:+.2f}\np={p[i]:+.2f}", fontsize=7)
            for k in (0, 1):
                ax[r, k].set_xticks([])
                ax[r, k].set_yticks([])
        ax[0, 0].set_title("input")
        ax[0, 1].set_title("VisualBackProp mask")
        plt.tight_layout()
        plt.savefig(os.path.join(out, "visualbackprop.png"), dpi=130)
        plt.close()
    print(json.dumps(res, indent=2))
    print(f"[eval] wrote {out}")


# ============================================================================
# 15. MATLAB reference writer  (was make_reference.py)
# ============================================================================
def add_reference_args(ap):
    ap.add_argument("--onnx", default=None)
    ap.add_argument("--out", default=os.path.join("matlab", "reference"))


def cmd_reference(a):
    os.makedirs(a.out, exist_ok=True)
    course = build_course()
    tex, _ = build_texture(course)
    cam = camera_precompute()
    route = build_route(course, ["S", "A", "S", "B", "S"])
    net = None
    if a.onnx:
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


# ============================================================================
# 16. Closed-loop simulation  (was sim/closed_loop.py; mirrored by abaja_sim('closedloop'))
# ============================================================================
# Navigation stack:  mission checkpoints -> A* route on the road graph ->
# high-level command (FOLLOW/LEFT/RIGHT) from route progress -> the CNN
# (selected head) outputs curvature -> steering actuator -> kinematic bicycle.
# The route is used only to issue commands and to score the run; the CNN steers
# from pixels alone.
def point_to_polyline(p, x, y):
    return float(np.min(np.hypot(p[:, 0] - x, p[:, 1] - y)))


def closed_loop_run(policy, cycles=2, latency_steps=1, speed=C.SPEED, seed=0, start_offset=0.0,
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
    res = dict(
        distance_m=float(route.s[-1]), elapsed_s=float(t), interventions=len(interventions),
        autonomy_pct=autonomy(len(interventions), t),
        cte_mean_abs=float(cte_abs.mean()), cte_rms=float(np.sqrt(np.mean(log[:, 5] ** 2))),
        cte_p95=float(np.percentile(cte_abs, 95)), cte_max=float(cte_abs.max()),
        steer_rate_rms_deg_s=math.degrees(smoothness(log[:, 4], C.DT)),
        forks_total=len(route.forks), forks_correct=int(sum(bool(r) for r in fork_res)),
        latency_steps=latency_steps, speed=speed)
    return res, log, interventions, route, course, tex


def add_closedloop_args(ap):
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


def cmd_closedloop(a):
    os.makedirs(a.out, exist_ok=True)

    if a.policy == "cnn":
        if not a.onnx:
            raise SystemExit("--policy cnn needs --onnx <path to pilotnet.onnx>")
        net = PilotNetONNX(a.onnx)
        pol = net.predict
    else:
        pol = "expert"
    print(f"[closedloop] policy={a.policy} cycles={a.cycles} speed={a.speed} "
          f"texture_seed={a.texture_seed} gain={a.gain}")
    res, log, iv, route, course, tex = closed_loop_run(pol, a.cycles, a.latency_steps, a.speed,
                                                       start_offset=a.start_offset,
                                                       texture_seed=a.texture_seed, gain=a.gain)
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
    print(f"[closedloop] wrote {a.out}")


# ============================================================================
# 17. Dataset download  (`download`)
# ============================================================================
# Every URL below was taken from the dataset's own README / download script and
# checked to respond (Nov 2026). Sizes are the ones stated by the source or
# measured with an HTTP HEAD request. See datasets/DATASETS.md.
GB = 1e9
SULLY_DRIVE = {   # github.com/SullyChen/driving-datasets README
    "2017": dict(id="1Ue4XohCOV5YXy57S_5tDfCVqzLr101M7", gb=2.2, frames="~45.5k",
                 url="https://drive.google.com/file/d/1Ue4XohCOV5YXy57S_5tDfCVqzLr101M7/view?usp=sharing"),
    "2018": dict(id="1PZWa6H0i1PCH9zuYcIh5Ouk_p-9Gh58B", gb=3.1, frames="~63k",
                 url="https://drive.google.com/open?id=1PZWa6H0i1PCH9zuYcIh5Ouk_p-9Gh58B"),
}
# github.com/udacity/self-driving-car-sim README, Term 1, Version 2 (2/07/17), Windows
UDACITY_SIM_URL = "https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/Term1-Sim/term1-simulator-windows.zip"
# Udacity CarND "Behavioral Cloning" sample recording (track 1), hosted on the same
# Udacity CDN as the Term 1 v1 simulator builds. Not linked from a public README
# (the course README says it is "provided in the classroom"); verified: HTTP 200,
# 333 MB zip containing data/driving_log.csv + data/IMG/.
UDACITY_SAMPLE_URL = "https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip"
COMMA_URL = "https://archive.org/download/comma-dataset/comma-dataset.zip"      # commaai/research get_data.sh
COMMA2K19_TORRENT = "http://academictorrents.com/details/65a2fbc964078aff62076ff4e103f18b951c5ddb"
COMMA2K19_SAMPLE_API = "https://api.github.com/repos/commaai/comma2k19/git/trees/master?recursive=1"
COMMA2K19_SAMPLE_RAW = "https://raw.githubusercontent.com/commaai/comma2k19/master/"
TARTAN_ENDPOINT = ("https://airlab-cloud.andrew.cmu.edu:8080/swift/v1/"
                   "AUTH_ac8533a83cff4d48bc8c608ad222d330/tartandrive")          # castacks/tartan_drive download_files.py
TARTAN_FILELIST = "https://raw.githubusercontent.com/castacks/tartan_drive/main/azfiles.txt"
RELLIS_IMAGES = dict(id="1F3Leu0H_m6aPVpZITragfreO_SGtL2yV", gb=11.0,
                     url="https://drive.google.com/file/d/1F3Leu0H_m6aPVpZITragfreO_SGtL2yV/view?usp=sharing")
RELLIS_FOLDER = "https://drive.google.com/drive/folders/1aZ1tJ3YYcWuL3oWKnrTIC5gq46zx1bMc?usp=sharing"
BIG_GB = 5.0                     # anything larger needs --yes


def add_download_args(ap):
    ap.add_argument("--what", choices=["sully", "udacity_sim", "udacity_sample", "comma", "comma2k19",
                                       "tartandrive", "rellis", "all"], required=True)
    ap.add_argument("--dest", default="datasets")
    ap.add_argument("--sully-set", choices=["2017", "2018", "both"], default="2017",
                    help="SullyChen set: 2017 = 'Dataset 1' (2.2 GB), 2018 = 07/01/2018 set (3.1 GB)")
    ap.add_argument("--tartan-files", type=int, default=1, help="number of TartanDrive archives (~70-127 GB each)")
    ap.add_argument("--yes", action="store_true", help=f"confirm downloads larger than {BIG_GB:.0f} GB")
    ap.add_argument("--keep-zip", action="store_true", help="keep archives after extraction")


def _free_gb(path):
    p = path
    while not os.path.exists(p):
        p = os.path.dirname(p)
    return shutil.disk_usage(p).free / GB


def _size_check(name, need_gb, dest, yes):
    """Print size + free space; return True if the download may start."""
    free = _free_gb(dest)
    print(f"[download] {name}: needs ~{need_gb:.1f} GB, free on {os.path.splitdrive(os.path.abspath(dest))[0] or '/'} "
          f"= {free:.1f} GB")
    if need_gb > free * 0.95:
        print(f"[download] {name}: NOT enough free disk space - skipped. Free up space or use --dest on another drive.")
        return False
    if need_gb > BIG_GB and not yes:
        print(f"[download] {name}: larger than {BIG_GB:.0f} GB - re-run with --yes to start it.")
        return False
    return True


def _http_download(url, path, retries=5):
    """Download to <path>.part with progress, resume (HTTP Range) and retries; rename when complete."""
    if os.path.exists(path):
        return path
    for attempt in range(1, retries + 1):
        try:
            _http_download_once(url, path + ".part")
            os.replace(path + ".part", path)
            return path
        except (OSError, TimeoutError) as e:          # urllib errors are OSError subclasses
            if attempt == retries:
                raise
            print(f"\n  network error ({e}); retry {attempt}/{retries - 1} in {5 * attempt} s (resuming)")
            time.sleep(5 * attempt)


def _http_download_once(url, path):
    import urllib.request
    os.makedirs(os.path.dirname(path), exist_ok=True)
    have = os.path.getsize(path) if os.path.exists(path) else 0
    req = urllib.request.Request(url, headers={"User-Agent": "pilotnet-abaja-downloader"})
    if have:
        req.add_header("Range", f"bytes={have}-")
    try:
        r = urllib.request.urlopen(req, timeout=60)
    except urllib.error.HTTPError as e:
        if e.code == 416:                       # already complete
            return path
        raise
    total = int(r.headers.get("Content-Length", 0)) + (have if r.status == 206 else 0)
    mode = "ab" if r.status == 206 else "wb"
    done = have if r.status == 206 else 0
    t0, last = time.time(), 0.0
    with open(path, mode) as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if time.time() - last > 2:
                last = time.time()
                pct = f"{100 * done / total:5.1f}%" if total else ""
                print(f"\r  {os.path.basename(path)}  {done / 1e6:9.1f} MB {pct}  "
                      f"{(done - have) / 1e6 / max(time.time() - t0, 1e-3):6.1f} MB/s", end="", flush=True)
    print(f"\r  {os.path.basename(path)}  {done / 1e6:9.1f} MB  done" + " " * 20)
    return path


def _extract(archive, dest):
    print(f"  extracting {os.path.basename(archive)} -> {dest}")
    if archive.endswith(".zip"):
        with zipfile.ZipFile(archive) as z:
            z.extractall(dest)
    else:
        import tarfile
        with tarfile.open(archive) as t:
            t.extractall(dest, filter="data")


def _ensure_gdown():
    try:
        import gdown  # noqa: F401
    except ImportError:
        print("  installing gdown (pip install gdown) ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "gdown"])


def _gdrive(file_id, out_path):
    _ensure_gdown()
    import gdown
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if os.path.exists(out_path) and zipfile.is_zipfile(out_path):
        print(f"  {out_path} already downloaded")
        return out_path
    r = gdown.download(id=file_id, output=out_path, quiet=False, resume=True)
    if not r or not os.path.exists(out_path) or not zipfile.is_zipfile(out_path):
        raise RuntimeError("Google Drive refused the download (quota exceeded or link changed)")
    return out_path


def dl_sully(a):
    root = os.path.join(a.dest, "sullychen")
    sets = ["2017", "2018"] if a.sully_set == "both" else [a.sully_set]
    for s in sets:
        info = SULLY_DRIVE[s]
        target = os.path.join(root, "driving_dataset" if s == "2017" else "driving_dataset_2018")
        if os.path.exists(os.path.join(target, "data.txt")):
            print(f"[download] SullyChen {s}: already present at {target}")
            continue
        if not _size_check(f"SullyChen {s} ({info['frames']} frames)", info["gb"] * 2.1, a.dest, a.yes):
            continue
        z = os.path.join(root, f"sully_{s}.zip")
        try:
            _gdrive(info["id"], z)
        except Exception as e:                       # noqa: BLE001
            print(f"\n[download] SullyChen {s} FAILED: {e}\n"
                  f"  Open this link in your browser and download the zip manually:\n    {info['url']}\n"
                  f"  Save it as {z} and re-run this command (it will extract it), or unzip it yourself\n"
                  f"  so that this file exists:  {os.path.join(target, 'data.txt')}")
            continue
        tmp = os.path.join(root, f"_extract_{s}")
        _extract(z, tmp)
        txt = sorted(glob.glob(os.path.join(tmp, "**", "data.txt"), recursive=True), key=len)
        if not txt:
            raise SystemExit(f"no data.txt inside {z}")
        if os.path.exists(target):
            shutil.rmtree(target)
        shutil.move(os.path.dirname(txt[0]), target)
        shutil.rmtree(tmp, ignore_errors=True)
        if not a.keep_zip:
            os.remove(z)
        n = sum(1 for _ in open(os.path.join(target, "data.txt")))
        print(f"[download] SullyChen {s}: {n} labelled frames -> {target}")
    print("  NOTE: SullyChen's README forbids using these datasets to pilot a real car (research only).")


def dl_udacity_sim(a):
    root = os.path.join(a.dest, "udacity_sim")
    exe = glob.glob(os.path.join(root, "**", "*.exe"), recursive=True)
    if not exe:
        if not _size_check("Udacity Term 1 simulator (Windows, v2)", 0.25, a.dest, a.yes):
            return
        z = _http_download(UDACITY_SIM_URL, os.path.join(root, "term1-simulator-windows.zip"))
        _extract(z, root)
        if not a.keep_zip:
            os.remove(z)
        exe = glob.glob(os.path.join(root, "**", "*.exe"), recursive=True)
    exe = [e for e in exe if "UnityCrashHandler" not in e]
    rec = os.path.join(a.dest, "udacity")
    print(f"""[download] Udacity simulator: {exe[0] if exe else root}

  Record your own training data (the simulator writes driving_log.csv + IMG/):
   1. Double-click {os.path.basename(exe[0]) if exe else 'the .exe'}; pick a screen resolution (640x480 is fine) and
      graphics quality "Fastest", then press Play.
   2. Choose the left track and press TRAINING MODE.
   3. Press R (or click the record button, top right) and choose the folder
         {rec}
      (create it first; the simulator puts driving_log.csv and IMG\\ in it).
   4. Press R again to start recording. Drive 3-5 laps in the centre of the road with
      mouse steering (smoother than the keyboard), 1-2 laps the other direction,
      plus a few "recovery" clips: stop recording, drift to the edge, start recording,
      steer back to the centre.
   5. Press R to stop; wait until the "capturing data" counter finishes.
   6. Train:  python pilotnet_all.py train --dataset udacity --root datasets\\udacity --balance --out runs\\udacity
""")


def dl_udacity_sample(a):
    root = os.path.join(a.dest, "udacity_sample")
    if os.path.exists(os.path.join(root, "data", "driving_log.csv")) and not os.path.exists(os.path.join(root, "data.zip")):
        print(f"[download] Udacity sample data already at {root}\\data")
        return
    if not _size_check("Udacity sample recording (track 1)", 0.7, a.dest, a.yes):
        return
    z = _http_download(UDACITY_SAMPLE_URL, os.path.join(root, "data.zip"))
    _extract(z, root)
    if not a.keep_zip:
        os.remove(z)
    n = sum(1 for _ in open(os.path.join(root, "data", "driving_log.csv"))) - 1
    print(f"[download] Udacity sample: {n} frames x 3 cameras -> {os.path.join(root, 'data')}")


def dl_comma(a):
    print(f"[download] comma.ai research 2016 driving data: 48.3 GB zip (45 GB compressed, 80 GB extracted)\n"
          f"  source: {COMMA_URL}\n  licence: CC BY-NC-SA 3.0 (non-commercial)")
    if not _size_check("comma.ai research", 48.3 + 80.0, a.dest, a.yes):
        return
    root = os.path.join(a.dest, "comma_research")
    z = _http_download(COMMA_URL, os.path.join(root, "comma-dataset.zip"))
    _extract(z, root)
    if not a.keep_zip:
        os.remove(z)


def dl_comma2k19(a):
    import urllib.request
    root = os.path.join(a.dest, "comma2k19")
    print(f"[download] comma2k19: ~100 GB in 10 chunks of ~10 GB, distributed ONLY by BitTorrent:\n"
          f"    {COMMA2K19_TORRENT}\n"
          f"  Use a torrent client (e.g. qBittorrent), select only the chunks you want, save them to\n"
          f"    {root}\n"
          f"  On Windows extract with comma's  utils/unzip_msft_fs.py  ('|' is not allowed in NTFS names).\n"
          f"  Downloading the official 1-minute sample segment (Example_1, ~52 MB) now ...")
    tree = json.load(urllib.request.urlopen(urllib.request.Request(
        COMMA2K19_SAMPLE_API, headers={"User-Agent": "pilotnet-abaja-downloader"}), timeout=60))["tree"]
    blobs = [t for t in tree if t["type"] == "blob" and t["path"].startswith("Example_1/")]
    files = [t["path"] for t in blobs]
    for t in blobs:
        p = t["path"]
        local = os.path.join(root, *p.replace("|", "_").split("/"))
        if os.path.exists(local) and os.path.getsize(local) != t["size"]:
            os.remove(local)                             # incomplete earlier attempt
        _http_download(COMMA2K19_SAMPLE_RAW + urllib.request.quote(p), local)
    print(f"[download] comma2k19 sample: {len(files)} files -> {os.path.join(root, 'Example_1')}")


def dl_tartandrive(a):
    import urllib.request
    names = [l.strip() for l in urllib.request.urlopen(TARTAN_FILELIST, timeout=60).read().decode().splitlines()
             if l.strip()][:max(1, a.tartan_files)]
    sizes = []
    for n in names:
        r = urllib.request.urlopen(urllib.request.Request(f"{TARTAN_ENDPOINT}/{n}", method="HEAD"), timeout=60)
        sizes.append(int(r.headers["Content-Length"]) / GB)
    print(f"[download] TartanDrive (rosbags, MIT licence): {len(names)} archive(s): " +
          ", ".join(f"{n} ({s:.0f} GB)" for n, s in zip(names, sizes)) +
          "\n  full dataset: 23 archives, ~2.06 TB")
    if not _size_check("TartanDrive", sum(sizes) * 2, a.dest, a.yes):
        return
    root = os.path.join(a.dest, "tartandrive")
    for n in names:
        z = _http_download(f"{TARTAN_ENDPOINT}/{n}", os.path.join(root, n))
        _extract(z, root)
        if not a.keep_zip:
            os.remove(z)


def dl_rellis(a):
    print(f"[download] RELLIS-3D (CC BY-NC-SA 3.0). It has NO steering labels (only a cmd_vel Twist in the\n"
          f"  ROS bags), so it is not usable for PilotNet training here - only for off-road scene images.\n"
          f"  Full data (Drive folder): {RELLIS_FOLDER}\n  Images only: {RELLIS_IMAGES['url']} (11 GB)")
    if not _size_check("RELLIS-3D images", RELLIS_IMAGES["gb"] * 2, a.dest, a.yes):
        return
    root = os.path.join(a.dest, "rellis3d")
    z = os.path.join(root, "Rellis_3D_pylon_camera_node.zip")
    try:
        _gdrive(RELLIS_IMAGES["id"], z)
    except Exception as e:                           # noqa: BLE001
        print(f"[download] RELLIS-3D FAILED: {e}\n  Open {RELLIS_IMAGES['url']} in a browser and unzip into {root}")
        return
    _extract(z, root)


def cmd_download(a):
    os.makedirs(a.dest, exist_ok=True)
    print(f"[download] destination {a.dest}  (free {_free_gb(a.dest):.1f} GB)")
    jobs = dict(sully=dl_sully, udacity_sim=dl_udacity_sim, udacity_sample=dl_udacity_sample, comma=dl_comma,
                comma2k19=dl_comma2k19, tartandrive=dl_tartandrive, rellis=dl_rellis)
    for name in (list(jobs) if a.what == "all" else [a.what]):
        jobs[name](a)


# ============================================================================
# 18. Dataset conversion  (`convert`) -> labels.csv + images/  (read with --dataset sim)
# ============================================================================
# Output format = the simulator format:
#   dst/images/000000.png ...  66x200 RGB (INTER_AREA resize of the cropped frame)
#   dst/labels.csv  filename,target,command,raw,t,source
#       command = 0 (FOLLOW) for every frame (no navigation commands in these logs)
#       raw     = the original steering value before normalisation
#   dst/dataset_meta.json  documents the sign convention and normalisation.
# Train with  --dataset sim --branches 1. Note --dataset sim treats +target as
# LEFT only for the shift augmentation, which is disabled for sim (shift_k = 0),
# so the sign convention does not change training; flips negate the target.
CONVERT_DOC = {
    "comma": dict(
        target="radians(steering_angle)  (steering-WHEEL angle in degrees -> radians, same as the SullyChen reader)",
        sign="as recorded; commaai/research Logs.md does not document it. Use --flip-sign to negate.",
        filter="|steering_angle| > 200 deg dropped (as in comma's dask_generator.py)",
        align="camera frame f <- first log sample whose cam1_ptr == f (cam1_ptr = camera frame index)"),
    "comma2k19": dict(
        target="radians(processed_log/CAN/steering_angle)  (steering-wheel angle in degrees -> radians)",
        sign="as recorded; checked on the Example_1 sample against the IMU yaw rate (see sign_check)",
        filter="none",
        align="steering linearly interpolated at global_pose/frame_times"),
    "tartandrive": dict(
        target="steer = /cmd TwistStamped twist.angular.z (dataset's own action[1]; throttle = linear.x)",
        sign="as recorded; units/normalisation are not documented by TartanDrive. Use --flip-sign to negate.",
        filter="none",
        align="each camera image <- latest /cmd at or before the image time (zero-order hold)"),
}


def add_convert_args(ap):
    ap.add_argument("--source", choices=list(CONVERT_DOC), required=True)
    ap.add_argument("--src", required=True, help="downloaded dataset folder (or a single file)")
    ap.add_argument("--dst", required=True)
    ap.add_argument("--every", type=int, default=1, help="keep every N-th frame")
    ap.add_argument("--max-frames", type=int, default=0, help="stop after N frames (0 = all)")
    ap.add_argument("--crop-top", type=float, default=0.0, help="drop this fraction of rows at the top (sky)")
    ap.add_argument("--crop-bottom", type=float, default=1.0, help="keep rows above this fraction (bonnet)")
    ap.add_argument("--flip-sign", action="store_true", help="negate the steering target")
    ap.add_argument("--image-topic", default="/multisense/left/image_rect_color", help="TartanDrive bags")
    ap.add_argument("--cmd-topic", default="/cmd", help="TartanDrive bags")


class _Writer:
    def __init__(self, a):
        self.a, self.n, self.rows, self.seen = a, 0, [], 0
        os.makedirs(os.path.join(a.dst, "images"), exist_ok=True)

    def full(self):
        return bool(self.a.max_frames) and self.n >= self.a.max_frames

    def add(self, img_rgb, target, raw, t, src):
        """img_rgb: HxWx3 uint8 RGB at any size."""
        k = self.seen
        self.seen += 1
        if k % self.a.every or self.full() or not np.isfinite(target):
            return
        h = img_rgb.shape[0]
        img = img_rgb[int(round(self.a.crop_top * h)):int(round(self.a.crop_bottom * h))]
        img = cv2.resize(img, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
        fn = f"{self.n:06d}.png"
        cv2.imwrite(os.path.join(self.a.dst, "images", fn), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        sgn = -1.0 if self.a.flip_sign else 1.0
        self.rows.append([fn, sgn * float(target), 0, float(raw), float(t), src])
        self.n += 1
        if self.n % 2000 == 0:
            print(f"  {self.n} frames ...", flush=True)

    def close(self, extra=None):
        with open(os.path.join(self.a.dst, "labels.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["filename", "target", "command", "raw", "t", "source"])
            w.writerows(self.rows)
        meta = dict(source=self.a.source, src=self.a.src, n_images=self.n, img_hw=[IMG_H, IMG_W],
                    crop=[self.a.crop_top, self.a.crop_bottom], every=self.a.every, flip_sign=self.a.flip_sign,
                    command="0 (FOLLOW) for every frame", **CONVERT_DOC[self.a.source], **(extra or {}))
        with open(os.path.join(self.a.dst, "dataset_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        t = np.array([r[1] for r in self.rows]) if self.rows else np.zeros(1)
        print(f"[convert] {self.n} frames -> {self.a.dst}  target mean={t.mean():+.4f} std={t.std():.4f} "
              f"min={t.min():+.3f} max={t.max():+.3f}")


def convert_comma(a, wr):
    import h5py
    cams = sorted(glob.glob(os.path.join(a.src, "**", "camera", "*.h5"), recursive=True))
    if not cams:
        raise SystemExit(f"no camera/*.h5 under {a.src} (expected commaai/research layout dataset/camera, dataset/log)")
    for cam_f in cams:
        log_f = os.path.join(os.path.dirname(os.path.dirname(cam_f)), "log", os.path.basename(cam_f))
        with h5py.File(cam_f, "r") as c5, h5py.File(log_f, "r") as l5:
            X = c5["X"]                                   # N x 3 x 160 x 320 uint8 RGB
            ptr = l5["cam1_ptr"][:].astype(np.int64)
            ang = l5["steering_angle"][:].astype(float)   # degrees (steering wheel)
            tt = l5["times"][:] if "times" in l5 else np.arange(len(ang)) / 100.0
            order = np.searchsorted(ptr, np.arange(X.shape[0]), side="left")
            print(f"  {os.path.basename(cam_f)}: {X.shape[0]} frames")
            for f_i in range(X.shape[0]):
                j = order[f_i]
                if j >= len(ptr) or ptr[j] != f_i or abs(ang[j]) > 200:
                    wr.seen += 1
                    continue
                wr.add(np.ascontiguousarray(X[f_i].transpose(1, 2, 0)), math.radians(ang[j]), ang[j],
                       tt[j], os.path.basename(cam_f))
                if wr.full():
                    return {}
    return {}


def _np_load(path):
    return np.load(path, allow_pickle=False)


def convert_comma2k19(a, wr):
    segs = sorted({os.path.dirname(os.path.dirname(p)) for p in
                   glob.glob(os.path.join(a.src, "**", "global_pose", "frame_times"), recursive=True)})
    if not segs:
        raise SystemExit(f"no <segment>/global_pose/frame_times under {a.src}")
    corr = []
    for seg in segs:
        ft = _np_load(os.path.join(seg, "global_pose", "frame_times"))
        st = _np_load(os.path.join(seg, "processed_log", "CAN", "steering_angle", "t"))
        sv = _np_load(os.path.join(seg, "processed_log", "CAN", "steering_angle", "value")).astype(float)
        gyro_t = os.path.join(seg, "processed_log", "IMU", "gyro", "t")
        if os.path.exists(gyro_t):           # sign check: IMU frame is [forward, right, down]
            gz = _np_load(os.path.join(seg, "processed_log", "IMU", "gyro", "value"))[:, 2]
            g_at = np.interp(st, _np_load(gyro_t), gz)
            m = np.abs(sv) > 2
            if m.sum() > 20:
                corr.append(float(np.corrcoef(sv[m], g_at[m])[0, 1]))
        ang = np.interp(ft, st, sv)
        cap = cv2.VideoCapture(os.path.join(seg, "video.hevc"))
        rel = os.path.relpath(seg, a.src)
        k = 0
        while k < len(ft):
            ok, bgr = cap.read()
            if not ok:
                break
            wr.add(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), math.radians(ang[k]), ang[k], ft[k], rel)
            k += 1
            if wr.full():
                break
        cap.release()
        print(f"  {rel}: {k}/{len(ft)} frames decoded")
        if k == 0:
            raise SystemExit("could not decode video.hevc - your OpenCV build lacks HEVC; pip install opencv-python")
        if wr.full():
            break
    extra = {}
    if corr:
        c = float(np.mean(corr))
        extra["sign_check"] = (f"corr(steering_angle, IMU yaw rate about the DOWN axis) = {c:+.2f}; "
                               + ("negative -> + steering = LEFT (counter-clockwise)" if c < 0 else
                                  "positive -> + steering = RIGHT"))
        print(f"  sign check: {extra['sign_check']}")
    return extra


def convert_tartandrive(a, wr):
    pts = sorted(glob.glob(os.path.join(a.src, "**", "*.pt"), recursive=True)) if os.path.isdir(a.src) else \
        ([a.src] if a.src.endswith(".pt") else [])
    bags = sorted(glob.glob(os.path.join(a.src, "**", "*.bag"), recursive=True)) if os.path.isdir(a.src) else \
        ([a.src] if a.src.endswith(".bag") else [])
    if pts:                                  # torch trajectories from tartan_drive multi_convert_bag.py
        for p in pts:
            d = torch.load(p, map_location="cpu", weights_only=False)
            img = d["observation"]["image_rgb"]          # T x 3 x H x W float 0..1
            act = d["action"]                            # T x 2 = [throttle, steer]
            img = img.numpy() if torch.is_tensor(img) else np.asarray(img)
            act = act.numpy() if torch.is_tensor(act) else np.asarray(act)
            print(f"  {os.path.basename(p)}: {len(act)} steps, image {img.shape[2]}x{img.shape[3]} "
                  "(upsampled to 66x200 - prefer the raw .bag files)")
            for k in range(len(act)):
                rgb = np.clip(img[k].transpose(1, 2, 0) * 255 + 0.5, 0, 255).astype(np.uint8)
                wr.add(rgb, act[k, 1], act[k, 1], k * 0.1, os.path.basename(p))
                if wr.full():
                    return {}
        return {}
    if not bags:
        raise SystemExit(f"no .bag or .pt files under {a.src}")
    try:
        from rosbags.highlevel import AnyReader
    except ImportError:
        raise SystemExit("reading TartanDrive .bag files needs:  pip install rosbags")
    from pathlib import Path
    for b in bags:
        with AnyReader([Path(b)]) as r:
            conns = [c for c in r.connections if c.topic in (a.image_topic, a.cmd_topic)]
            if not any(c.topic == a.cmd_topic for c in conns) or not any(c.topic == a.image_topic for c in conns):
                print(f"  {os.path.basename(b)}: topics {a.image_topic} / {a.cmd_topic} missing - skipped")
                continue
            steer, n0 = None, wr.n
            for c, t_ns, raw in r.messages(connections=conns):
                msg = r.deserialize(raw, c.msgtype)
                if c.topic == a.cmd_topic:
                    tw = msg.twist if hasattr(msg, "twist") else msg
                    steer = float(tw.angular.z)
                    continue
                if steer is None:
                    continue
                ch = 3 if msg.encoding.lower() in ("rgb8", "bgr8") else 4
                im = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)[:, :msg.width * ch]
                im = im.reshape(msg.height, msg.width, ch)[..., :3]
                if msg.encoding.lower().startswith("bgr"):
                    im = im[..., ::-1]
                wr.add(np.ascontiguousarray(im), steer, steer, t_ns * 1e-9, os.path.basename(b))
                if wr.full():
                    return {}
            print(f"  {os.path.basename(b)}: {wr.n - n0} frames")
    return {}


def cmd_convert(a):
    print(f"[convert] {a.source}: {a.src} -> {a.dst}")
    wr = _Writer(a)
    extra = dict(comma=convert_comma, comma2k19=convert_comma2k19, tartandrive=convert_tartandrive)[a.source](a, wr)
    wr.close(extra)


# ============================================================================
# 19. Unit tests  (was tests/test_pipeline.py, plus new ones)  -> `selftest`
# ============================================================================
try:
    import pytest
    approx = pytest.approx
except ImportError:                     # pytest is optional for `selftest`
    class approx:                       # noqa: N801
        def __init__(self, v, rel=1e-6, abs=1e-12):
            self.v, self.rel, self.abs = v, rel, abs

        def __eq__(self, o):
            return math.isclose(o, self.v, rel_tol=self.rel, abs_tol=self.abs)


def test_pilotnet_shapes_and_params():
    m = PilotNet()
    assert m.flat_dim == 1152
    assert 250_000 < count_parameters(m) < 255_000          # "about 250 thousand parameters"
    assert m(torch.rand(4, 3, 66, 200) * 255).shape == (4, 1)
    m3 = PilotNet(n_branches=3)
    out = m3(torch.rand(4, 3, 66, 200) * 255)
    sel = select_branch(out, torch.tensor([0, 1, 2, 1]))
    assert torch.allclose(sel, out[torch.arange(4), torch.tensor([0, 1, 2, 1])])


def test_yuv_layer_matches_formula():
    m = PilotNet()
    x = torch.tensor([255.0, 0.0, 0.0]).view(1, 3, 1, 1)        # pure red
    yuv = (m.norm(x) + 1) * 127.5
    assert torch.allclose(yuv.flatten(), torch.tensor([0.299 * 255, -0.14713 * 255 + 128, 0.615 * 255 + 128]),
                          atol=1e-3)


def test_vbp_mask_shape_and_range():
    m = PilotNet()
    mask = visual_backprop(m, torch.rand(2, 3, 66, 200) * 255)
    assert mask.shape == (2, 66, 200)
    assert float(mask.min()) >= 0 and float(mask.max()) <= 1 + 1e-6


def test_flip_negates_and_swaps_commands():
    img = np.arange(66 * 200 * 3, dtype=np.uint8).reshape(66, 200, 3)
    out, t, c = flip(img, 0.3, 1)
    assert t == -0.3 and c == 2 and np.array_equal(out[:, 0], img[:, -1])


def test_block_split_has_no_overlap_and_is_blocky():
    sp = block_split(10_000, 500, 0.1, 0.1)
    a, b, c = map(set, (sp["train"], sp["val"], sp["test"]))
    assert not (a & b or a & c or b & c) and len(a | b | c) == 10_000
    assert all((i // 500) == (sp["test"][0] // 500) for i in sp["test"][:500])


def test_astar_and_commands():
    course = build_course()
    names = [course.edges[e]["to"] for e in astar(course, "S", "B")]
    assert names == ["J1", "B"]
    r = build_route(course, ["S", "A", "S", "B", "S"])
    assert [f["cmd"] for f in r.forks] == [C.LEFT, C.RIGHT]
    assert np.all(np.diff(r.s) > 0)


def test_bicycle_turning_radius():
    st = [0.0, 0.0, 0.0, C.MAX_STEER]
    for _ in range(int(2 * math.pi * (C.WHEELBASE / math.tan(C.MAX_STEER)) / C.SPEED / C.DT)):
        st = bicycle_step(st, C.MAX_STEER, C.SPEED)
    assert math.hypot(st[0], st[1]) < 0.5                    # closed a full circle


def test_camera_geometry():
    cam = camera_precompute()
    # bottom-centre pixel looks at the ground just ahead of the camera
    pf = cam["pf"][-1, cam["pf"].shape[1] // 2]
    pitch_bottom = C.CAM_PITCH + math.atan((C.IMG_H / 2) / ((C.IMG_W / 2) / math.tan(C.CAM_HFOV / 2)))
    assert pf == approx(C.CAM_X + C.CAM_HEIGHT / math.tan(pitch_bottom), rel=0.03)
    assert not cam["valid"][0].any()                          # top row is sky


def test_metrics():
    r = regression_metrics([0, 1, 2], [0, 1, 2])
    assert r["rmse"] == 0 and r["r2"] == approx(1)
    assert autonomy(10, 600) == approx(90.0)           # NVIDIA's worked example


# ---- new tests -------------------------------------------------------------
REF_DIR = os.path.join(ROOT, "matlab", "reference")
DEFAULT_ONNX = os.path.join(ROOT, "runs", "sim", "pilotnet.onnx")


def test_renderer_matches_reference_pngs():
    """Byte-identical to the stored Python/MATLAB reference renders."""
    ref = json.load(open(os.path.join(REF_DIR, "reference.json")))
    course = build_course()
    tex, _ = build_texture(course)
    cam = camera_precompute()
    for p in ref["poses"]:
        img = render(cam, tex, p["x"], p["y"], p["psi"])
        want = cv2.cvtColor(cv2.imread(os.path.join(REF_DIR, p["file"])), cv2.COLOR_BGR2RGB)
        assert np.array_equal(img, want), p["file"]
    r = build_route(course, ["S", "A", "S", "B", "S"])
    assert abs(r.s[-1] - ref["route_length"]) < 1e-9


def test_cnn_matches_reference_outputs():
    """ONNX outputs on the reference images equal the stored ones (< 1e-6)."""
    if not os.path.exists(DEFAULT_ONNX):
        return "skipped (no runs/sim/pilotnet.onnx)"
    ref = json.load(open(os.path.join(REF_DIR, "reference.json")))
    net = PilotNetONNX(DEFAULT_ONNX)
    for p in ref["poses"]:
        if "cnn" not in p:
            continue
        img = cv2.cvtColor(cv2.imread(os.path.join(REF_DIR, p["file"])), cv2.COLOR_BGR2RGB)
        assert np.max(np.abs(net.predict(img) - np.array(p["cnn"]))) < 1e-6, p["file"]


def test_pytorch_matches_onnx():
    ck = os.path.join(ROOT, "runs", "sim", "best.pt")
    if not (os.path.exists(ck) and os.path.exists(DEFAULT_ONNX)):
        return "skipped (no runs/sim/best.pt)"
    model, _ = load_model(ck)
    x = (np.random.default_rng(1).random((4, 3, 66, 200)) * 255).astype(np.float32)
    with torch.no_grad():
        y_pt = model(torch.from_numpy(x)).numpy()
    y_ort = PilotNetONNX(DEFAULT_ONNX).sess.run(None, {"image": x})[0]
    assert np.max(np.abs(y_pt - y_ort)) < 1e-4


def test_closed_loop_expert_short():
    res = closed_loop_run("expert", cycles=1)[0]
    assert res["interventions"] == 0 and res["forks_correct"] == res["forks_total"] == 2
    assert res["cte_mean_abs"] < 0.1


def _tmpdir(name):
    import tempfile
    d = os.path.join(tempfile.gettempdir(), "pilotnet_selftest", name)
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    return d


def _convert_ns(**kw):
    base = dict(every=1, max_frames=0, crop_top=0.0, crop_bottom=1.0, flip_sign=False,
                image_topic="/multisense/left/image_rect_color", cmd_topic="/cmd")
    base.update(kw)
    return argparse.Namespace(**base)


def test_convert_comma_synthetic():
    try:
        import h5py
    except ImportError:
        return "skipped (pip install h5py)"
    d = _tmpdir("comma")
    os.makedirs(os.path.join(d, "dataset", "camera"))
    os.makedirs(os.path.join(d, "dataset", "log"))
    n = 6
    with h5py.File(os.path.join(d, "dataset", "camera", "r.h5"), "w") as f:
        f["X"] = np.random.default_rng(0).integers(0, 255, (n, 3, 160, 320), dtype=np.uint8)
    ptr = np.repeat(np.arange(n), 5)                    # 100 Hz log, 20 Hz camera
    ang = np.linspace(-30, 30, len(ptr))
    ang[ptr == 3] = 500                                  # out-of-range frame -> dropped
    with h5py.File(os.path.join(d, "dataset", "log", "r.h5"), "w") as f:
        f["cam1_ptr"], f["steering_angle"], f["times"] = ptr.astype(float), ang, np.arange(len(ptr)) / 100
    out = os.path.join(d, "out")
    cmd_convert(_convert_ns(source="comma", src=d, dst=out))
    samples, info = read_sim(out)
    assert len(samples) == n - 1 and all(s.command == 0 for s in samples)
    assert samples[0].target == approx(math.radians(ang[0]))
    assert load_image(samples[0].path, info.crop).shape == (66, 200, 3)


def test_convert_tartandrive_pt_synthetic():
    d = _tmpdir("tartan")
    T = 5
    torch.save(dict(observation=dict(image_rgb=torch.rand(T, 3, 64, 64)),
                    action=torch.tensor([[0.5, s] for s in np.linspace(-1, 1, T)], dtype=torch.float32)),
               os.path.join(d, "traj.pt"))
    out = os.path.join(d, "out")
    cmd_convert(_convert_ns(source="tartandrive", src=d, dst=out, flip_sign=True))
    samples, _ = read_sim(out)
    assert len(samples) == T and samples[0].target == approx(1.0) and samples[-1].target == approx(-1.0)


def test_convert_tartandrive_bag_synthetic():
    """A tiny ROS1 bag with TartanDrive's topics/types (bgr8 Image + TwistStamped /cmd)."""
    try:
        from rosbags.rosbag1 import Writer
        from rosbags.typesys import Stores, get_typestore
    except ImportError:
        return "skipped (pip install rosbags)"
    ts = get_typestore(Stores.ROS1_NOETIC)
    T = ts.types
    Time, Header = T["builtin_interfaces/msg/Time"], T["std_msgs/msg/Header"]
    Vec, Tw, TwS, Img = (T["geometry_msgs/msg/Vector3"], T["geometry_msgs/msg/Twist"],
                         T["geometry_msgs/msg/TwistStamped"], T["sensor_msgs/msg/Image"])
    d = _tmpdir("tartan_bag")
    bag = os.path.join(d, "run.bag")
    steers = [0.25, -0.5, 0.75]
    with Writer(bag) as w:
        ci = w.add_connection("/multisense/left/image_rect_color", Img.__msgtype__, typestore=ts)
        cc = w.add_connection("/cmd", TwS.__msgtype__, typestore=ts)
        for k, s in enumerate(steers):
            t = 10 ** 9 * (k + 1)
            hd = Header(seq=k, stamp=Time(sec=k + 1, nanosec=0), frame_id="")
            w.write(cc, t, ts.serialize_ros1(TwS(header=hd, twist=Tw(linear=Vec(x=0.3, y=0.0, z=0.0),
                                                                         angular=Vec(x=0.0, y=0.0, z=s))),
                                             TwS.__msgtype__))
            bgr = np.zeros((544, 1024, 3), np.uint8)
            bgr[..., 0] = 255                                   # pure blue in BGR order
            w.write(ci, t + 1000, ts.serialize_ros1(Img(header=hd, height=544, width=1024, encoding="bgr8",
                                                        is_bigendian=0, step=1024 * 3, data=bgr.reshape(-1)),
                                                    Img.__msgtype__))
    out = os.path.join(d, "out")
    cmd_convert(_convert_ns(source="tartandrive", src=d, dst=out))
    samples, info = read_sim(out)
    assert [s.target for s in samples] == approx(steers)
    img = load_image(samples[0].path, info.crop)
    assert img.shape == (66, 200, 3) and img[..., 2].min() == 255 and img[..., 0].max() == 0   # RGB blue


def test_convert_comma2k19_sample():
    src = os.path.join(ROOT, "datasets", "comma2k19", "Example_1")
    if not os.path.isdir(src) or glob.glob(os.path.join(src, "**", "*.part"), recursive=True):
        return "skipped (sample missing or incomplete: python pilotnet_all.py download --what comma2k19)"
    out = _tmpdir("c2k19")
    cmd_convert(_convert_ns(source="comma2k19", src=src, dst=out, max_frames=40))
    samples, info = read_sim(out)
    assert len(samples) == 40
    assert load_image(samples[-1].path, info.crop).shape == (66, 200, 3)


def cmd_selftest(a):
    tests = [(k, v) for k, v in globals().items() if k.startswith("test_") and callable(v)]
    if a.k:
        tests = [(k, v) for k, v in tests if a.k in k]
    n_fail = 0
    t_all = time.time()
    for name, fn in tests:
        t0 = time.time()
        try:
            note = fn()
            status = "PASS" + (f"  ({note})" if isinstance(note, str) else "")
        except Exception as e:                      # noqa: BLE001
            n_fail += 1
            status = f"FAIL  {type(e).__name__}: {e}"
        print(f"  {status.split()[0]:4s}  {name:45s} {time.time() - t0:6.2f}s  {' '.join(status.split()[1:])}",
              flush=True)
    print(f"[selftest] {len(tests) - n_fail}/{len(tests)} passed in {time.time() - t_all:.1f}s -> "
          f"{'PASS' if n_fail == 0 else 'FAIL'}")
    if n_fail:
        raise SystemExit(1)


# ============================================================================
# 20. Command-line interface
# ============================================================================
PATH_ARGS = ("root", "out", "ckpt", "onnx", "dest", "src", "dst")


def build_parser():
    ap = argparse.ArgumentParser(prog="pilotnet_all.py", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True, metavar="COMMAND")
    p = sp.add_parser("selftest", help="run all unit tests, print PASS/FAIL")
    p.add_argument("-k", default="", help="only tests whose name contains this")
    add_download_args(sp.add_parser("download", help="download datasets into datasets/"))
    add_gen_args(sp.add_parser("gen", help="generate a simulator dataset"))
    add_train_args(sp.add_parser("train", help="train PilotNet"))
    add_evaluate_args(sp.add_parser("evaluate", help="open-loop evaluation + VisualBackProp"))
    add_export_args(sp.add_parser("export", help="export ONNX + meta + MATLAB .mat weights"))
    add_reference_args(sp.add_parser("reference", help="write matlab/reference/ check data"))
    add_closedloop_args(sp.add_parser("closedloop", help="closed-loop simulation"))
    add_convert_args(sp.add_parser("convert", help="convert a downloaded dataset to labels.csv + images/"))
    return ap


COMMANDS_CLI = dict(selftest="cmd_selftest", download="cmd_download", gen="cmd_gen", train="cmd_train",
                    evaluate="cmd_evaluate", export="cmd_export", reference="cmd_reference",
                    closedloop="cmd_closedloop", convert="cmd_convert")

WORKERS_ADVICE = """
[hint] A DataLoader worker failed. On Windows, workers are started with 'spawn' and
       re-import this file; that can fail through low virtual memory ("paging file is too
       small", WinError 1455), antivirus locks or a broken pipe. Re-run with  --workers 0
       (loads images in the main process: slower, but always works), or try --workers 2.
"""


def main(argv=None):
    a = build_parser().parse_args(argv)
    cmd = a.cmd
    del a.cmd                                   # keep vars(a) identical to the old scripts
    for k in PATH_ARGS:
        if isinstance(getattr(a, k, None), str):
            setattr(a, k, _abs(getattr(a, k)))
    fn = globals()[COMMANDS_CLI[cmd]]
    try:
        fn(a)
    except (RuntimeError, OSError, EOFError, BrokenPipeError) as e:
        msg = f"{type(e).__name__} {e}".lower()
        if getattr(a, "workers", 0) > 0 and cmd in ("train", "evaluate") and any(
                k in msg for k in ("worker", "dataloader", "pickl", "1455", "paging file", "pipe", "spawn",
                                   "eoferror", "shared memory")):
            print(WORKERS_ADVICE, file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
