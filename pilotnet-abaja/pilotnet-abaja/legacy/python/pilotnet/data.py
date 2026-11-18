"""Dataset readers for the three supported sources.

sully   : SullyChen/driving-datasets (GitHub). data.txt lines
          "filename.jpg angle" (2017 set) or "filename.jpg angle,date time"
          (2018 set). angle = steering-wheel angle in degrees. Target =
          radians(angle), as in SullyChen/Autopilot-TensorFlow.
udacity : Udacity self-driving-car-sim (GitHub) recordings, driving_log.csv
          with center,left,right,steering,throttle,brake,speed. Steering is
          already normalised to [-1, 1] (x25 deg), + = right. Side cameras are
          used with a +-0.2 correction, the standard 3-camera trick from
          NVIDIA's paper.
sim     : this project's simulator (MATLAB or Python), labels.csv with
          filename,target,command,...; target = kappa / kappa_max, + = left.

Splits are made over *contiguous blocks* of frames, never per-frame: adjacent
video frames are near-duplicates, so a random per-frame split leaks training
images into the test set and inflates every metric.
"""
from __future__ import annotations

import csv
import math
import os
from dataclasses import dataclass

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .augment import augment

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
