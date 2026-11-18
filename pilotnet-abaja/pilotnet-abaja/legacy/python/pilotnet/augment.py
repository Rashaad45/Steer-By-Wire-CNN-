"""Training-time augmentation on uint8 RGB images of shape (66, 200, 3).

Geometric augmentations change the label consistently:
  * horizontal flip  -> target = -target, LEFT <-> RIGHT command swap
  * horizontal shift -> target += sign * shift_px * k  (recovery heuristic used
    for the Udacity data; the simulator does not need it because it renders
    physically perturbed poses with exact expert labels).
Photometric augmentations leave the label unchanged.
"""
from __future__ import annotations

import cv2
import numpy as np

from .model import COMMANDS

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
