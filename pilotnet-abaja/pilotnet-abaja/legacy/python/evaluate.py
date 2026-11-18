"""Open-loop evaluation on the held-out block split + VisualBackProp + latency.

python evaluate.py --ckpt runs/sim/best.pt --root data/sim
Writes <ckpt dir>/eval/{metrics.json, predictions.csv, *.png}
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from export_onnx import load_model
from pilotnet.data import READERS, SteeringDataset
from pilotnet.metrics import regression_metrics, smoothness
from pilotnet.model import COMMANDS, select_branch
from pilotnet.vbp import visual_backprop


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--vbp", type=int, default=8, help="number of VisualBackProp examples")
    ap.add_argument("--fps", type=float, default=10.0, help="frame rate for smoothness metric")
    a = ap.parse_args(argv)
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
    dl = DataLoader(SteeringDataset(sel, info, train=False), 256, shuffle=False, num_workers=2)

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


if __name__ == "__main__":
    main()
