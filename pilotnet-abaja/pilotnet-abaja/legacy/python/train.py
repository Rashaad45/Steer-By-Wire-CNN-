"""Train PilotNet (optionally command-conditioned) for steering regression.

Examples
--------
# Udacity simulator recording (github.com/udacity/self-driving-car-sim)
python train.py --dataset udacity --root data/udacity --out runs/udacity

# SullyChen 2017 dataset (github.com/SullyChen/driving-datasets)
python train.py --dataset sully --root data/driving_dataset --out runs/sully --cache

# aBAJA simulator data (MATLAB generate_dataset.m or python -m sim.gen_dataset)
python train.py --dataset sim --root data/sim --branches 3 --balance --out runs/sim
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from pilotnet.data import READERS, SteeringDataset, balance_weights, block_split
from pilotnet.model import PilotNet, count_parameters, select_branch


def get_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
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
    return ap.parse_args(argv)


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


def main(argv=None):
    a = get_args(argv)
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


if __name__ == "__main__":
    main()
