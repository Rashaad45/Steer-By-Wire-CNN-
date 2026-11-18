"""Export a trained checkpoint to ONNX (+ metadata) and verify it numerically.

python export_onnx.py --ckpt runs/sim/best.pt --out runs/sim/pilotnet.onnx

ONNX graph: input  "image"  float32 (N, 3, 66, 200), RGB 0..255
            output "steer"  float32 (N, n_branches)
The RGB->YUV conversion and normalisation are inside the graph.
"""
from __future__ import annotations

import argparse
import json
import os
import warnings

import numpy as np
import torch

from pilotnet.model import COMMANDS, PilotNet


def load_model(ckpt_path: str, device="cpu") -> tuple[PilotNet, dict]:
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ck["cfg"]
    m = PilotNet(n_branches=cfg["n_branches"], dropout=cfg.get("dropout", 0.2))
    m.load_state_dict(ck["state_dict"])
    return m.eval().to(device), cfg


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--opset", type=int, default=17)
    a = ap.parse_args(argv)
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
    print(f"exported {mat}  (for matlab/pilotnet_native_*.m)")


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


if __name__ == "__main__":
    main()
