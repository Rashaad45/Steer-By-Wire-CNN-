"""Minimal ONNX Runtime wrapper used by the Python closed-loop simulator and by
MATLAB (through MATLAB's Python interface, see matlab/cnn_load.m).

From MATLAB:
    mod = py.importlib.import_module('pilotnet_infer');
    net = mod.PilotNetONNX('pilotnet.onnx');
    buf = reshape(permute(img,[3 2 1]),1,[]);     % HxWx3 uint8 -> row-major bytes
    y   = cellfun(@double, cell(net.predict_bytes(py.bytes(buf), int32(66), int32(200))));
"""
from __future__ import annotations

import numpy as np


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
