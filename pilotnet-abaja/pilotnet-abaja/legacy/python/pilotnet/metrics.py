"""Open-loop (offline) and closed-loop metrics."""
from __future__ import annotations

import numpy as np


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
