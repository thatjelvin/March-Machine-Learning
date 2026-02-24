"""
Probability calibration methods:
  - Isotonic regression
  - Platt scaling (logistic on logits)
  - Temperature scaling (single parameter)

Selection by cross-validated Brier score.
"""

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from typing import Tuple, Optional


def _brier(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


# ═══════════════════════════════════════════════════════════════════════════════
#  Isotonic calibration
# ═══════════════════════════════════════════════════════════════════════════════

def fit_isotonic(y_true, y_pred):
    """Fit isotonic regression calibrator."""
    ir = IsotonicRegression(out_of_bounds="clip", y_min=0.001, y_max=0.999)
    ir.fit(y_pred, y_true)
    return ir


def apply_isotonic(ir, y_pred):
    return ir.predict(y_pred)


# ═══════════════════════════════════════════════════════════════════════════════
#  Platt scaling
# ═══════════════════════════════════════════════════════════════════════════════

def fit_platt(y_true, y_pred):
    """Fit Platt scaling (logistic regression on raw predictions)."""
    # Clip to avoid log(0)
    eps = 1e-7
    y_pred_clip = np.clip(y_pred, eps, 1 - eps)
    logits = np.log(y_pred_clip / (1 - y_pred_clip)).reshape(-1, 1)
    lr = LogisticRegression(C=1e10, solver="lbfgs", max_iter=5000)
    lr.fit(logits, y_true)
    return lr


def apply_platt(lr, y_pred):
    eps = 1e-7
    y_pred_clip = np.clip(y_pred, eps, 1 - eps)
    logits = np.log(y_pred_clip / (1 - y_pred_clip)).reshape(-1, 1)
    return lr.predict_proba(logits)[:, 1]


# ═══════════════════════════════════════════════════════════════════════════════
#  Temperature scaling
# ═══════════════════════════════════════════════════════════════════════════════

def fit_temperature(y_true, y_pred):
    """Find optimal temperature T that minimizes Brier score."""
    eps = 1e-7
    y_pred_clip = np.clip(y_pred, eps, 1 - eps)
    logits = np.log(y_pred_clip / (1 - y_pred_clip))

    def objective(T):
        scaled = 1.0 / (1.0 + np.exp(-logits / T))
        return _brier(y_true, scaled)

    result = minimize_scalar(objective, bounds=(0.1, 10.0), method="bounded")
    return result.x  # optimal temperature


def apply_temperature(T, y_pred):
    eps = 1e-7
    y_pred_clip = np.clip(y_pred, eps, 1 - eps)
    logits = np.log(y_pred_clip / (1 - y_pred_clip))
    return 1.0 / (1.0 + np.exp(-logits / T))


# ═══════════════════════════════════════════════════════════════════════════════
#  Auto-select best calibration method
# ═══════════════════════════════════════════════════════════════════════════════

def select_calibration(y_true, y_pred, n_folds: int = 5) -> Tuple[str, object]:
    """
    Cross-validate each calibration method and pick the one with
    lowest average Brier score.

    Returns (method_name, fitted_calibrator) trained on ALL data.
    """
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    results = {"none": [], "isotonic": [], "platt": [], "temperature": []}

    for train_idx, val_idx in kf.split(y_pred):
        yt = y_true[train_idx]
        yp_tr = y_pred[train_idx]
        yp_val = y_pred[val_idx]
        yt_val = y_true[val_idx]

        # No calibration
        results["none"].append(_brier(yt_val, yp_val))

        # Isotonic
        ir = fit_isotonic(yt, yp_tr)
        results["isotonic"].append(_brier(yt_val, apply_isotonic(ir, yp_val)))

        # Platt
        try:
            pl = fit_platt(yt, yp_tr)
            results["platt"].append(_brier(yt_val, apply_platt(pl, yp_val)))
        except Exception:
            results["platt"].append(1.0)

        # Temperature
        try:
            T = fit_temperature(yt, yp_tr)
            results["temperature"].append(_brier(yt_val, apply_temperature(T, yp_val)))
        except Exception:
            results["temperature"].append(1.0)

    # Pick best
    mean_brier = {k: np.mean(v) for k, v in results.items()}
    best = min(mean_brier, key=mean_brier.get)
    print(f"  Calibration CV Brier: {mean_brier}")
    print(f"  Selected: {best}")

    # Refit on all data
    if best == "none":
        calibrator = None
    elif best == "isotonic":
        calibrator = fit_isotonic(y_true, y_pred)
    elif best == "platt":
        calibrator = fit_platt(y_true, y_pred)
    elif best == "temperature":
        calibrator = fit_temperature(y_true, y_pred)

    return best, calibrator


def apply_calibration(method: str, calibrator, y_pred):
    """Apply the selected calibration method."""
    if method == "none" or calibrator is None:
        return y_pred
    elif method == "isotonic":
        return apply_isotonic(calibrator, y_pred)
    elif method == "platt":
        return apply_platt(calibrator, y_pred)
    elif method == "temperature":
        return apply_temperature(calibrator, y_pred)
    else:
        return y_pred
