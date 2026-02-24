"""
Meta-ensemble: stacking + weighted averaging of model predictions.
Optimised for Brier score.
"""

import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import Ridge
from typing import Dict, List, Tuple


def _brier(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


# ═══════════════════════════════════════════════════════════════════════════════
#  Weighted average (grid search)
# ═══════════════════════════════════════════════════════════════════════════════

def optimize_weights_grid(y_true: np.ndarray,
                           model_preds: Dict[str, np.ndarray],
                           step: float = 0.05) -> Tuple[Dict[str, float], float]:
    """
    Grid-search blend weights in [0,1] with given step.
    For 2 models this is fast; for >2 we use scipy optimise instead.
    """
    names = list(model_preds.keys())
    preds = [model_preds[n] for n in names]
    n_models = len(names)

    if n_models == 2:
        best_w, best_brier = None, 1.0
        for w in np.arange(0, 1 + step, step):
            blend = w * preds[0] + (1 - w) * preds[1]
            b = _brier(y_true, blend)
            if b < best_brier:
                best_brier = b
                best_w = {names[0]: w, names[1]: 1 - w}
        return best_w, best_brier
    else:
        # Use scipy constrained optimization
        return optimize_weights_scipy(y_true, model_preds)


def optimize_weights_scipy(y_true: np.ndarray,
                            model_preds: Dict[str, np.ndarray]
                            ) -> Tuple[Dict[str, float], float]:
    """
    Optimize blend weights via scipy with constraints:
    - weights >= 0
    - weights sum to 1
    """
    names = list(model_preds.keys())
    preds = np.column_stack([model_preds[n] for n in names])
    n = len(names)

    def objective(w):
        blend = preds @ w
        return _brier(y_true, blend)

    # Constraints
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    bounds = [(0, 1)] * n
    x0 = np.ones(n) / n

    result = minimize(objective, x0, method="SLSQP",
                      bounds=bounds, constraints=constraints)

    weights = {names[i]: result.x[i] for i in range(n)}
    return weights, result.fun


# ═══════════════════════════════════════════════════════════════════════════════
#  Stacking meta-learner
# ═══════════════════════════════════════════════════════════════════════════════

def train_stacker(y_true: np.ndarray,
                   model_preds: Dict[str, np.ndarray],
                   alpha: float = 1.0) -> Tuple[object, float]:
    """
    Train a Ridge regression meta-learner on OOF predictions.
    Returns (fitted_model, training_brier).
    """
    names = list(model_preds.keys())
    X = np.column_stack([model_preds[n] for n in names])

    stacker = Ridge(alpha=alpha, fit_intercept=True)
    stacker.fit(X, y_true)

    blend = np.clip(stacker.predict(X), 0.001, 0.999)
    brier = _brier(y_true, blend)

    print(f"  Stacker coefficients: {dict(zip(names, stacker.coef_))}")
    print(f"  Stacker intercept: {stacker.intercept_:.4f}")
    print(f"  Stacker train Brier: {brier:.6f}")

    return stacker, brier


def predict_stacker(stacker, model_preds: Dict[str, np.ndarray],
                     names: List[str]) -> np.ndarray:
    """Apply stacker to new predictions."""
    X = np.column_stack([model_preds[n] for n in names])
    return np.clip(stacker.predict(X), 0.001, 0.999)


# ═══════════════════════════════════════════════════════════════════════════════
#  Select best ensemble
# ═══════════════════════════════════════════════════════════════════════════════

def select_best_ensemble(y_true: np.ndarray,
                          model_preds: Dict[str, np.ndarray],
                          fold_indices: List[np.ndarray] = None
                          ) -> Tuple[str, object, Dict]:
    """
    Compare weighted average vs stacking, select best by Brier + variance.
    Returns (method, model_or_weights, metrics).
    """
    # Weighted average
    wa_weights, wa_brier = optimize_weights_grid(y_true, model_preds)
    names = list(model_preds.keys())
    wa_preds = sum(wa_weights[n] * model_preds[n] for n in names)

    # Stacking
    stacker, stack_brier = train_stacker(y_true, model_preds)
    stack_preds = np.clip(stacker.predict(
        np.column_stack([model_preds[n] for n in names])
    ), 0.001, 0.999)

    # Compute fold-level variance if indices available
    wa_fvar, st_fvar = 0.0, 0.0
    if fold_indices is not None and len(fold_indices) > 0:
        wa_fold_briers = [_brier(y_true[idx], wa_preds[idx]) for idx in fold_indices]
        st_fold_briers = [_brier(y_true[idx], stack_preds[idx]) for idx in fold_indices]
        wa_fvar = np.std(wa_fold_briers)
        st_fvar = np.std(st_fold_briers)

    metrics = {
        "weighted_avg": {"brier": wa_brier, "fold_std": wa_fvar, "weights": wa_weights},
        "stacking": {"brier": stack_brier, "fold_std": st_fvar},
    }

    # Prefer method with lower Brier; break ties by lower variance
    if wa_brier <= stack_brier:
        print(f"  Selected: weighted_avg (Brier={wa_brier:.6f})")
        return "weighted_avg", wa_weights, metrics
    else:
        print(f"  Selected: stacking (Brier={stack_brier:.6f})")
        return "stacking", stacker, metrics
