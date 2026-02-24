"""
Model B: LightGBM with Bayesian hyperparameter optimization (Optuna).
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
from sklearn.model_selection import StratifiedKFold
from typing import Dict, Tuple, Optional

from src.config import (
    RANDOM_SEED, LGBM_SEARCH, LGBM_N_TRIALS, LGBM_EARLY_STOP,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _brier_score(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def _lgb_brier_eval(y_pred, dtrain):
    """Custom LightGBM evaluation metric: Brier score (lower = better)."""
    y_true = dtrain.get_label()
    brier = _brier_score(y_true, y_pred)
    return "brier", brier, False  # False = lower is better


def optimize_lgbm(X_train: pd.DataFrame, y_train: np.ndarray,
                   n_trials: int = LGBM_N_TRIALS,
                   n_folds: int = 5) -> Dict:
    """
    Run Optuna Bayesian HPO to find best LightGBM parameters.
    Returns best_params dict.
    """
    def objective(trial):
        params = {
            "objective": "binary",
            "metric": "custom",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "random_state": RANDOM_SEED,
            "num_leaves": trial.suggest_int("num_leaves", *LGBM_SEARCH["num_leaves"]),
            "max_depth": trial.suggest_int("max_depth", *LGBM_SEARCH["max_depth"]),
            "learning_rate": trial.suggest_float("learning_rate", *LGBM_SEARCH["learning_rate"], log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", *LGBM_SEARCH["min_child_samples"]),
            "reg_alpha": trial.suggest_float("reg_alpha", *LGBM_SEARCH["reg_alpha"]),
            "reg_lambda": trial.suggest_float("reg_lambda", *LGBM_SEARCH["reg_lambda"]),
            "subsample": trial.suggest_float("subsample", *LGBM_SEARCH["subsample"]),
            "colsample_bytree": trial.suggest_float("colsample_bytree", *LGBM_SEARCH["colsample_bytree"]),
        }

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
        brier_scores = []

        for train_idx, val_idx in skf.split(X_train, y_train):
            X_tr = X_train.iloc[train_idx] if hasattr(X_train, "iloc") else X_train[train_idx]
            y_tr = y_train[train_idx]
            X_val = X_train.iloc[val_idx] if hasattr(X_train, "iloc") else X_train[val_idx]
            y_val = y_train[val_idx]

            dtrain = lgb.Dataset(X_tr, label=y_tr)
            dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

            model = lgb.train(
                params,
                dtrain,
                num_boost_round=2000,
                valid_sets=[dval],
                feval=_lgb_brier_eval,
                callbacks=[
                    lgb.early_stopping(stopping_rounds=LGBM_EARLY_STOP, verbose=False),
                    lgb.log_evaluation(period=0),
                ],
            )

            preds = model.predict(X_val)
            brier_scores.append(_brier_score(y_val, preds))

        return np.mean(brier_scores)

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    best.update({
        "objective": "binary",
        "metric": "custom",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "random_state": RANDOM_SEED,
    })

    print(f"  LightGBM HPO: best Brier = {study.best_value:.6f}")
    return best


def train_lgbm(X_train, y_train, X_val=None, y_val=None,
               params: Optional[Dict] = None,
               num_boost_round: int = 2000) -> Tuple:
    """
    Train LightGBM with given params and optional early stopping.
    Returns (model, feature_importances_dict).
    """
    if params is None:
        params = {
            "objective": "binary",
            "metric": "custom",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "max_depth": 6,
            "learning_rate": 0.05,
            "min_child_samples": 30,
            "reg_alpha": 1.0,
            "reg_lambda": 1.0,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": RANDOM_SEED,
        }

    dtrain = lgb.Dataset(X_train, label=y_train)
    callbacks = [lgb.log_evaluation(period=0)]

    valid_sets = [dtrain]
    if X_val is not None and y_val is not None:
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
        valid_sets.append(dval)
        callbacks.append(lgb.early_stopping(stopping_rounds=LGBM_EARLY_STOP, verbose=False))

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        valid_sets=valid_sets,
        feval=_lgb_brier_eval,
        callbacks=callbacks,
    )

    # Feature importances
    feature_names = (X_train.columns.tolist()
                     if hasattr(X_train, "columns")
                     else [f"f{i}" for i in range(X_train.shape[1])])
    importances = dict(zip(feature_names, model.feature_importance(importance_type="gain")))

    return model, importances


def predict_lgbm(model, X):
    """Return probabilities."""
    return model.predict(X)
