"""
Model A: Regularized Logistic Regression with elastic net path search.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

from src.config import RANDOM_SEED


def build_logistic_model(l1_ratios=None):
    """
    Build StandardScaler + ElasticNet LogisticRegression pipeline.
    """
    if l1_ratios is None:
        l1_ratios = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegressionCV(
            penalty="elasticnet",
            solver="saga",
            l1_ratios=l1_ratios,
            Cs=20,
            cv=5,
            scoring="neg_brier_score",
            max_iter=5000,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )),
    ])
    return pipe


def train_logistic(X_train, y_train, l1_ratios=None):
    """Train and return fitted pipeline + metadata."""
    model = build_logistic_model(l1_ratios)
    model.fit(X_train, y_train)

    lr = model.named_steps["lr"]
    info = {
        "C": lr.C_[0],
        "l1_ratio": lr.l1_ratio_[0] if hasattr(lr, "l1_ratio_") else None,
        "coefs": dict(zip(
            X_train.columns if hasattr(X_train, "columns") else range(X_train.shape[1]),
            lr.coef_[0]
        )),
        "intercept": lr.intercept_[0],
    }
    return model, info


def predict_logistic(model, X):
    """Return calibrated probabilities."""
    return model.predict_proba(X)[:, 1]
