from __future__ import annotations

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def fit_model(X, y):
    m = make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=1000)
    )
    return m.fit(X, y)


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def ece(y, p, bins: int = 5) -> float:
    """Expected calibration error with equal-count bins."""
    order = np.argsort(p)
    chunks = np.array_split(order, bins)
    n = len(y)
    return float(sum(len(c) / n * abs(y[c].mean() - p[c].mean()) for c in chunks if len(c)))


def slope_intercept(y, p):
    """Calibration slope and intercept from a logistic fit of y on logit(p)."""
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(logit(p)[:, None], y)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def metrics(y, p) -> dict:
    s, i = slope_intercept(y, p)
    return {
        "auc": float(roc_auc_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "ece": ece(y, p),
        "slope": s,
        "intercept": i,
        "prevalence": float(y.mean()),
        "mean_pred": float(p.mean()),
    }


def recalibrate_intercept(z_cal, y_cal):
    """Recalibration-in-the-large: shift the logit by the intercept of an offset-only model."""
    lo, hi = -10.0, 10.0
    for _ in range(60):
        mid = (lo + hi) / 2
        pm = 1 / (1 + np.exp(-(z_cal + mid)))
        if pm.mean() > y_cal.mean():
            hi = mid
        else:
            lo = mid
    shift = (lo + hi) / 2
    return lambda z: 1 / (1 + np.exp(-(z + shift)))


def recalibrate_platt(z_cal, y_cal):
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(z_cal[:, None], y_cal)
    return lambda z: lr.predict_proba(z[:, None])[:, 1]
