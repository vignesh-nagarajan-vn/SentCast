"""Out-of-sample forecast engine.

Rows are forecast origins in time order. Row i holds the predictors known at the forecast
date of origin i and the target that origin i forecasts. With horizon h, the target of
origin s is fully realised at the forecast date of origin s + h, so a forecast made at
origin i may be estimated on origins 0 .. i - h only.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


def train_slice(i: int, h: int, n0: int, scheme: str, window: int | None) -> slice | None:
    """Training origins for a forecast at origin i, or None if fewer than n0 are available."""
    end = i - h + 1          # exclusive
    if scheme == "rolling":
        if window is None:
            raise ValueError("rolling scheme needs a window")
        start = max(0, end - window)
        need = min(n0, window)
    elif scheme == "expanding":
        start, need = 0, n0
    else:
        raise ValueError(f"unknown scheme {scheme!r}")
    if end - start < need:
        return None
    return slice(start, end)


def first_oos_origin(h: int, n0: int) -> int:
    return n0 + h - 1


def ols_forecasts(y: np.ndarray, X: np.ndarray | None, *, n0: int, h: int = 1, scheme: str = "expanding",
                  window: int | None = None) -> np.ndarray:
    """One forecast per origin; nan where no forecast is made. X=None gives the training mean."""
    y = np.asarray(y, dtype=float)
    n = y.size
    out = np.full(n, np.nan)
    if X is not None:
        X = np.asarray(X, dtype=float).reshape(n, -1)
    for i in range(n):
        sl = train_slice(i, h, n0, scheme, window)
        if sl is None:
            continue
        yt = y[sl]
        if X is None:
            out[i] = yt.mean()
            continue
        Xt = X[sl]
        ok = np.isfinite(yt) & np.isfinite(Xt).all(axis=1)
        if ok.sum() < Xt.shape[1] + 2 or not np.isfinite(X[i]).all():
            continue
        A = np.column_stack([np.ones(ok.sum()), Xt[ok]])
        beta, *_ = np.linalg.lstsq(A, yt[ok], rcond=None)
        out[i] = beta[0] + X[i] @ beta[1:]
    return out


def logit_forecasts(u: np.ndarray, X: np.ndarray | None, *, n0: int, h: int = 1, scheme: str = "expanding",
                    window: int | None = None) -> np.ndarray:
    """Probability that u = 1, one per origin. X=None gives the training share of ones."""
    u = np.asarray(u, dtype=float)
    n = u.size
    out = np.full(n, np.nan)
    if X is not None:
        X = np.asarray(X, dtype=float).reshape(n, -1)
    for i in range(n):
        sl = train_slice(i, h, n0, scheme, window)
        if sl is None:
            continue
        ut = u[sl]
        if X is None:
            out[i] = ut.mean()
            continue
        Xt = X[sl]
        ok = np.isfinite(ut) & np.isfinite(Xt).all(axis=1)
        if ok.sum() < Xt.shape[1] + 2 or not np.isfinite(X[i]).all():
            continue
        ut, Xt = ut[ok], Xt[ok]
        if ut.min() == ut.max():
            out[i] = ut[0]
            continue
        mu, sd = Xt.mean(axis=0), Xt.std(axis=0)
        sd = np.where(sd > 0, sd, 1.0)
        model = LogisticRegression(C=1e6, max_iter=1000)
        model.fit((Xt - mu) / sd, ut.astype(int))
        out[i] = model.predict_proba(((X[i] - mu) / sd).reshape(1, -1))[0, 1]
    return out
