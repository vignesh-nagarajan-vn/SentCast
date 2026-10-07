"""Forecast evaluation: error metrics, tests and bootstrap intervals."""
from __future__ import annotations

import math
import zlib

import numpy as np
from scipy.stats import norm


def rmse(e: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(e))))


def mae(e: np.ndarray) -> float:
    return float(np.mean(np.abs(e)))


def r2_os(e_model: np.ndarray, e_bench: np.ndarray) -> float:
    """Out-of-sample R-squared: 1 - SSE(model) / SSE(benchmark)."""
    return float(1.0 - np.sum(np.square(e_model)) / np.sum(np.square(e_bench)))


def newey_west_var_of_mean(x: np.ndarray, lags: int) -> float:
    """Variance of the sample mean of x with a Bartlett kernel of the given lag length."""
    x = np.asarray(x, dtype=float)
    n = x.size
    d = x - x.mean()
    lrv = float(d @ d) / n
    for j in range(1, lags + 1):
        w = 1.0 - j / (lags + 1.0)
        lrv += 2.0 * w * float(d[j:] @ d[:-j]) / n
    return lrv / n


def clark_west(y: np.ndarray, yhat_bench: np.ndarray, yhat_model: np.ndarray, h: int = 1) -> tuple[float, float]:
    """Clark-West (2007) MSPE-adjusted statistic for a model that nests the benchmark.

    Returns (statistic, one-sided p-value). The alternative is that the larger model has
    lower mean squared prediction error.
    """
    y, fb, fm = (np.asarray(a, dtype=float) for a in (y, yhat_bench, yhat_model))
    f = np.square(y - fb) - (np.square(y - fm) - np.square(fb - fm))
    var = newey_west_var_of_mean(f, h - 1)
    if not var > 0:
        return float("nan"), float("nan")
    stat = float(f.mean() / math.sqrt(var))
    return stat, float(1.0 - norm.cdf(stat))


def diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray, h: int = 1) -> tuple[float, float]:
    """Diebold-Mariano (1995) test of equal expected loss. Returns (statistic, two-sided p).

    A negative statistic means forecast a has the lower loss.
    """
    d = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    var = newey_west_var_of_mean(d, h - 1)
    if not var > 0:
        return float("nan"), float("nan")
    stat = float(d.mean() / math.sqrt(var))
    return stat, float(2.0 * (1.0 - norm.cdf(abs(stat))))


def pesaran_timmermann(actual_up: np.ndarray, pred_up: np.ndarray) -> tuple[float, float]:
    """Pesaran-Timmermann (1992) test for two outcomes. Returns (statistic, one-sided p).

    Undefined (nan) when the forecasts or the outcomes never change.
    """
    a = np.asarray(actual_up, dtype=float)
    p = np.asarray(pred_up, dtype=float)
    n = a.size
    hit = float(np.mean(a == p))
    py, px = float(a.mean()), float(p.mean())
    p_star = py * px + (1 - py) * (1 - px)
    var_hit = p_star * (1 - p_star) / n
    var_star = (((2 * py - 1) ** 2) * px * (1 - px) + ((2 * px - 1) ** 2) * py * (1 - py)) / n \
        + 4 * py * px * (1 - py) * (1 - px) / n ** 2
    denom = var_hit - var_star
    if not denom > 1e-15:
        return float("nan"), float("nan")
    stat = (hit - p_star) / math.sqrt(denom)
    return float(stat), float(1.0 - norm.cdf(stat))


def _rng(seed: int, key: str) -> np.random.Generator:
    return np.random.default_rng([seed, zlib.crc32(key.encode("utf-8"))])


def stationary_bootstrap_indices(n: int, mean_block: float, reps: int, rng: np.random.Generator) -> np.ndarray:
    """Politis-Romano (1994) stationary bootstrap. Returns an integer array (reps, n)."""
    p = 1.0 / mean_block
    start = rng.integers(0, n, size=(reps, n))
    new_block = rng.random((reps, n)) < p
    new_block[:, 0] = True
    idx = np.empty((reps, n), dtype=np.int64)
    idx[:, 0] = start[:, 0]
    for t in range(1, n):
        idx[:, t] = np.where(new_block[:, t], start[:, t], (idx[:, t - 1] + 1) % n)
    return idx


def bootstrap_ci_r2(e_model: np.ndarray, e_bench: np.ndarray, *, mean_block: float, reps: int, level: float,
                    seed: int, key: str) -> tuple[float, float]:
    """Percentile interval for the out-of-sample R-squared."""
    sm, sb = np.square(e_model), np.square(e_bench)
    idx = stationary_bootstrap_indices(sm.size, mean_block, reps, _rng(seed, key))
    stat = 1.0 - sm[idx].sum(axis=1) / sb[idx].sum(axis=1)
    lo, hi = np.quantile(stat, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi)


def bootstrap_ci_mean(x: np.ndarray, *, mean_block: float, reps: int, level: float, seed: int,
                      key: str) -> tuple[float, float]:
    """Percentile interval for a mean, used for hit rates."""
    x = np.asarray(x, dtype=float)
    idx = stationary_bootstrap_indices(x.size, mean_block, reps, _rng(seed, key))
    stat = x[idx].mean(axis=1)
    lo, hi = np.quantile(stat, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi)
