"""Power of the main test, by simulation. Not pre-specified.

Question: how large would the predictive content of the composite have to be for the
test of E03 and E05 to find it on the test months of the text sample?

Design. The composite x is held at its observed values. For a given true R-squared r the
target is simulated as

    y*[t+1] = mean(y) + sd(y) * ( sqrt(r) * z(x[t]) + sqrt(1 - r) * e )

where z() standardises over the sample and e is drawn with replacement from the
standardised observed target, which keeps its tails. Each simulated sample goes through
the forecasts of M0 and M2 (expanding window, same first window) and the Clark-West
test. The decision rule asks, for this comparison, for R2_OS above zero and a
p-value below alpha; the share of samples that meet both is the power.

Inputs are files in results/ only. Run: python -m sentcast.power
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from .config import RESULTS, load_config, read_result, write_result
from .evaluate import _rng

TARGETS = {"sp500_return": "E03", "ics_change": "E05"}
GRID_STEP, GRID_MAX, REPS = 0.005, 0.30, 10000
LEVELS = {"power_50": 0.5, "power_80": 0.8}
SELECTED = {"r2_1pct": 0.01, "r2_2pct": 0.02, "r2_5pct": 0.05, "r2_10pct": 0.10}


def forecast_operators(x: np.ndarray, n0: int) -> tuple[np.ndarray, np.ndarray]:
    """Matrices F0, F2 with F @ y the expanding-window forecasts of M0 and M2 at the test origins.

    With x fixed, an OLS forecast is linear in y, so one matrix serves every simulated sample.
    """
    n = x.size
    f0, f2 = np.zeros((n - n0, n)), np.zeros((n - n0, n))
    for row, i in enumerate(range(n0, n)):
        f0[row, :i] = 1.0 / i
        design = np.column_stack([np.ones(i), x[:i]])
        f2[row, :i] = np.array([1.0, x[i]]) @ np.linalg.pinv(design)
    return f0, f2


def evaluate_samples(ys: np.ndarray, f0: np.ndarray, f2: np.ndarray, n0: int) -> tuple[np.ndarray, np.ndarray]:
    """(R2_OS, one-sided Clark-West p) of M2 against M0 for each row of ys."""
    y = ys[:, n0:]
    fb, fm = ys @ f0.T, ys @ f2.T
    sb, sm = np.square(y - fb), np.square(y - fm)
    f = sb - (sm - np.square(fb - fm))
    d = f - f.mean(axis=1, keepdims=True)
    stat = f.mean(axis=1) / np.sqrt(np.square(d).mean(axis=1) / f.shape[1])
    return 1.0 - sm.sum(axis=1) / sb.sum(axis=1), 1.0 - norm.cdf(stat)


def power_curve(y: np.ndarray, x: np.ndarray, *, n0: int, alpha: float, reps: int, grid: np.ndarray,
                rng: np.random.Generator) -> list[dict]:
    f0, f2 = forecast_operators(x, n0)
    zy, zx = (y - y.mean()) / y.std(), (x - x.mean()) / x.std()
    noise = rng.choice(zy, size=(reps, y.size), replace=True)          # the same draws for every r
    out = []
    for r in grid:
        ys = y.mean() + y.std() * (np.sqrt(r) * zx + np.sqrt(1.0 - r) * noise)
        r2, p = evaluate_samples(ys, f0, f2, n0)
        out.append({"true_r2": float(r), "reject_rate": float(np.mean(p < alpha)),
                    "r2_os_positive_rate": float(np.mean(r2 > 0)),
                    "power": float(np.mean((p < alpha) & (r2 > 0))), "median_r2_os": float(np.median(r2))})
    return out


def _inputs(exp: str) -> tuple[np.ndarray, np.ndarray]:
    """Target and composite by origin, as the experiment used them."""
    name = next(p.stem for p in RESULTS.glob(f"{exp}_forecasts.csv"))
    f = pd.read_csv(RESULTS / f"{name}.csv")
    x = pd.read_csv(RESULTS / "E02_indices_monthly.csv").set_index("index_month")["composite"]
    y, x = f["y"].to_numpy(float), x.reindex(f["origin_month"]).to_numpy(float)
    if not (np.isfinite(y).all() and np.isfinite(x).all()):
        raise ValueError(f"{exp}: target or composite has missing months")
    return y, x


def main() -> None:
    cfg = load_config()
    alpha, seed = cfg["evaluation"]["alpha"], cfg["seed"]
    grid = np.round(np.arange(0.0, GRID_MAX + GRID_STEP / 2, GRID_STEP), 6)
    results = {}
    for target, exp in TARGETS.items():
        source = next(p.stem for p in RESULTS.glob(f"{exp}_*.json"))
        n0 = read_result(source)["sample"]["initial_window"]
        y, x = _inputs(exp)
        curve = power_curve(y, x, n0=n0, alpha=alpha, reps=REPS, grid=grid, rng=_rng(seed, f"power:{target}"))
        by_r2 = {c["true_r2"]: c for c in curve}
        results[target] = {
            "source": f"results/{source}.json", "n_origins": int(y.size), "initial_window": int(n0),
            "n_oos": int(y.size - n0), "at_zero": by_r2[0.0],
            "selected": {k: by_r2[v] for k, v in SELECTED.items()},
            # smallest true R-squared on the grid at which the power reaches the level; None if it never does
            "smallest_r2": {k: next((c["true_r2"] for c in curve if c["power"] >= lv), None) for k, lv in LEVELS.items()},
            "grid": curve,
        }
    write_result("power", {"name": "power", "pre_specified": False, "results": results},
                 {"alpha": alpha, "seed": seed, "reps": REPS, "grid_step": GRID_STEP, "grid_max": GRID_MAX,
                  "levels": LEVELS, "comparison": "M2_vs_M0", "predictor": "composite", "horizon": 1})
    for target, r in results.items():
        print(target, "power at zero", r["at_zero"]["power"], "smallest R2 for 50% / 80% power", r["smallest_r2"])


if __name__ == "__main__":
    main()
