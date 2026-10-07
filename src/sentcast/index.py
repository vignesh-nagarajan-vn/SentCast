"""From headline scores to daily and monthly indices."""
from __future__ import annotations

import numpy as np
import pandas as pd


class LeakageError(RuntimeError):
    """Raised when text dated on or after its forecast date would enter a forecast."""


def assign_windows(dates: pd.Series, tau: pd.Series) -> pd.Series:
    """Index month for each date: month m such that tau(m-1) <= date < tau(m).

    dates are naive UTC calendar dates. Dates outside every window get NaT.
    """
    months = tau.index
    edges = pd.DatetimeIndex(tau.values)
    # pos = number of forecast dates <= date; the date belongs to the window that ends at edges[pos]
    pos = np.searchsorted(edges.values, dates.values.astype("datetime64[ns]"), side="right")
    ok = (pos >= 1) & (pos < len(edges))
    out = pd.Series(pd.PeriodIndex(months[np.where(ok, pos, 0)]), index=dates.index)
    out[~ok] = pd.NaT
    return out


def check_no_leakage(ts: pd.Series, window: pd.Series, tau: pd.Series) -> None:
    """Every text timestamp must be strictly before the forecast date of its window."""
    known = window.notna()
    if not known.any():
        return
    cutoff = pd.DatetimeIndex(tau.reindex(window[known]).values)
    stamps = pd.DatetimeIndex(ts[known].dt.tz_localize(None) if ts.dt.tz is not None else ts[known])
    bad = stamps >= cutoff
    if bad.any():
        raise LeakageError(f"{int(bad.sum())} text rows are dated on or after their forecast date")


def daily_index(df: pd.DataFrame, cols: list[str], min_headlines: int) -> pd.DataFrame:
    """Mean score per UTC date; dates with too few headlines are dropped."""
    g = df.groupby("date")
    daily = g[cols].mean()
    daily["n"] = g.size()
    return daily[daily["n"] >= min_headlines]


def monthly_index(df: pd.DataFrame, cols: list[str], tau: pd.Series, *, min_headlines: int, min_days: int,
                  weighting: str = "day") -> pd.DataFrame:
    """Monthly index per window.

    weighting='day': mean of the daily index over valid days (the plan's definition).
    weighting='headline': mean over all headlines in the window.
    Adds n_headlines and n_days. Months that fail the coverage rule are dropped.
    """
    work = df[["ts", "date"] + cols].copy()
    work["window"] = assign_windows(work["date"], tau)
    check_no_leakage(work["ts"], work["window"], tau)
    work = work[work["window"].notna()]
    per_day = work.groupby(["window", "date"])[cols].mean()
    per_day["n"] = work.groupby(["window", "date"]).size()
    valid = per_day[per_day["n"] >= min_headlines]
    days = valid.groupby(level="window").size().rename("n_days")
    heads = valid.groupby(level="window")["n"].sum().rename("n_headlines")
    if weighting == "day":
        out = valid.groupby(level="window")[cols].mean()
    elif weighting == "headline":
        keep = work.set_index(["window", "date"]).index.isin(valid.index)
        out = work[keep].groupby("window")[cols].mean()
    else:
        raise ValueError(weighting)
    out = out.join(days).join(heads)
    out = out[out["n_days"] >= min_days]
    out.index = pd.PeriodIndex(out.index, freq="M", name="month")
    return out.sort_index()


def composite(monthly: pd.DataFrame, members: list[str], fit_months: pd.PeriodIndex) -> pd.Series:
    """Equal-weight mean of member indices, each standardised on fit_months only."""
    z = []
    for k in members:
        base = monthly.loc[monthly.index.isin(fit_months), k]
        z.append((monthly[k] - base.mean()) / base.std(ddof=0))
    return pd.concat(z, axis=1).mean(axis=1).rename("composite")
