"""Targets, predictors and their alignment to forecast dates (PLAN.md sections 2, 6, 8).

Every table is indexed by forecast origin month m. Row m holds what is known at the
forecast date tau(m) and the targets for month m+1 (h = 1) or months m+1..m+3 (h = 3).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import data
from .config import load_config


# every (macro series, target) pair that enters a forecast; each needs a configured release lag
MACRO_USES = [("sp500_return", "sp500"), ("sp500_return", "ics"), ("ics", "ics"), ("ics", "sp500"),
              ("epu", "sp500"), ("epu", "ics")]
DAILY_PUBLISHED = ["sffed_nsi"]


class ReleaseLagMissing(KeyError):
    """A macro series is used for a target without a configured release lag."""


def check_release_lags(cfg: dict | None = None) -> None:
    """Fail unless every macro series in use has a release or publication lag."""
    cfg = load_config() if cfg is None else cfg
    for series, target in MACRO_USES:
        release_lag(series, target, cfg)
    for series in DAILY_PUBLISHED:
        if series not in (cfg.get("publication_lag_days") or {}):
            raise ReleaseLagMissing(f"no publication lag for daily series {series!r}")


def release_lag(series: str, target: str, cfg: dict | None = None) -> int:
    cfg = load_config() if cfg is None else cfg
    try:
        return int(cfg["release_lag_months"][series][target])
    except (KeyError, TypeError) as exc:
        raise ReleaseLagMissing(f"no release lag for series {series!r} used with target {target!r}") from exc


def lagged(series: pd.Series, name: str, target: str, origins: pd.PeriodIndex, cfg: dict | None = None) -> pd.Series:
    """Value of a monthly series usable at origin m for the given target: reference month m - lag."""
    lag = release_lag(name, target, cfg)
    return pd.Series(series.reindex(origins - lag).values, index=origins)


def targets(tau: pd.Series | None = None) -> pd.DataFrame:
    """Monthly targets by reference month: excess return R, direction U, raw and simple returns, ICS change C."""
    tau = data.forecast_dates() if tau is None else tau
    px = data.sp500_month_end(tau)["price"]
    tb = data.tbill_monthly()
    ics = data.ics_monthly()
    months = px.index
    logret = 100 * np.log(px / px.shift(1))
    rf = 100 * np.log1p(tb.reindex(months - 1).values / 1200.0)
    out = pd.DataFrame(index=months)
    out["price"] = px
    out["ret_raw"] = logret                                   # E10a
    out["ret_excess"] = logret - rf                           # T1a
    out["ret_simple_excess"] = 100 * (px / px.shift(1) - 1) - tb.reindex(months - 1).values / 12.0   # E10b
    out["up"] = np.where(px.shift(1).notna(), (px > px.shift(1)).astype(float), np.nan)              # T1b
    out["ics"] = ics.reindex(months)
    out["ics_change"] = out["ics"] - out["ics"].shift(1)      # T2
    return out


def sffed_monthly(tau: pd.Series, lag_days: int) -> pd.Series:
    """Mean daily San Francisco Fed index over [tau(m-1) - lag, tau(m) - lag), by origin month m."""
    daily = data.sffed_daily()
    edges = pd.DatetimeIndex(tau.values) - pd.Timedelta(days=lag_days)
    vals = {}
    for i in range(1, len(edges)):
        seg = daily[(daily.index >= edges[i - 1]) & (daily.index < edges[i])]
        if len(seg) >= 15:
            vals[tau.index[i]] = float(seg.mean())
    return pd.Series(vals, name="sffed_nsi")


def base_table(tau: pd.Series | None = None) -> pd.DataFrame:
    """Origin-indexed table of targets, own lags and published predictors, for all months with data."""
    cfg = load_config()
    check_release_lags(cfg)
    tau = data.forecast_dates() if tau is None else tau
    t = targets(tau)
    origins = t.index
    tab = pd.DataFrame(index=origins)
    tab["forecast_date"] = tau.reindex(origins).values
    # targets of origin m are the values of month m+1
    for col in ["ret_excess", "ret_raw", "ret_simple_excess", "up", "ics_change"]:
        tab[f"y_{col}"] = t[col].reindex(origins + 1).values
    # three-month targets: months m+1..m+3
    r = t["ret_excess"]
    tab["y3_ret_excess"] = (r.reindex(origins + 1).values + r.reindex(origins + 2).values
                            + r.reindex(origins + 3).values)
    tab["y3_ics_change"] = t["ics"].reindex(origins + 3).values - t["ics"].values
    # own lags, through the release-lag table
    tab["lag_ret_excess"] = lagged(t["ret_excess"], "sp500_return", "sp500", origins, cfg)
    tab["lag_ret_raw"] = lagged(t["ret_raw"], "sp500_return", "sp500", origins, cfg)
    tab["lag_ret_simple_excess"] = lagged(t["ret_simple_excess"], "sp500_return", "sp500", origins, cfg)
    tab["lag_ics_change"] = lagged(t["ics_change"], "ics", "ics", origins, cfg)
    tab["lag_ret_excess_for_ics"] = lagged(t["ret_excess"], "sp500_return", "ics", origins, cfg)
    # published predictors
    tab["ics_level_for_sp500"] = lagged(t["ics"], "ics", "sp500", origins, cfg)
    epu = data.epu_monthly()
    tab["epu_for_sp500"] = lagged(epu, "epu", "sp500", origins, cfg)
    tab["epu_for_ics"] = lagged(epu, "epu", "ics", origins, cfg)
    tab["sffed_nsi"] = sffed_monthly(tau, int(cfg["publication_lag_days"]["sffed_nsi"])).reindex(origins).values
    return tab


def text_sample(tab: pd.DataFrame) -> pd.DataFrame:
    """Rows for origins first_index_month .. (last_target_month - 1)."""
    s = load_config()["sample"]
    first = pd.Period(s["first_index_month"], "M")
    last = pd.Period(s["last_target_month"], "M") - 1
    return tab[(tab.index >= first) & (tab.index <= last)]


def long_sample(tab: pd.DataFrame) -> pd.DataFrame:
    s = load_config()["sample"]
    first = pd.Period(s["long_first_target_month"], "M") - 1
    last = pd.Period(s["last_target_month"], "M") - 1
    return tab[(tab.index >= first) & (tab.index <= last)]
