"""Leakage controls.

These tests fail if text dated on or after a forecast date can enter that forecast, if a
macro series is used without a release lag, or if a forecast can see data from its future.
"""
import copy

import numpy as np
import pandas as pd
import pytest

from sentcast import features
from sentcast.config import INTERIM, load_config
from sentcast.forecast import logit_forecasts, ols_forecasts, train_slice
from sentcast.index import LeakageError, assign_windows, check_no_leakage, composite, monthly_index


def _tau():
    months = pd.period_range("2020-01", "2020-06", freq="M")
    days = pd.to_datetime(["2020-01-31", "2020-02-28", "2020-03-31", "2020-04-30", "2020-05-29", "2020-06-30"])
    return pd.Series(days, index=months)


def _corpus(dates, score=0.0):
    d = pd.to_datetime(dates)
    return pd.DataFrame({"ts": d.tz_localize("UTC") + pd.Timedelta(hours=12), "date": d, "s": score})


# --- text timestamps --------------------------------------------------------------------

def test_every_text_date_is_before_the_forecast_date_of_its_window():
    tau = _tau()
    dates = pd.Series(pd.date_range("2020-01-15", "2020-07-15", freq="D"))
    win = assign_windows(dates, tau)
    known = win.notna()
    cutoff = pd.DatetimeIndex(tau.reindex(win[known]).values)
    assert (dates[known].values < cutoff.values).all()


def test_text_on_the_forecast_date_goes_to_the_next_window():
    tau = _tau()
    win = assign_windows(pd.Series(pd.to_datetime(["2020-02-27", "2020-02-28", "2020-02-29"])), tau)
    assert [str(w) for w in win] == ["2020-02", "2020-03", "2020-03"]


def test_text_outside_all_windows_is_unassigned():
    win = assign_windows(pd.Series(pd.to_datetime(["2020-01-30", "2020-06-30", "2021-01-01"])), _tau())
    assert win.isna().all()


def test_check_no_leakage_fails_on_text_dated_on_or_after_the_forecast_date():
    tau = _tau()
    df = _corpus(["2020-02-27", "2020-02-28"])
    forced = pd.Series(pd.PeriodIndex(["2020-02", "2020-02"], freq="M"))   # second row is on the forecast date
    with pytest.raises(LeakageError):
        check_no_leakage(df["ts"], forced, tau)
    check_no_leakage(df["ts"].iloc[:1], forced.iloc[:1], tau)              # the first row alone is fine


def test_monthly_index_ignores_text_on_or_after_the_forecast_date():
    tau = _tau()
    before = pd.date_range("2020-01-31", "2020-02-27", freq="D")
    after = pd.date_range("2020-02-28", "2020-03-05", freq="D")
    df = pd.concat([_corpus(before.repeat(5), 1.0), _corpus(after.repeat(5), 1000.0)], ignore_index=True)
    m = monthly_index(df, ["s"], tau, min_headlines=5, min_days=15)
    assert m.loc[pd.Period("2020-02", "M"), "s"] == pytest.approx(1.0)


@pytest.mark.skipif(not (INTERIM / "scores.parquet").exists(), reason="corpus not built")
def test_real_corpus_has_no_text_on_or_after_its_forecast_date():
    from sentcast import data, pipeline

    tau = data.forecast_dates()
    df = pipeline.scored_corpus()
    win = assign_windows(df["date"], tau)
    check_no_leakage(df["ts"], win, tau)
    known = win.notna()
    cutoff = pd.DatetimeIndex(tau.reindex(win[known]).values)
    assert (df.loc[known, "ts"].dt.tz_localize(None).values < cutoff.values).all()


# --- macro release lags -----------------------------------------------------------------

def test_missing_release_lag_is_an_error():
    cfg = copy.deepcopy(load_config())
    with pytest.raises(features.ReleaseLagMissing):
        features.release_lag("no_such_series", "sp500", cfg)
    del cfg["release_lag_months"]["epu"]["ics"]
    with pytest.raises(features.ReleaseLagMissing):
        features.release_lag("epu", "ics", cfg)
    with pytest.raises(features.ReleaseLagMissing):
        features.check_release_lags(cfg)


def test_missing_publication_lag_is_an_error():
    cfg = copy.deepcopy(load_config())
    del cfg["publication_lag_days"]["sffed_nsi"]
    with pytest.raises(features.ReleaseLagMissing):
        features.check_release_lags(cfg)


def test_every_macro_series_in_use_has_a_release_lag():
    features.check_release_lags(load_config())
    cfg = load_config()
    for series, target in features.MACRO_USES:
        assert isinstance(cfg["release_lag_months"][series][target], int)
        assert cfg["release_lag_months"][series][target] >= 0


def test_lagged_uses_the_reference_month_allowed_by_the_lag():
    months = pd.period_range("2020-01", "2020-12", freq="M")
    series = pd.Series(np.arange(12.0), index=months)
    cfg = {"release_lag_months": {"x": {"t": 1}, "y": {"t": 0}}}
    origins = months[3:6]
    assert list(features.lagged(series, "x", "t", origins, cfg)) == [2.0, 3.0, 4.0]
    assert list(features.lagged(series, "y", "t", origins, cfg)) == [3.0, 4.0, 5.0]


def test_ics_is_lagged_when_it_predicts_the_stock_market():
    cfg = load_config()
    assert cfg["release_lag_months"]["ics"]["sp500"] >= 1
    assert cfg["release_lag_months"]["epu"]["sp500"] >= 1
    assert cfg["publication_lag_days"]["sffed_nsi"] >= 7


# --- forecasts cannot see their future --------------------------------------------------

def test_training_window_ends_before_the_origin():
    assert train_slice(48, 1, 48, "expanding", None) == slice(0, 48)
    assert train_slice(47, 1, 48, "expanding", None) is None
    assert train_slice(50, 3, 48, "expanding", None) == slice(0, 48)     # targets of origins 48, 49 not yet realised
    assert train_slice(49, 3, 48, "expanding", None) is None
    assert train_slice(100, 1, 48, "rolling", 48) == slice(52, 100)


@pytest.mark.parametrize("h", [1, 3])
def test_forecast_does_not_change_when_unrealised_targets_change(h):
    rng = np.random.default_rng(0)
    n, n0, i = 80, 30, 60
    x = rng.normal(size=n)
    y = 0.5 * x + rng.normal(size=n)
    base = ols_forecasts(y, x, n0=n0, h=h)[i]
    y2 = y.copy()
    y2[i - h + 1:] = 1e6            # targets not realised at the forecast date of origin i
    x2 = x.copy()
    x2[i + 1:] = -1e6               # predictors of later origins
    assert ols_forecasts(y2, x2, n0=n0, h=h)[i] == pytest.approx(base)
    assert ols_forecasts(y2, None, n0=n0, h=h)[i] == pytest.approx(ols_forecasts(y, None, n0=n0, h=h)[i])


def test_direction_forecast_does_not_change_when_future_changes():
    rng = np.random.default_rng(1)
    n, n0, i = 80, 30, 60
    x = rng.normal(size=n)
    u = (x + rng.normal(size=n) > 0).astype(float)
    base = logit_forecasts(u, x, n0=n0)[i]
    u2 = u.copy()
    u2[i:] = 1 - u2[i:]
    x2 = x.copy()
    x2[i + 1:] = 99.0
    assert logit_forecasts(u2, x2, n0=n0)[i] == pytest.approx(base)


def test_composite_is_standardised_on_the_initial_window_only():
    months = pd.period_range("2015-03", periods=60, freq="M")
    rng = np.random.default_rng(2)
    m = pd.DataFrame(rng.normal(size=(60, 2)), index=months, columns=["a", "b"])
    fit = months[:48]
    c1 = composite(m, ["a", "b"], fit)
    m2 = m.copy()
    m2.iloc[48:] += 100.0           # later values must not move earlier composite values
    c2 = composite(m2, ["a", "b"], fit)
    assert np.allclose(c1.iloc[:48], c2.iloc[:48])
