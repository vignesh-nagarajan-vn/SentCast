"""Smoke tests of the experiment helpers on synthetic data. No real data is read."""
import numpy as np
import pandas as pd
import pytest

from sentcast import experiments as ex


@pytest.fixture()
def synthetic():
    rng = np.random.default_rng(11)
    n = 138
    origins = pd.period_range("2015-03", periods=n, freq="M")
    signal = rng.normal(size=n)
    noise = rng.normal(size=n)
    y = 0.8 * signal + rng.normal(size=n)                   # a target the "good" index predicts
    tab = pd.DataFrame({
        "forecast_date": origins.to_timestamp("M"),
        "y_ret_excess": y,
        "lag_ret_excess": np.r_[0.0, y[:-1]],
        "y_up": (y > 0).astype(float),
        "y_ics_change": rng.normal(size=n),                 # a target nothing predicts
        "lag_ics_change": rng.normal(size=n),
        "sffed_nsi": rng.normal(size=n),
    }, index=origins)
    idx = pd.DataFrame({k: noise + rng.normal(scale=0.1, size=n) for k in ["lm", "vader", "finbert", "clf", "gdelt_tone"]},
                       index=origins)
    idx["composite"] = signal
    return tab, idx


def test_a_true_signal_is_found_and_noise_is_not(synthetic):
    tab, idx = synthetic
    res, fc = ex.eval_continuous(tab, "y_ret_excess", "lag_ret_excess", idx, n0=48, tag="t", names=ex.PREDICTORS)
    good = res["composite"]["M2_vs_M0"]
    assert good["n_oos"] == 90
    assert good["r2_os"] > 0.2 and good["p"] < 0.001
    assert good["r2_ci_lo"] < good["r2_os"] < good["r2_ci_hi"]
    assert ex.decide_continuous(res)["outcome"] == "positive"
    assert res["finbert"]["M2_vs_M0"]["r2_os"] < 0.05
    assert res["finbert"]["M2_vs_M0"]["p_bonferroni"] >= res["finbert"]["M2_vs_M0"]["p"]
    assert "p_bonferroni" not in res["composite"]["M2_vs_M0"]
    assert {"y", "M0", "M1", "M2_composite", "M3_composite"} <= set(fc.columns)
    assert fc["M0"].isna().sum() == 48


def test_no_signal_gives_a_null(synthetic):
    tab, idx = synthetic
    res, _ = ex.eval_continuous(tab, "y_ics_change", "lag_ics_change", idx, n0=48, tag="t2", names=ex.PREDICTORS)
    assert ex.decide_continuous(res)["outcome"] == "null"


def test_direction_models(synthetic):
    tab, idx = synthetic
    res, fc = ex.eval_direction(tab, idx, n0=48, tag="d", names=ex.PREDICTORS)
    good = res["composite"]["L2"]
    assert good["n_oos"] == 90
    assert good["hit_rate"] > good["hit_rate_base"]
    assert good["pt_p"] < 0.01
    assert ex.decide_direction(res)["outcome"] == "positive"
    assert 0 <= res["baselines"]["L0"]["brier"] <= 1
    assert "L2_composite" in fc.columns


def test_subset_evaluation_uses_fewer_origins(synthetic):
    tab, idx = synthetic
    subset = pd.Series(tab.index >= tab.index[100], index=tab.index)
    res, _ = ex.eval_continuous(tab, "y_ret_excess", "lag_ret_excess", idx, n0=48, tag="s", names=["composite"],
                                subset=subset)
    assert res["composite"]["M2_vs_M0"]["n_oos"] == 38


def test_incremental_and_horizon_three(synthetic):
    tab, idx = synthetic
    r = ex._incremental(tab, "y_ret_excess", ["sffed_nsi"], idx["composite"], n0=48, key="i")
    assert r["r2_os"] > 0.2 and r["controls"] == ["sffed_nsi"]
    tab = tab.copy()
    y = tab["y_ret_excess"]
    tab["y3"] = y + y.shift(-1) + y.shift(-2)               # last two origins have no realised target
    res, fc = ex.eval_continuous(tab, "y3", "lag_ret_excess", idx, n0=48, h=3, tag="h3", names=["composite"])
    assert res["composite"]["M2_vs_M0"]["n_oos"] == 136 - 50
    assert ex._sample_info(tab[tab["y3"].notna()], 48, 3)["n_oos"] == 86
