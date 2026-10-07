import numpy as np
import pytest

from sentcast import evaluate as ev
from sentcast.forecast import logit_forecasts, ols_forecasts


def test_r2_os():
    e_b = np.array([1.0, -1.0, 2.0])
    assert ev.r2_os(e_b, e_b) == pytest.approx(0.0)
    assert ev.r2_os(e_b / 2, e_b) == pytest.approx(0.75)
    assert ev.r2_os(e_b * 2, e_b) == pytest.approx(-3.0)


def test_newey_west_at_lag_zero_is_the_plain_variance_of_the_mean():
    x = np.random.default_rng(0).normal(size=200)
    assert ev.newey_west_var_of_mean(x, 0) == pytest.approx(x.var() / x.size)
    assert ev.newey_west_var_of_mean(x, 2) > 0


def test_clark_west_detects_a_real_predictor_and_not_noise():
    rng = np.random.default_rng(3)
    n, n0 = 400, 100
    x = rng.normal(size=n)
    y = 0.4 * x + rng.normal(size=n)
    f0, f2 = ols_forecasts(y, None, n0=n0), ols_forecasts(y, x, n0=n0)
    m = np.isfinite(f0)
    stat, p = ev.clark_west(y[m], f0[m], f2[m])
    assert stat > 3 and p < 0.001
    noise = rng.normal(size=n)
    f2n = ols_forecasts(y, noise, n0=n0)
    _, p_noise = ev.clark_west(y[m], f0[m], f2n[m])
    assert p_noise > 0.05


def test_clark_west_is_undefined_for_identical_forecasts():
    y = np.arange(10.0)
    stat, p = ev.clark_west(y, y * 0, y * 0)
    assert np.isnan(stat) and np.isnan(p)


def test_diebold_mariano_sign_and_symmetry():
    rng = np.random.default_rng(4)
    a = rng.normal(size=300) ** 2
    b = a + 0.5 + rng.normal(scale=0.1, size=300)
    stat, p = ev.diebold_mariano(a, b)
    assert stat < 0 and p < 0.001
    stat2, p2 = ev.diebold_mariano(b, a)
    assert stat2 == pytest.approx(-stat) and p2 == pytest.approx(p)


def test_pesaran_timmermann():
    rng = np.random.default_rng(5)
    u = (rng.random(200) > 0.4).astype(float)
    stat, p = ev.pesaran_timmermann(u, u)
    assert stat > 5 and p < 1e-6
    stat, p = ev.pesaran_timmermann(u, np.ones_like(u))          # a constant forecast has no timing content
    assert np.isnan(stat) and np.isnan(p)
    _, p_rand = ev.pesaran_timmermann(u, (rng.random(200) > 0.4).astype(float))
    assert p_rand > 0.01


def test_stationary_bootstrap_indices_and_intervals():
    rng = np.random.default_rng(6)
    idx = ev.stationary_bootstrap_indices(90, 6, 500, rng)
    assert idx.shape == (500, 90) and idx.min() >= 0 and idx.max() <= 89
    steps = (np.diff(idx, axis=1) % 90 == 1).mean()
    assert 0.75 < steps < 0.90                                   # expected 1 - 1/6 continuation rate
    e_b = rng.normal(size=90)
    e_m = e_b * 0.9
    lo, hi = ev.bootstrap_ci_r2(e_m, e_b, mean_block=6, reps=2000, level=0.90, seed=1, key="t")
    assert lo <= ev.r2_os(e_m, e_b) <= hi
    again = ev.bootstrap_ci_r2(e_m, e_b, mean_block=6, reps=2000, level=0.90, seed=1, key="t")
    assert (lo, hi) == again                                     # seeded
    lo, hi = ev.bootstrap_ci_mean(np.r_[np.ones(60), np.zeros(30)], mean_block=6, reps=2000, level=0.90, seed=1, key="h")
    assert lo < 2 / 3 < hi


def test_ols_forecasts_recover_a_linear_relation_and_start_after_the_initial_window():
    x = np.linspace(-1, 1, 60)
    y = 2.0 + 3.0 * x
    y_target = y.copy()
    f = ols_forecasts(y_target, x, n0=20)
    assert np.isnan(f[:20]).all() and np.isfinite(f[20:]).all()
    assert np.allclose(f[20:], y_target[20:])
    f0 = ols_forecasts(y_target, None, n0=20)
    assert f0[20] == pytest.approx(y_target[:20].mean())
    assert f0[59] == pytest.approx(y_target[:59].mean())


def test_rolling_window_uses_the_most_recent_observations():
    y = np.arange(100.0)
    f = ols_forecasts(y, None, n0=10, scheme="rolling", window=10)
    assert f[50] == pytest.approx(y[40:50].mean())


def test_logit_forecasts_are_probabilities_and_the_base_rate_is_the_training_share():
    rng = np.random.default_rng(7)
    x = rng.normal(size=120)
    u = (x + 0.5 * rng.normal(size=120) > 0).astype(float)
    p = logit_forecasts(u, x, n0=40)
    assert np.isnan(p[:40]).all()
    assert ((p[40:] >= 0) & (p[40:] <= 1)).all()
    assert ((p[40:] >= 0.5) == (x[40:] > 0)).mean() > 0.9        # agrees with the true decision rule
    p0 = logit_forecasts(u, None, n0=40)
    assert p0[40] == pytest.approx(u[:40].mean())
