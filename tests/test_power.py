"""Power simulation: its fast forecasts and test statistics match the code the experiments use."""
import numpy as np

from sentcast import evaluate, forecast, power


def _data(n=80, seed=3):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    return 0.4 + 0.3 * x + rng.normal(size=n), x


def test_operators_reproduce_the_expanding_window_forecasts():
    y, x = _data()
    f0, f2 = power.forecast_operators(x, 30)
    assert np.allclose(f0 @ y, forecast.ols_forecasts(y, None, n0=30)[30:])
    assert np.allclose(f2 @ y, forecast.ols_forecasts(y, x, n0=30)[30:])


def test_statistics_match_evaluate():
    y, x = _data()
    f0, f2 = power.forecast_operators(x, 30)
    r2, p = power.evaluate_samples(y[None, :], f0, f2, 30)
    fb, fm = forecast.ols_forecasts(y, None, n0=30)[30:], forecast.ols_forecasts(y, x, n0=30)[30:]
    assert np.isclose(r2[0], evaluate.r2_os(y[30:] - fm, y[30:] - fb))
    assert np.isclose(p[0], evaluate.clark_west(y[30:], fb, fm)[1])


def test_power_rises_with_the_true_r2():
    y, x = _data(n=138)
    curve = power.power_curve(y, x, n0=48, alpha=0.05, reps=2000, grid=np.array([0.0, 0.05, 0.25]),
                              rng=np.random.default_rng(0))
    powers = [c["power"] for c in curve]
    assert powers[0] < 0.12 and powers[0] < powers[1] < powers[2] and powers[2] > 0.9
