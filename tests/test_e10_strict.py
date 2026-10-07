"""The strict consumer-sentiment variant (E10c) must not use the latest survey value anywhere."""
import numpy as np

from sentcast.forecast import ols_forecasts


def test_with_information_lag_two_the_latest_realised_target_is_not_used():
    rng = np.random.default_rng(3)
    n, n0, i = 80, 30, 60
    x = rng.normal(size=n)
    y = 0.5 * x + rng.normal(size=n)
    base = ols_forecasts(y, x, n0=n0, h=2)[i]
    y2 = y.copy()
    y2[i - 1] = 1e6      # target of origin i-1 = change into month i, which needs the month-i survey value
    assert ols_forecasts(y2, x, n0=n0, h=2)[i] == base
    assert ols_forecasts(y2, x, n0=n0, h=1)[i] != base      # the one-month setting does use it
