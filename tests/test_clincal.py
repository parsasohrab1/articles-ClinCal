import numpy as np

from clincal.calib import ece, metrics, recalibrate_intercept, recalibrate_platt, slope_intercept


def test_ece_zero_for_perfect_and_large_for_biased():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.05, 0.95, 4000)
    y = (rng.uniform(size=4000) < p).astype(int)
    assert ece(y, p) < 0.04
    assert ece(y, np.clip(p + 0.3, 0, 1)) > 0.2


def test_slope_near_one_when_calibrated():
    rng = np.random.default_rng(1)
    p = rng.uniform(0.05, 0.95, 5000)
    y = (rng.uniform(size=5000) < p).astype(int)
    s, i = slope_intercept(y, p)
    assert abs(s - 1) < 0.15 and abs(i) < 0.15


def test_intercept_recalibration_matches_prevalence():
    rng = np.random.default_rng(2)
    z = rng.normal(0, 1, 3000)
    y = (rng.uniform(size=3000) < 1 / (1 + np.exp(-(z + 1.5)))).astype(int)
    f = recalibrate_intercept(z, y)
    assert abs(f(z).mean() - y.mean()) < 0.01


def test_platt_improves_brier_on_shifted_labels():
    rng = np.random.default_rng(3)
    z = rng.normal(0, 1, 3000)
    y = (rng.uniform(size=3000) < 1 / (1 + np.exp(-(0.5 * z + 1.0)))).astype(int)
    base = 1 / (1 + np.exp(-z))
    f = recalibrate_platt(z, y)
    assert metrics(y, f(z))["brier"] < metrics(y, base)["brier"]
