"""Evaluation metrics for Gaussian predictive distributions.

All functions take ``y`` (true), ``mean``, ``std`` as 1-D arrays in the
original target units. Point-accuracy metrics ignore ``std``; the rest measure
the quality of the *uncertainty*.

CRPS uses the closed form for a Gaussian forecast (Gneiting & Raftery, 2007),
so no sampling is needed.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def rmse(y, mean, std=None):
    return float(np.sqrt(np.mean((y - mean) ** 2)))


def mae(y, mean, std=None):
    return float(np.mean(np.abs(y - mean)))


def gaussian_nll(y, mean, std):
    """Mean negative log-likelihood under N(mean, std^2)."""
    var = np.clip(std ** 2, 1e-12, None)
    return float(np.mean(0.5 * np.log(2 * np.pi * var) + (y - mean) ** 2 / (2 * var)))


def crps_gaussian(y, mean, std):
    """Closed-form CRPS for a Gaussian forecast, averaged over points.

    CRPS(N(mu, sigma^2), y) = sigma * [ z (2 Phi(z) - 1) + 2 phi(z) - 1/sqrt(pi) ]
    with z = (y - mu) / sigma.  Lower is better.
    """
    std = np.clip(std, 1e-12, None)
    z = (y - mean) / std
    crps = std * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1.0 / np.sqrt(np.pi))
    return float(np.mean(crps))


def interval_coverage(y, mean, std, level=0.95):
    """Empirical coverage of the central ``level`` predictive interval."""
    z = norm.ppf(0.5 + level / 2.0)
    lo, hi = mean - z * std, mean + z * std
    return float(np.mean((y >= lo) & (y <= hi)))


def mean_interval_width(mean, std, level=0.95):
    z = norm.ppf(0.5 + level / 2.0)
    return float(np.mean(2 * z * std))


def calibration_curve(y, mean, std, levels=None):
    """Nominal vs empirical coverage across a grid of central-interval levels."""
    if levels is None:
        levels = np.linspace(0.05, 0.95, 19)
    emp = np.array([interval_coverage(y, mean, std, lv) for lv in levels])
    return np.asarray(levels), emp


def miscalibration_area(y, mean, std, levels=None):
    """Area between the calibration curve and the diagonal (0 = perfect)."""
    levels, emp = calibration_curve(y, mean, std, levels)
    return float(np.trapezoid(np.abs(emp - levels), levels))


def evaluate(y, mean, std, level=0.95):
    """Bundle every metric into one dict for a single (model, split)."""
    return {
        "RMSE": rmse(y, mean),
        "MAE": mae(y, mean),
        "NLL": gaussian_nll(y, mean, std),
        "CRPS": crps_gaussian(y, mean, std),
        f"Coverage@{int(level * 100)}": interval_coverage(y, mean, std, level),
        f"Width@{int(level * 100)}": mean_interval_width(mean, std, level),
        "MiscalArea": miscalibration_area(y, mean, std),
    }
