"""Figure helpers. Each returns a matplotlib Figure and is colour-blind safe."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import metrics, selective

# Okabe-Ito palette (colour-blind friendly).
PALETTE = {
    "NNGP": "#0072B2",
    "DeepEnsemble": "#D55E00",
    "MCDropout": "#009E73",
    "RBF-GP": "#CC79A7",
}


def plot_calibration(preds, y, level_grid=None, title="", path=None):
    """Reliability diagram: nominal vs empirical central-interval coverage."""
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="ideal")
    for name, (mean, std) in preds.items():
        levels, emp = metrics.calibration_curve(y, mean, std, level_grid)
        ax.plot(levels, emp, marker="o", ms=3, color=PALETTE.get(name),
                label=name)
    ax.set_xlabel("nominal coverage")
    ax.set_ylabel("empirical coverage")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)
    ax.set_aspect("equal")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig


def plot_risk_coverage(preds, y, title="", path=None):
    """Risk-coverage curves (RMSE on most-confident fraction)."""
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for name, (mean, std) in preds.items():
        cov, risk = selective.risk_coverage_curve(y, mean, std)
        ax.plot(cov, risk, marker="o", ms=3, color=PALETTE.get(name), label=name)
    ax.set_xlabel("coverage (fraction retained, most confident first)")
    ax.set_ylabel("RMSE on retained set")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig


def plot_uncertainty_under_shift(preds_random, y_random,
                                 preds_shift, y_shift, path=None):
    """Mean predictive std on the random vs the shifted test set, per model.

    The headline visual: which models inflate their uncertainty when the test
    inputs move outside the training support?
    """
    names = list(preds_random.keys())
    rnd = [np.mean(preds_random[n][1]) for n in names]
    shf = [np.mean(preds_shift[n][1]) for n in names]
    x = np.arange(len(names))
    w = 0.38
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.bar(x - w / 2, rnd, w, label="random split", color="#999999")
    ax.bar(x + w / 2, shf, w, label="covariate-shift split", color="#0072B2")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15)
    ax.set_ylabel("mean predictive std (target units)")
    ax.set_title("Predictive uncertainty: in-support vs out-of-support")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig


def plot_predictions(preds, y, title="", path=None):
    """Predicted vs true with 95% error bars, one panel per model."""
    from scipy.stats import norm
    z = norm.ppf(0.975)
    n = len(preds)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.4), sharex=True, sharey=True)
    if n == 1:
        axes = [axes]
    lo = min(y.min(), min(m.min() for m, _ in preds.values()))
    hi = max(y.max(), max(m.max() for m, _ in preds.values()))
    for ax, (name, (mean, std)) in zip(axes, preds.items()):
        ax.errorbar(y, mean, yerr=z * std, fmt="o", ms=3, alpha=0.5,
                    ecolor="#cccccc", color=PALETTE.get(name))
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("true")
    axes[0].set_ylabel("predicted")
    fig.suptitle(title)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig
