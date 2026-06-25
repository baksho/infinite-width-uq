"""infinite-width-uq: calibrated regression with Neural Network Gaussian Processes.

A four-way comparison of predictive-uncertainty quality between the
infinite-width (NNGP) limit of a network and three finite baselines, evaluated
on both an i.i.d. random split and a covariate-shift extrapolation split.
"""

from . import data, models, metrics, selective, plotting

__all__ = ["data", "models", "metrics", "selective", "plotting"]
__version__ = "0.1.0"
