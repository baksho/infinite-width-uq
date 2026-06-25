"""Selective prediction: does the predictive std actually rank errors?

If a model's uncertainty is useful, deferring the least-confident predictions
(highest std) should leave the most-confident subset with lower error. The
risk-coverage curve plots error against the retained fraction; the area under
it (AURC) summarises this -- lower is better.
"""

from __future__ import annotations

import numpy as np


def risk_coverage_curve(y, mean, std, coverages=None):
    """RMSE on the most-confident ``c`` fraction, for each coverage ``c``.

    Points are sorted by ascending predictive std and the lowest-std prefix is
    retained at each coverage level.
    """
    if coverages is None:
        coverages = np.linspace(0.1, 1.0, 19)
    order = np.argsort(std)             # most confident first
    y_s, m_s = y[order], mean[order]
    sq_err = (y_s - m_s) ** 2
    n = len(y)
    risks = []
    for c in coverages:
        k = max(1, int(round(c * n)))
        risks.append(float(np.sqrt(np.mean(sq_err[:k]))))
    return np.asarray(coverages), np.asarray(risks)


def aurc(y, mean, std, coverages=None):
    """Area under the risk-coverage curve (trapezoidal). Lower is better."""
    cov, risk = risk_coverage_curve(y, mean, std, coverages)
    return float(np.trapezoid(risk, cov) / (cov[-1] - cov[0]))
