"""Run the full four-way UQ comparison on the diabetes dataset.

Fits NNGP, DeepEnsemble, MCDropout and RBF-GP on (a) a random split and
(b) a covariate-shift split that holds out the upper BMI tail, then writes
metric tables and figures to ``results/``.

Usage::

    python experiments/run_diabetes.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from iwuq import data, metrics, plotting          # noqa: E402
from iwuq.models import MODEL_REGISTRY            # noqa: E402
from iwuq.selective import aurc                   # noqa: E402

HERE = os.path.dirname(__file__)
FIG_DIR = os.path.join(HERE, "..", "results", "figures")
TAB_DIR = os.path.join(HERE, "..", "results", "tables")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)

SHIFT_FEATURE = "bmi"
SEED = 0


def fit_all(split):
    """Fit every model on a split and return {name: (mean, std)}."""
    preds = {}
    for name, cls in MODEL_REGISTRY.items():
        model = cls(seed=SEED) if "seed" in cls.__init__.__code__.co_varnames else cls()
        model.fit(split.X_train, split.y_train)
        preds[name] = model.predict_dist(split.X_test)
    return preds


def metrics_table(preds, y):
    rows = {}
    for name, (mean, std) in preds.items():
        row = metrics.evaluate(y, mean, std)
        row["AURC"] = aurc(y, mean, std)
        rows[name] = row
    return pd.DataFrame(rows).T


def main():
    ds = data.load_diabetes_dataset()
    print(f"Loaded {ds.name}: X={ds.X.shape}, y={ds.y.shape}, "
          f"features={ds.feature_names}")

    splits = {
        "random": data.random_split(ds, test_frac=0.2, seed=SEED),
        "shift": data.covariate_shift_split(ds, SHIFT_FEATURE, test_frac=0.2),
    }

    preds_by_split, tables = {}, {}
    for key, split in splits.items():
        print(f"\n=== Fitting on {split.kind} split "
              f"(train={len(split.y_train)}, test={len(split.y_test)}) ===")
        preds = fit_all(split)
        preds_by_split[key] = (preds, split.y_test)
        tbl = metrics_table(preds, split.y_test).round(3)
        tables[key] = tbl
        print(tbl.to_string())
        tbl.to_csv(os.path.join(TAB_DIR, f"metrics_{key}.csv"))
        with open(os.path.join(TAB_DIR, f"metrics_{key}.md"), "w") as fh:
            fh.write(tbl.to_markdown())

    # Figures.
    p_rand, y_rand = preds_by_split["random"]
    p_shift, y_shift = preds_by_split["shift"]

    plotting.plot_calibration(
        p_rand, y_rand, title="Calibration -- random split",
        path=os.path.join(FIG_DIR, "calibration_random.png"))
    plotting.plot_calibration(
        p_shift, y_shift, title="Calibration -- covariate-shift split",
        path=os.path.join(FIG_DIR, "calibration_shift.png"))
    plotting.plot_risk_coverage(
        p_shift, y_shift, title="Risk-coverage -- covariate-shift split",
        path=os.path.join(FIG_DIR, "risk_coverage_shift.png"))
    plotting.plot_uncertainty_under_shift(
        p_rand, y_rand, p_shift, y_shift,
        path=os.path.join(FIG_DIR, "uncertainty_under_shift.png"))
    plotting.plot_predictions(
        p_shift, y_shift, title="Predicted vs true (95% interval) -- shift split",
        path=os.path.join(FIG_DIR, "predictions_shift.png"))

    print(f"\nWrote tables to {TAB_DIR} and figures to {FIG_DIR}.")
    return tables


if __name__ == "__main__":
    main()
