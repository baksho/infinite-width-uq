"""Run the four-way UQ comparison on a QSAR (pIC50) dataset.

This mirrors ``run_diabetes.py`` exactly; only the loader differs. It is NOT
run in the published results because it needs an external dataset and RDKit.

Usage::

    pip install rdkit
    python experiments/run_qsar.py --csv path/to/activities.csv \
        --smiles-col smiles --target-col pIC50 --subsample 1500
"""

from __future__ import annotations

import argparse
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
SEED = 0


def fit_all(split):
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--smiles-col", default="smiles")
    ap.add_argument("--target-col", default="pIC50")
    ap.add_argument("--shift-feature", default=None,
                    help="ECFP bit to shift on; defaults to most-variable bit.")
    ap.add_argument("--subsample", type=int, default=1500,
                    help="Cap training size; exact GP/NNGP inference is O(n^3).")
    args = ap.parse_args()

    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(TAB_DIR, exist_ok=True)

    ds = data.load_qsar_dataset(args.csv, args.smiles_col, args.target_col)
    if args.subsample and len(ds.y) > args.subsample:
        rng = np.random.default_rng(SEED)
        keep = rng.choice(len(ds.y), args.subsample, replace=False)
        ds = data.Dataset(ds.X[keep], ds.y[keep], ds.feature_names, ds.name)
    print(f"Loaded {ds.name}: X={ds.X.shape}")

    # Default shift feature: the most-variable fingerprint bit.
    shift_feat = args.shift_feature or ds.feature_names[int(np.argmax(ds.X.var(0)))]

    splits = {
        "random": data.random_split(ds, test_frac=0.2, seed=SEED),
        "shift": data.covariate_shift_split(ds, shift_feat, test_frac=0.2),
    }
    for key, split in splits.items():
        print(f"\n=== {split.kind} (train={len(split.y_train)}, "
              f"test={len(split.y_test)}) ===")
        preds = fit_all(split)
        tbl = metrics_table(preds, split.y_test).round(3)
        print(tbl.to_string())
        tbl.to_csv(os.path.join(TAB_DIR, f"qsar_metrics_{key}.csv"))


if __name__ == "__main__":
    main()
