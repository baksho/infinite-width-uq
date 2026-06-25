"""Datasets and split strategies for the infinite-width UQ benchmark.

Two datasets share one interface so the model/eval code is dataset-agnostic:

* ``load_diabetes_dataset``  -- scikit-learn's bundled diabetes regression set.
  No download, ~442 samples / 10 features. The native sweet spot for exact
  NNGP / RBF-GP inference, which is O(n^3) in the number of training points.

* ``load_qsar_dataset``      -- a molecular potency (pIC50) regression loader.
  Requires an EXTERNAL csv of (smiles, pIC50) plus RDKit for featurisation, so
  it is written but not exercised here. The downstream pipeline is identical:
  it returns the same (X, y, feature_names) triple every other dataset returns.

The point of the two split strategies is the headline experiment: compare model
behaviour on an i.i.d. random split versus a deliberate covariate-shift split
where the test set lies *outside* the training support along one feature.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Dataset:
    """A regression dataset with named features."""

    X: np.ndarray            # (n, d) float64
    y: np.ndarray            # (n,)   float64
    feature_names: list[str]
    name: str


@dataclass
class Split:
    """A single train/test partition (already index-disjoint)."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    kind: str                # "random" or "covariate-shift:<feature>"
    shift_feature: Optional[str] = None


# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #
def load_diabetes_dataset() -> Dataset:
    """Load scikit-learn's bundled diabetes regression dataset.

    Target is a quantitative measure of disease progression one year after
    baseline. Ten baseline features (age, sex, bmi, bp and six serum
    measurements s1..s6) come pre-standardised in scikit-learn.
    """
    from sklearn.datasets import load_diabetes

    raw = load_diabetes()
    return Dataset(
        X=np.asarray(raw.data, dtype=np.float64),
        y=np.asarray(raw.target, dtype=np.float64),
        feature_names=list(raw.feature_names),
        name="diabetes",
    )


def load_qsar_dataset(
    csv_path: str,
    smiles_col: str = "smiles",
    target_col: str = "pIC50",
    radius: int = 2,
    n_bits: int = 1024,
) -> Dataset:
    """Load a QSAR potency dataset from a CSV of SMILES + continuous target.

    This path is intentionally not run in this repository's published results
    because it needs (a) an external dataset you supply and (b) RDKit for
    featurisation. Everything downstream is identical to the diabetes path.

    Expected CSV schema (header row required)::

        smiles,pIC50
        Cc1ccccc1,6.42
        ...

    Suggested public sources (verify access/licensing yourself before use):
    a single-target activity export from ChEMBL, or a MoleculeNet regression
    task. Because exact GP / NNGP inference is O(n^3), subsample large sets to
    a few thousand rows before fitting.

    Featurisation: binary Morgan (ECFP-like) fingerprints of length ``n_bits``.
    """
    import pandas as pd

    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError as exc:  # pragma: no cover - external dependency
        raise ImportError(
            "load_qsar_dataset needs RDKit. Install with "
            "`pip install rdkit` (or `conda install -c conda-forge rdkit`)."
        ) from exc

    df = pd.read_csv(csv_path)
    for col in (smiles_col, target_col):
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in {csv_path}.")

    feats, targets = [], []
    for smiles, target in zip(df[smiles_col], df[target_col]):
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            continue  # skip unparseable SMILES
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        arr = np.zeros((n_bits,), dtype=np.float64)
        from rdkit.DataStructs import ConvertToNumpyArray

        ConvertToNumpyArray(fp, arr)
        feats.append(arr)
        targets.append(float(target))

    if not feats:
        raise ValueError("No valid molecules parsed from the CSV.")

    return Dataset(
        X=np.vstack(feats),
        y=np.asarray(targets, dtype=np.float64),
        feature_names=[f"ecfp_{i}" for i in range(n_bits)],
        name="qsar",
    )


# --------------------------------------------------------------------------- #
# Splits
# --------------------------------------------------------------------------- #
def random_split(ds: Dataset, test_frac: float = 0.2, seed: int = 0) -> Split:
    """An i.i.d. random train/test partition."""
    rng = np.random.default_rng(seed)
    n = len(ds.y)
    idx = rng.permutation(n)
    n_test = int(round(test_frac * n))
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    return Split(
        X_train=ds.X[train_idx], y_train=ds.y[train_idx],
        X_test=ds.X[test_idx], y_test=ds.y[test_idx],
        kind="random",
    )


def covariate_shift_split(
    ds: Dataset, feature: str, test_frac: float = 0.2
) -> Split:
    """Hold out the upper tail of one feature as an out-of-support test set.

    Training data are the lower ``1 - test_frac`` quantile of ``feature``; the
    test set is the upper ``test_frac`` tail. The test inputs therefore lie
    *beyond* the training support in that one direction -- a controlled
    extrapolation stress test rather than interpolation.
    """
    if feature not in ds.feature_names:
        raise KeyError(f"Feature '{feature}' not in {ds.feature_names}.")
    j = ds.feature_names.index(feature)
    order = np.argsort(ds.X[:, j])              # ascending feature value
    n = len(order)
    n_test = int(round(test_frac * n))
    train_idx, test_idx = order[:-n_test], order[-n_test:]
    return Split(
        X_train=ds.X[train_idx], y_train=ds.y[train_idx],
        X_test=ds.X[test_idx], y_test=ds.y[test_idx],
        kind=f"covariate-shift:{feature}",
        shift_feature=feature,
    )
