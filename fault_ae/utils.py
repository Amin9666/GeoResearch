"""
Shared feature engineering and utilities for the Fault AE prediction pipeline.

Feature engineering strategy:
  - Log1p-transform highly right-skewed AE features so that outliers do not
    dominate and relationships become more linear.
  - Add physically motivated derived features (power, energy-per-count, etc.).
  - Rename columns that contain hyphens so that PySR (Julia) can use them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ── Column name mapping (hyphens → underscores for PySR/Julia) ──────────
# Epsilon added to denominators in derived features to avoid division by zero.
EPSILON = 1.0

COL_RENAME = {
    "A-FRQ":      "A_FRQ",
    "R-FRQ":      "R_FRQ",
    "I-FRQ":      "I_FRQ",
    "ABS-ENERGY": "ABS_ENERGY",
    "FRQ-C":      "FRQ_C",
    "P-FRQ":      "P_FRQ",
}

TARGET_COL = "Load"

# Raw feature columns (after rename)
RAW_FEATURE_COLS = [
    "RISE", "COUN", "ENER", "DURATION", "AMP", "A_FRQ",
    "RMS", "ASL", "PCNTS", "R_FRQ", "I_FRQ", "ABS_ENERGY", "FRQ_C", "P_FRQ",
]

# Highly skewed features that benefit from log1p transform
_LOG_COLS = ["RISE", "COUN", "ENER", "DURATION", "ABS_ENERGY", "PCNTS", "AMP", "ASL", "RMS"]


def load_data(path: str | "Path") -> tuple[pd.DataFrame, pd.Series]:
    """Load the Excel file, rename columns, return (X_engineered, y)."""
    df = pd.read_excel(path)
    df = df.rename(columns=COL_RENAME)
    y = df[TARGET_COL].copy()
    X = engineer_features(df[RAW_FEATURE_COLS].copy())
    return X, y


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build an enriched feature matrix from raw AE columns.

    Returns a DataFrame with the original columns PLUS:
      - <col>_log  : log1p transform for skewed columns
      - Power_log  : log1p(ENER / (DURATION + 1))  — average power proxy
      - AbsE_cnt_log : log1p(ABS_ENERGY / (COUN + 1)) — energy per hit
      - AMP_RMS    : AMP * RMS product
      - ASL_AMP_diff : ASL – AMP  (both in dB units)
      - cnt_dur    : COUN / (DURATION + 1)  — count density
      - freq_ratio : A_FRQ / (FRQ_C + 1)  — spectral centroid ratio
    """
    out = df.copy()

    # Log1p transforms
    for c in _LOG_COLS:
        if c in out.columns:
            out[f"{c}_log"] = np.log1p(out[c].clip(lower=0))

    # Derived AE features
    out["Power_log"] = np.log1p((out["ENER"] / (out["DURATION"] + EPSILON)).clip(lower=0))
    out["AbsE_cnt_log"] = np.log1p((out["ABS_ENERGY"] / (out["COUN"] + EPSILON)).clip(lower=0))
    out["AMP_RMS"] = out["AMP"] * out["RMS"]
    out["ASL_AMP_diff"] = out["ASL"] - out["AMP"]
    out["cnt_dur"] = out["COUN"] / (out["DURATION"] + EPSILON)
    out["freq_ratio"] = out["A_FRQ"] / (out["FRQ_C"] + EPSILON)

    return out


def feature_names(df: pd.DataFrame) -> list[str]:
    """Return column names as a list."""
    return list(df.columns)
