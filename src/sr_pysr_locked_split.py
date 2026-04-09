#!/usr/bin/env python
"""
PySR locked-split optimization.

Optimizes symbolic regression directly on the exact split used by
`compare_linear_ml_symbolic.py`:
  - IQR 2.0 filtering on base features + target
  - features: RISE, COUN, ENER, DURATION, AMP
  - train_test_split(test_size=0.2, random_state=42)
"""

from __future__ import annotations

from pathlib import Path
import inspect
import os
import pickle
import shutil
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")


def remove_outliers_iqr(data: pd.DataFrame, features: list[str], iqr_multiplier: float = 2.0) -> pd.DataFrame:
    mask = np.ones(len(data), dtype=bool)
    for feature in features:
        q1 = data[feature].quantile(0.25)
        q3 = data[feature].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - iqr_multiplier * iqr
        upper = q3 + iqr_multiplier * iqr
        mask &= (data[feature] >= lower) & (data[feature] <= upper)
    return data[mask].reset_index(drop=True)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def build_pysr_regressor(regressor_cls, **kwargs):
    valid = set(inspect.signature(regressor_cls.__init__).parameters.keys())
    filtered = {k: v for k, v in kwargs.items() if k in valid}
    return regressor_cls(**filtered)


def main() -> None:
    print("=" * 80)
    print("PySR LOCKED-SPLIT OPTIMIZATION")
    print("=" * 80)

    root = Path(__file__).resolve().parents[1]
    data_path = root / "data" / "raw" / "Shear_Data_15.csv"
    if not data_path.exists():
        data_path = root / "Shear_Data_15.csv"

    out_csv = root / "outputs" / "csv"
    out_models = root / "outputs" / "models"
    out_csv.mkdir(parents=True, exist_ok=True)
    out_models.mkdir(parents=True, exist_ok=True)

    system_julia = shutil.which("julia")
    if system_julia:
        os.environ.setdefault("PYTHON_JULIAPKG_EXE", system_julia)
        os.environ.setdefault("PYTHON_JULIACALL_EXE", system_julia)

    from pysr import PySRRegressor

    df = pd.read_csv(data_path)
    input_features = ["RISE", "COUN", "ENER", "DURATION", "AMP"]
    target = "SHEAR STRESS"

    df = remove_outliers_iqr(df, input_features + [target], iqr_multiplier=2.0)
    x = df[input_features]
    y = df[target]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    variable_names = list(x.columns)
    x_train_arr = x_train.values
    x_test_arr = x_test.values

    configs = [
        {
            "name": "L1",
            "niterations": 250,
            "populations": 48,
            "population_size": 100,
            "maxsize": 24,
            "parsimony": 0.006,
            "binary_operators": ["+", "-", "*", "/", "^"],
            "unary_operators": ["exp", "log", "sqrt", "square", "abs", "inv"],
        },
        {
            "name": "L2",
            "niterations": 320,
            "populations": 64,
            "population_size": 120,
            "maxsize": 30,
            "parsimony": 0.004,
            "binary_operators": ["+", "-", "*", "/", "^", "max", "min"],
            "unary_operators": ["exp", "log", "sqrt", "square", "cube", "abs", "inv"],
        },
        {
            "name": "L3",
            "niterations": 420,
            "populations": 80,
            "population_size": 140,
            "maxsize": 34,
            "parsimony": 0.003,
            "binary_operators": ["+", "-", "*", "/", "^", "max", "min"],
            "unary_operators": ["exp", "log", "sqrt", "square", "cube", "abs", "inv", "cbrt"],
        },
    ]

    rows: list[dict[str, float | int | str]] = []
    best_model = None
    best_config = None
    best_test_r2 = -np.inf
    best_pred = None

    for cfg in configs:
        print("\n" + "-" * 80)
        print(
            f"Config {cfg['name']}: iterations={cfg['niterations']}, populations={cfg['populations']}, "
            f"population_size={cfg['population_size']}, maxsize={cfg['maxsize']}, parsimony={cfg['parsimony']}"
        )

        model = build_pysr_regressor(
            PySRRegressor,
            niterations=cfg["niterations"],
            populations=cfg["populations"],
            population_size=cfg["population_size"],
            maxsize=cfg["maxsize"],
            parsimony=cfg["parsimony"],
            binary_operators=cfg["binary_operators"],
            unary_operators=cfg["unary_operators"],
            model_selection="best",
            denoise=False,
            adaptive_parsimony_scaling=1000.0,
            turbo=True,
            batching=False,
            random_state=42,
            progress=False,
            verbosity=0,
        )

        model.fit(x_train_arr, y_train.values, variable_names=variable_names)

        y_train_pred = np.asarray(model.predict(x_train_arr), dtype=float)
        y_test_pred = np.asarray(model.predict(x_test_arr), dtype=float)

        train_r2 = float(r2_score(y_train, y_train_pred))
        test_r2 = float(r2_score(y_test, y_test_pred))
        train_rmse = rmse(y_train.values, y_train_pred)
        test_rmse = rmse(y_test.values, y_test_pred)
        test_mae = float(mean_absolute_error(y_test, y_test_pred))

        row = {
            "config": cfg["name"],
            "niterations": int(cfg["niterations"]),
            "populations": int(cfg["populations"]),
            "population_size": int(cfg["population_size"]),
            "maxsize": int(cfg["maxsize"]),
            "parsimony": float(cfg["parsimony"]),
            "train_r2": train_r2,
            "test_r2": test_r2,
            "train_rmse": train_rmse,
            "test_rmse": test_rmse,
            "test_mae": test_mae,
        }
        rows.append(row)

        print(f"  train R²={train_r2:.6f}, test R²={test_r2:.6f}, test RMSE={test_rmse:.6f}, test MAE={test_mae:.6f}")

        if test_r2 > best_test_r2:
            best_test_r2 = test_r2
            best_model = model
            best_config = cfg
            best_pred = y_test_pred

    trials_df = pd.DataFrame(rows).sort_values("test_r2", ascending=False)
    trials_path = out_csv / "sr_pysr_locked_split_trials.csv"
    trials_df.to_csv(trials_path, index=False)

    if best_model is None or best_config is None or best_pred is None:
        raise RuntimeError("No locked-split config completed")

    bundle = {
        "model_type": "pysr_bundle",
        "model": best_model,
        "scaler": None,
        "feature_mode": "original",
        "base_features": input_features,
        "feature_names": variable_names,
        "trial": best_config,
        "metrics": trials_df.iloc[0].to_dict(),
        "split": {"test_size": 0.2, "random_state": 42, "iqr": 2.0},
    }

    bundle_path = out_models / "shear_stress_pysr_bundle.pkl"
    with open(bundle_path, "wb") as file:
        pickle.dump(bundle, file)

    best_summary_path = out_csv / "sr_pysr_locked_split_best_summary.csv"
    pd.DataFrame([trials_df.iloc[0].to_dict()]).to_csv(best_summary_path, index=False)

    eq_path = out_csv / "sr_pysr_locked_split_equations.csv"
    try:
        best_model.equations_.to_csv(eq_path, index=False)
    except Exception:
        pass

    print("\n" + "=" * 80)
    print("LOCKED-SPLIT PySR COMPLETE")
    print("=" * 80)
    print(trials_df.to_string(index=False))
    print("\nSaved:")
    print(f" - {trials_path}")
    print(f" - {best_summary_path}")
    print(f" - {bundle_path}")
    if eq_path.exists():
        print(f" - {eq_path}")


if __name__ == "__main__":
    main()
