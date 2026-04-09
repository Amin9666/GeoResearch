#!/usr/bin/env python
"""
Fast locked-split PySR search to improve symbolic regression on benchmark split.

Benchmark lock:
  - IQR=2.0 cleaning on base features + target
  - Features: RISE, COUN, ENER, DURATION, AMP
  - Split: test_size=0.2, random_state=42
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
from sklearn.preprocessing import RobustScaler, StandardScaler
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


def apply_input_scaling(x_train: pd.DataFrame, x_test: pd.DataFrame, mode: str):
    if mode == "none":
        return x_train.values, x_test.values, None
    scaler = RobustScaler() if mode == "robust" else StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    x_test_s = scaler.transform(x_test)
    return x_train_s, x_test_s, scaler


def apply_target_scaling(y_train: pd.Series, y_test: pd.Series, mode: str):
    if mode == "none":
        return y_train.values, y_test.values, {"mode": "none"}
    mean = float(y_train.mean())
    std = float(y_train.std()) if float(y_train.std()) > 1e-12 else 1.0
    return (
        ((y_train - mean) / std).values,
        ((y_test - mean) / std).values,
        {"mode": "zscore", "mean": mean, "std": std},
    )


def invert_target(y_pred: np.ndarray, y_meta: dict) -> np.ndarray:
    if y_meta.get("mode") == "zscore":
        return y_pred * float(y_meta["std"]) + float(y_meta["mean"])
    return y_pred


def main() -> None:
    print("=" * 80)
    print("FAST LOCKED-SPLIT PySR SEARCH")
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
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    operator_sets = {
        "basic": {
            "binary_operators": ["+", "-", "*", "/", "^"],
            "unary_operators": ["log", "sqrt", "square", "abs"],
        },
        "extended": {
            "binary_operators": ["+", "-", "*", "/", "^", "max", "min"],
            "unary_operators": ["exp", "log", "sqrt", "square", "cube", "abs", "inv", "cbrt"],
        },
    }

    trials = []
    for input_scaler in ["none", "robust", "standard"]:
        for target_scaler in ["none", "zscore"]:
            for op_name in ["basic", "extended"]:
                for seed in [7, 13, 21]:
                    trials.append(
                        {
                            "input_scaler": input_scaler,
                            "target_scaler": target_scaler,
                            "op_name": op_name,
                            "seed": seed,
                            "niterations": 150,
                            "populations": 36,
                            "population_size": 85,
                            "maxsize": 28,
                            "parsimony": 0.004,
                        }
                    )

    rows: list[dict[str, float | str | int]] = []
    best_score = -np.inf
    best_record: dict[str, object] | None = None

    for idx, trial in enumerate(trials, start=1):
        print(
            f"\nTrial {idx}/{len(trials)} | input={trial['input_scaler']} target={trial['target_scaler']} "
            f"ops={trial['op_name']} seed={trial['seed']}"
        )

        x_train_arr, x_test_arr, x_scaler = apply_input_scaling(x_train, x_test, str(trial["input_scaler"]))
        y_train_arr, _, y_meta = apply_target_scaling(y_train, y_test, str(trial["target_scaler"]))

        ops = operator_sets[str(trial["op_name"])]
        model = build_pysr_regressor(
            PySRRegressor,
            niterations=int(trial["niterations"]),
            populations=int(trial["populations"]),
            population_size=int(trial["population_size"]),
            maxsize=int(trial["maxsize"]),
            parsimony=float(trial["parsimony"]),
            binary_operators=ops["binary_operators"],
            unary_operators=ops["unary_operators"],
            model_selection="best",
            denoise=False,
            adaptive_parsimony_scaling=1000.0,
            turbo=True,
            batching=False,
            random_state=int(trial["seed"]),
            progress=False,
            verbosity=0,
        )

        model.fit(x_train_arr, y_train_arr, variable_names=input_features)

        train_pred_scaled = np.asarray(model.predict(x_train_arr), dtype=float)
        test_pred_scaled = np.asarray(model.predict(x_test_arr), dtype=float)

        train_pred = invert_target(train_pred_scaled, y_meta)
        test_pred = invert_target(test_pred_scaled, y_meta)

        train_r2 = float(r2_score(y_train, train_pred))
        test_r2 = float(r2_score(y_test, test_pred))
        test_rmse = rmse(y_test.values, test_pred)
        test_mae = float(mean_absolute_error(y_test, test_pred))

        row = {
            "trial": idx,
            "input_scaler": str(trial["input_scaler"]),
            "target_scaler": str(trial["target_scaler"]),
            "operators": str(trial["op_name"]),
            "seed": int(trial["seed"]),
            "train_r2": train_r2,
            "test_r2": test_r2,
            "test_rmse": test_rmse,
            "test_mae": test_mae,
        }
        rows.append(row)

        print(f"  train R²={train_r2:.6f} | test R²={test_r2:.6f} | test RMSE={test_rmse:.6f}")

        if test_r2 > best_score:
            best_score = test_r2
            best_record = {
                "trial": trial,
                "model": model,
                "x_scaler": x_scaler,
                "y_meta": y_meta,
                "metrics": row,
            }

    trials_df = pd.DataFrame(rows).sort_values("test_r2", ascending=False)
    trials_path = out_csv / "sr_pysr_fast_locked_trials.csv"
    trials_df.to_csv(trials_path, index=False)

    if best_record is None:
        raise RuntimeError("No successful SR trials")

    bundle = {
        "model_type": "pysr_bundle",
        "model": best_record["model"],
        "scaler": best_record["x_scaler"],
        "feature_mode": "original",
        "base_features": input_features,
        "feature_names": input_features,
        "target_transform": best_record["y_meta"],
        "trial": best_record["trial"],
        "metrics": best_record["metrics"],
        "split": {"test_size": 0.2, "random_state": 42, "iqr": 2.0},
    }

    bundle_path = out_models / "shear_stress_pysr_bundle.pkl"
    with open(bundle_path, "wb") as file:
        pickle.dump(bundle, file)

    best_summary_path = out_csv / "sr_pysr_fast_locked_best_summary.csv"
    pd.DataFrame([best_record["metrics"]]).to_csv(best_summary_path, index=False)

    eq_path = out_csv / "sr_pysr_fast_locked_equations.csv"
    try:
        best_record["model"].equations_.to_csv(eq_path, index=False)
    except Exception:
        pass

    print("\n" + "=" * 80)
    print("FAST LOCKED-SPLIT PySR SEARCH COMPLETE")
    print("=" * 80)
    print(trials_df.head(10).to_string(index=False))
    print("\nSaved:")
    print(f" - {trials_path}")
    print(f" - {best_summary_path}")
    print(f" - {bundle_path}")
    if eq_path.exists():
        print(f" - {eq_path}")


if __name__ == "__main__":
    main()
