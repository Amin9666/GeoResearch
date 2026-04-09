#!/usr/bin/env python
"""
PySR Best Push: targeted symbolic-regression optimization.

This script runs a compact search across preprocessing and feature variants,
then saves the best symbolic-regression artifact bundle for deployment/comparison.
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
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

warnings.filterwarnings("ignore")


def remove_outliers_iqr(data: pd.DataFrame, features: list[str], iqr_multiplier: float) -> pd.DataFrame:
    mask = np.ones(len(data), dtype=bool)
    for feature in features:
        q1 = data[feature].quantile(0.25)
        q3 = data[feature].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - iqr_multiplier * iqr
        upper = q3 + iqr_multiplier * iqr
        mask &= (data[feature] >= lower) & (data[feature] <= upper)
    return data[mask].reset_index(drop=True)


def make_features(x: pd.DataFrame, mode: str) -> pd.DataFrame:
    if mode == "original":
        return x.copy()

    x_feat = x.copy()
    for col in ["RISE", "COUN", "ENER", "DURATION", "AMP"]:
        x_feat[f"{col}_sq"] = x[col] ** 2
        if (x[col] > 0).all():
            x_feat[f"log_{col}"] = np.log(x[col] + 1e-8)
            x_feat[f"sqrt_{col}"] = np.sqrt(x[col])

    x_feat["AMP_x_RISE"] = x["AMP"] * x["RISE"]
    x_feat["ENER_x_DURATION"] = x["ENER"] * x["DURATION"]
    x_feat["AMP_DURATION_ratio"] = x["AMP"] / (x["DURATION"] + 1e-6)
    return x_feat


def apply_scaler(x_train: pd.DataFrame, x_test: pd.DataFrame, scaler_name: str):
    if scaler_name == "none":
        return x_train.values, x_test.values, None

    scaler = RobustScaler() if scaler_name == "robust" else StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    x_test_s = scaler.transform(x_test)
    return x_train_s, x_test_s, scaler


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def build_pysr_regressor(regressor_cls, **kwargs):
    valid = set(inspect.signature(regressor_cls.__init__).parameters.keys())
    filtered = {k: v for k, v in kwargs.items() if k in valid}
    return regressor_cls(**filtered)


def main() -> None:
    print("=" * 80)
    print("PySR BEST PUSH - SYMBOLIC REGRESSION OPTIMIZATION")
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
    base_features = ["RISE", "COUN", "ENER", "DURATION", "AMP"]
    target = "SHEAR STRESS"

    trials = [
        {"name": "A", "iqr": 2.0, "feat": "original", "scaler": "none", "niterations": 120},
        {"name": "B", "iqr": 2.0, "feat": "engineered", "scaler": "robust", "niterations": 120},
        {"name": "C", "iqr": 1.5, "feat": "engineered", "scaler": "robust", "niterations": 140},
        {"name": "D", "iqr": 1.0, "feat": "engineered", "scaler": "standard", "niterations": 160},
    ]

    all_rows: list[dict[str, float | str | int]] = []
    best_record: dict[str, object] | None = None
    best_r2 = -np.inf

    for trial in trials:
        print("\n" + "-" * 80)
        print(
            f"Trial {trial['name']}: iqr={trial['iqr']}, features={trial['feat']}, "
            f"scaler={trial['scaler']}, iterations={trial['niterations']}"
        )

        df_clean = remove_outliers_iqr(df, base_features + [target], iqr_multiplier=float(trial["iqr"]))
        x_base = df_clean[base_features]
        y = df_clean[target]

        x_feat = make_features(x_base, str(trial["feat"]))
        feat_names = list(x_feat.columns)

        x_train, x_test, y_train, y_test = train_test_split(
            x_feat, y, test_size=0.2, random_state=42
        )
        x_train_arr, x_test_arr, scaler_obj = apply_scaler(x_train, x_test, str(trial["scaler"]))

        model = build_pysr_regressor(
            PySRRegressor,
            niterations=int(trial["niterations"]),
            populations=36,
            population_size=90,
            maxsize=32,
            parsimony=0.004,
            binary_operators=["+", "-", "*", "/", "^", "max", "min"],
            unary_operators=["exp", "log", "sqrt", "square", "abs", "inv"],
            model_selection="best",
            denoise=False,
            adaptive_parsimony_scaling=1000.0,
            turbo=True,
            batching=False,
            random_state=42,
            progress=False,
            verbosity=0,
        )

        model.fit(x_train_arr, y_train.values, variable_names=feat_names)

        y_train_pred = np.asarray(model.predict(x_train_arr), dtype=float)
        y_test_pred = np.asarray(model.predict(x_test_arr), dtype=float)

        train_r2 = float(r2_score(y_train, y_train_pred))
        test_r2 = float(r2_score(y_test, y_test_pred))
        train_rmse = rmse(y_train.values, y_train_pred)
        test_rmse = rmse(y_test.values, y_test_pred)
        test_mae = float(mean_absolute_error(y_test, y_test_pred))

        gb = GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=4,
            subsample=0.9,
            random_state=42,
        )
        gb.fit(x_train_arr, y_train)
        gb_r2 = float(r2_score(y_test, gb.predict(x_test_arr)))

        rf = RandomForestRegressor(n_estimators=300, max_depth=15, random_state=42, n_jobs=-1)
        rf.fit(x_train_arr, y_train)
        rf_r2 = float(r2_score(y_test, rf.predict(x_test_arr)))

        row = {
            "trial": str(trial["name"]),
            "iqr": float(trial["iqr"]),
            "feature_mode": str(trial["feat"]),
            "scaler": str(trial["scaler"]),
            "niterations": int(trial["niterations"]),
            "n_samples": int(len(df_clean)),
            "n_features": int(x_feat.shape[1]),
            "train_r2": train_r2,
            "test_r2": test_r2,
            "train_rmse": train_rmse,
            "test_rmse": test_rmse,
            "test_mae": test_mae,
            "gb_test_r2_same_split": gb_r2,
            "rf_test_r2_same_split": rf_r2,
        }
        all_rows.append(row)

        print(
            f"  test R²={test_r2:.6f}, RMSE={test_rmse:.6f}, MAE={test_mae:.6f} | "
            f"GB(R²)={gb_r2:.6f}, RF(R²)={rf_r2:.6f}"
        )

        if test_r2 > best_r2:
            best_r2 = test_r2
            best_record = {
                "trial": trial,
                "model": model,
                "scaler": scaler_obj,
                "feature_mode": str(trial["feat"]),
                "feature_names": feat_names,
                "metrics": row,
            }

    trials_df = pd.DataFrame(all_rows).sort_values("test_r2", ascending=False)
    trials_path = out_csv / "sr_pysr_best_push_trials.csv"
    trials_df.to_csv(trials_path, index=False)

    if best_record is None:
        raise RuntimeError("No PySR trial completed.")

    best_bundle = {
        "model_type": "pysr_bundle",
        "model": best_record["model"],
        "scaler": best_record["scaler"],
        "feature_mode": best_record["feature_mode"],
        "base_features": base_features,
        "feature_names": best_record["feature_names"],
        "trial": best_record["trial"],
        "metrics": best_record["metrics"],
    }

    bundle_path = out_models / "shear_stress_pysr_bundle.pkl"
    with open(bundle_path, "wb") as file:
        pickle.dump(best_bundle, file)

    best_summary = pd.DataFrame([best_record["metrics"]])
    best_summary_path = out_csv / "sr_pysr_best_push_best_summary.csv"
    best_summary.to_csv(best_summary_path, index=False)

    best_model = best_record["model"]
    equations_path = out_csv / "sr_pysr_best_push_equations.csv"
    try:
        best_model.equations_.to_csv(equations_path, index=False)
    except Exception:
        pass

    print("\n" + "=" * 80)
    print("BEST PySR PUSH COMPLETE")
    print("=" * 80)
    print(trials_df.to_string(index=False))
    print("\nSaved:")
    print(f" - {trials_path}")
    print(f" - {best_summary_path}")
    print(f" - {bundle_path}")
    if equations_path.exists():
        print(f" - {equations_path}")


if __name__ == "__main__":
    main()
