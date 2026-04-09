"""
Compare Linear Regression vs Three ML Algorithms vs Symbolic Regression.

Outputs:
  - outputs/csv/linear_ml_symbolic_metrics.csv
  - outputs/csv/linear_ml_symbolic_predictions.csv
  - outputs/figures/linear_ml_symbolic_metrics.png
  - outputs/figures/linear_ml_symbolic_actual_vs_predicted.png
    - outputs/figures/linear_ml_symbolic_residual_distributions.png
"""

from __future__ import annotations

from pathlib import Path
import os
import shutil
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
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


def evaluate(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": rmse,
        "MAE": float(mean_absolute_error(y_true, y_pred)),
    }


def make_symbolic_features(x: pd.DataFrame, mode: str) -> pd.DataFrame:
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


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data_path = root / "data" / "raw" / "Shear_Data_15.csv"
    if not data_path.exists():
        data_path = root / "Shear_Data_15.csv"

    out_csv_dir = root / "outputs" / "csv"
    out_fig_dir = root / "outputs" / "figures"
    out_csv_dir.mkdir(parents=True, exist_ok=True)
    out_fig_dir.mkdir(parents=True, exist_ok=True)

    system_julia = shutil.which("julia")
    if system_julia:
        os.environ.setdefault("PYTHON_JULIAPKG_EXE", system_julia)
        os.environ.setdefault("PYTHON_JULIACALL_EXE", system_julia)

    df = pd.read_csv(data_path)

    input_features = ["RISE", "COUN", "ENER", "DURATION", "AMP"]
    target = "SHEAR STRESS"

    df = remove_outliers_iqr(df, input_features + [target], iqr_multiplier=2.0)
    x = df[input_features]
    y = df[target]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    models = {
        "Linear Regression": Pipeline(
            [("scaler", RobustScaler()), ("model", LinearRegression())]
        ),
        "Ridge Regression": Pipeline(
            [("scaler", RobustScaler()), ("model", Ridge(alpha=1.0))]
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=300, max_depth=15, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=4,
            subsample=0.9,
            random_state=42,
        ),
    }

    predictions: dict[str, np.ndarray] = {}
    metrics_rows: list[dict[str, float | str]] = []

    for name, model in models.items():
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        predictions[name] = y_pred
        m = evaluate(y_test, y_pred)
        metrics_rows.append({"Model": name, "Source": "scikit-learn", **m})

    symbolic_name = "Symbolic Regression"
    symbolic_source = ""
    symbolic_pred: np.ndarray | None = None

    symbolic_bundle_path = root / "outputs" / "models" / "shear_stress_pysr_bundle.pkl"
    symbolic_model_path = root / "outputs" / "models" / "shear_stress_pysr_model.pkl"

    try:
        import pickle

        if symbolic_bundle_path.exists():
            with open(symbolic_bundle_path, "rb") as file:
                bundle = pickle.load(file)

            if isinstance(bundle, dict) and bundle.get("model_type") == "pysr_bundle":
                feature_mode = str(bundle.get("feature_mode", "original"))
                x_sr = make_symbolic_features(x_test, feature_mode)
                scaler = bundle.get("scaler")
                x_sr_arr = scaler.transform(x_sr) if scaler is not None else x_sr.values
                y_pred_symbolic = np.asarray(bundle["model"].predict(x_sr_arr), dtype=float)

                target_transform = bundle.get("target_transform")
                if isinstance(target_transform, dict) and target_transform.get("mode") == "zscore":
                    y_pred_symbolic = (
                        y_pred_symbolic * float(target_transform.get("std", 1.0))
                        + float(target_transform.get("mean", 0.0))
                    )

                if np.any(~np.isfinite(y_pred_symbolic)):
                    raise ValueError("PySR bundle returned non-finite values")

                symbolic_pred = y_pred_symbolic
                symbolic_source = f"PySR bundle: {symbolic_bundle_path.name}"

        if symbolic_pred is None and symbolic_model_path.exists():
            with open(symbolic_model_path, "rb") as file:
                symbolic_model = pickle.load(file)
            y_pred_symbolic = symbolic_model.predict(x_test.to_numpy())
            y_pred_symbolic = np.asarray(y_pred_symbolic, dtype=float)

            if np.any(~np.isfinite(y_pred_symbolic)):
                raise ValueError("PySR model returned non-finite values")

            symbolic_pred = y_pred_symbolic
            symbolic_source = f"PySR model: {symbolic_model_path.name}"
    except Exception:
        symbolic_pred = None

    if symbolic_pred is None:
        try:
            from shear_stress_equation import predict_shear_stress

            y_pred_symbolic = predict_shear_stress(
                x_test["RISE"].to_numpy(),
                x_test["COUN"].to_numpy(),
                x_test["ENER"].to_numpy(),
                x_test["DURATION"].to_numpy(),
                x_test["AMP"].to_numpy(),
            )
            y_pred_symbolic = np.asarray(y_pred_symbolic, dtype=float)

            if np.any(~np.isfinite(y_pred_symbolic)):
                raise ValueError("Symbolic equation returned non-finite values")

            symbolic_pred = y_pred_symbolic
            symbolic_source = "Symbolic equation file: shear_stress_equation.py"
        except Exception:
            symbolic_name = "Symbolic Regression (Polynomial Proxy)"
            symbolic_proxy = Pipeline(
                [
                    ("scaler", RobustScaler()),
                    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                    ("model", Ridge(alpha=0.1)),
                ]
            )
            symbolic_proxy.fit(x_train, y_train)
            symbolic_pred = symbolic_proxy.predict(x_test)
            symbolic_source = "Fallback proxy: PolynomialFeatures + Ridge"

    predictions[symbolic_name] = symbolic_pred
    m_symbolic = evaluate(y_test, symbolic_pred)
    metrics_rows.append({"Model": symbolic_name, "Source": symbolic_source, **m_symbolic})

    metrics_df = pd.DataFrame(metrics_rows).sort_values("R2", ascending=False)
    metrics_csv = out_csv_dir / "linear_ml_symbolic_metrics.csv"
    metrics_df.to_csv(metrics_csv, index=False)

    predictions_df = pd.DataFrame({"Actual": y_test.to_numpy()})
    for name, y_pred in predictions.items():
        safe_col = name.replace(" ", "_").replace("(", "").replace(")", "")
        predictions_df[safe_col] = y_pred
    pred_csv = out_csv_dir / "linear_ml_symbolic_predictions.csv"
    predictions_df.to_csv(pred_csv, index=False)

    sns.set_style("whitegrid")
    plt.rcParams["figure.figsize"] = (14, 6)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.barplot(data=metrics_df, x="Model", y="R2", ax=axes[0], palette="viridis")
    axes[0].set_title("Model Comparison by R²")
    axes[0].set_xlabel("")
    axes[0].tick_params(axis="x", rotation=25)

    sns.barplot(data=metrics_df, x="Model", y="RMSE", ax=axes[1], palette="magma")
    axes[1].set_title("Model Comparison by RMSE")
    axes[1].set_xlabel("")
    axes[1].tick_params(axis="x", rotation=25)

    fig.tight_layout()
    metrics_fig = out_fig_dir / "linear_ml_symbolic_metrics.png"
    fig.savefig(metrics_fig, dpi=300, bbox_inches="tight")
    plt.close(fig)

    n_models = len(predictions)
    cols = 3
    rows = int(np.ceil(n_models / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = np.array(axes).reshape(rows, cols)

    min_val = float(min(y_test.min(), min(np.min(v) for v in predictions.values())))
    max_val = float(max(y_test.max(), max(np.max(v) for v in predictions.values())))

    for idx, (name, y_pred) in enumerate(predictions.items()):
        r = idx // cols
        c = idx % cols
        ax = axes[r, c]
        ax.scatter(y_test, y_pred, alpha=0.55, s=18)
        ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1.5)
        ax.set_title(name)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        m = evaluate(y_test, y_pred)
        ax.text(
            0.03,
            0.97,
            f"R²={m['R2']:.4f}\nRMSE={m['RMSE']:.4f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.75},
        )

    for idx in range(n_models, rows * cols):
        r = idx // cols
        c = idx % cols
        axes[r, c].axis("off")

    fig.suptitle("Actual vs Predicted: Linear, ML Baselines, and Symbolic Regression", y=1.02)
    fig.tight_layout()
    scatter_fig = out_fig_dir / "linear_ml_symbolic_actual_vs_predicted.png"
    fig.savefig(scatter_fig, dpi=300, bbox_inches="tight")
    plt.close(fig)

    residuals_fig = out_fig_dir / "linear_ml_symbolic_residual_distributions.png"
    fig, ax = plt.subplots(figsize=(12, 6))
    bins = 40
    for name, y_pred in predictions.items():
        residuals = y_test.to_numpy() - y_pred
        ax.hist(residuals, bins=bins, alpha=0.35, label=name, density=True)
    ax.axvline(0, color="black", linestyle="--", linewidth=1.5)
    ax.set_title("Residual Distribution Comparison")
    ax.set_xlabel("Residual (Actual - Predicted)")
    ax.set_ylabel("Density")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(residuals_fig, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("=" * 72)
    print("COMPARISON COMPLETE")
    print("=" * 72)
    print(metrics_df.to_string(index=False))
    print(f"\nSymbolic source used: {symbolic_source}")
    print("\nSaved files:")
    print(f" - {metrics_csv}")
    print(f" - {pred_csv}")
    print(f" - {metrics_fig}")
    print(f" - {scatter_fig}")
    print(f" - {residuals_fig}")


if __name__ == "__main__":
    main()
