"""
Symbolic Regression via PySR — Iris Regression Task.

Predict petal_length (cm) from sepal_length, sepal_width, petal_width.
Uses an evolutionary algorithm (PySR) to discover the actual mathematical
structure of the formula.

NOTE: PySR requires Julia to be installed on your system.
See the README for Julia setup instructions.

Outputs:
  outputs/figures/iris_sr_actual_vs_predicted.png
  outputs/figures/iris_sr_residuals_scatter.png
  outputs/figures/iris_sr_residuals_hist.png
  outputs/csv/iris_sr_formula.csv
  outputs/csv/iris_sr_metrics.csv
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pysr import PySRRegressor
from sklearn.datasets import load_iris
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

MODEL_COLOR = "#C4AD66"


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out_fig = root / "outputs" / "figures"
    out_csv = root / "outputs" / "csv"
    for d in (out_fig, out_csv):
        d.mkdir(parents=True, exist_ok=True)

    sns.set_style("whitegrid")

    # 1. Load data
    iris = load_iris(as_frame=True)
    df = iris.frame.copy()
    df.columns = [c.replace(" (cm)", "").replace(" ", "_") for c in df.columns]

    feature_cols = ["sepal_length", "sepal_width", "petal_width"]
    target_col = "petal_length"

    X = df[feature_cols]
    y = df[target_col]

    # 2. 80/20 split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3. Train PySR (true symbolic regression)
    model = PySRRegressor(
        niterations=40,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["sqrt", "log", "square"],
        random_state=42,
        verbosity=0,
    )
    model.fit(X_train.values, y_train.values)
    y_pred = model.predict(X_test.values)

    # 4. Evaluate
    metrics = evaluate(y_test.values, y_pred)
    print(f"Symbolic Regression (PySR): R²={metrics['R2']:.4f}, RMSE={metrics['RMSE']:.4f}, MAE={metrics['MAE']:.4f}")

    # 5. Extract best equation string
    best_equation = str(model.sympy())
    print(f"\nDiscovered equation: petal_length = {best_equation}")

    # 6. Plots
    residuals = y_test.values - y_pred
    vmin = min(y_test.values.min(), y_pred.min())
    vmax = max(y_test.values.max(), y_pred.max())

    # (a) Actual vs Predicted
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_test.values, y_pred, alpha=0.6, s=40, color=MODEL_COLOR, edgecolors="white", linewidths=0.5)
    ax.plot([vmin, vmax], [vmin, vmax], "k--", linewidth=1.5, label="Perfect fit")
    ax.set_xlabel("Actual petal_length (cm)", fontsize=12)
    ax.set_ylabel("Predicted petal_length (cm)", fontsize=12)
    ax.set_title("Symbolic Regression (PySR) — Actual vs Predicted", fontsize=13)
    ax.text(0.04, 0.96, f"R²={metrics['R2']:.4f}\nRMSE={metrics['RMSE']:.4f}",
            transform=ax.transAxes, ha="left", va="top", fontsize=10,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8})
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_fig / "iris_sr_actual_vs_predicted.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # (b) Residuals vs Predicted
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(y_pred, residuals, alpha=0.6, s=40, color=MODEL_COLOR, edgecolors="white", linewidths=0.5)
    ax.axhline(0, color="black", linestyle="--", linewidth=1.5)
    ax.set_xlabel("Predicted petal_length (cm)", fontsize=12)
    ax.set_ylabel("Residual (Actual − Predicted)", fontsize=12)
    ax.set_title("Symbolic Regression — Residuals vs Predicted", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_fig / "iris_sr_residuals_scatter.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # (c) Residuals histogram + KDE
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(residuals, bins=20, density=True, alpha=0.6, color=MODEL_COLOR, edgecolor="black", linewidth=0.5)
    sns.kdeplot(residuals, ax=ax, color="saddlebrown", linewidth=2)
    ax.axvline(0, color="black", linestyle="--", linewidth=1.5)
    ax.set_xlabel("Residual (Actual − Predicted)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Symbolic Regression — Residual Distribution", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_fig / "iris_sr_residuals_hist.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------
    # 7. Save CSVs
    # ------------------------------------------------------------------
    formula_df = pd.DataFrame({"equation": [best_equation]})
    formula_df.to_csv(out_csv / "iris_sr_formula.csv", index=False)

    metrics_df = pd.DataFrame([{"Model": "Symbolic Regression (PySR)", **metrics}])
    metrics_df.to_csv(out_csv / "iris_sr_metrics.csv", index=False)

    # 8. Print summary
    print("\n" + "=" * 60)
    print("SYMBOLIC REGRESSION (PySR) — IRIS REGRESSION SUMMARY")
    print("=" * 60)
    print(f"  R²   : {metrics['R2']:.4f}")
    print(f"  RMSE : {metrics['RMSE']:.4f} cm")
    print(f"  MAE  : {metrics['MAE']:.4f} cm")
    print(f"\nDiscovered Formula:")
    print(f"  petal_length = {best_equation}")
    print(f"\nSaved figures → {out_fig}")
    print(f"Saved formula → {out_csv / 'iris_sr_formula.csv'}")
    print(f"Saved metrics → {out_csv / 'iris_sr_metrics.csv'}")


if __name__ == "__main__":
    main()
