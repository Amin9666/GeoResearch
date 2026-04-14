"""
Linear Regression on the Iris Regression Task.

Predict petal_length (cm) from sepal_length, sepal_width, petal_width
using the classic sklearn iris dataset.

Outputs:
  outputs/figures/iris_lr_actual_vs_predicted.png
  outputs/figures/iris_lr_residuals_scatter.png
  outputs/figures/iris_lr_residuals_hist.png
  outputs/figures/iris_lr_cv_scores.png
  outputs/figures/iris_lr_feature_importance.png
  outputs/csv/iris_lr_metrics.csv
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
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

MODEL_COLOR = "#4878CF"


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


    iris = load_iris(as_frame=True)
    df = iris.frame.copy()
    df.columns = [c.replace(" (cm)", "").replace(" ", "_") for c in df.columns]
    # columns: sepal_length, sepal_width, petal_length, petal_width, target

    feature_cols = ["sepal_length", "sepal_width", "petal_width"]
    target_col = "petal_length"

    X = df[feature_cols]
    y = df[target_col]


    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )


    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    # 4. Train Linear Regression
    model = LinearRegression()
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    # 5. Evaluate
    metrics = evaluate(y_test.values, y_pred)
    print(f"Linear Regression: R²={metrics['R2']:.4f}, RMSE={metrics['RMSE']:.4f}, MAE={metrics['MAE']:.4f}")
    # 6. 10-fold cross-validation
    pipe = Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())])
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipe, X, y, cv=kf, scoring="r2")
    print(f"CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # 7. Plots
    residuals = y_test.values - y_pred
    vmin = min(y_test.min(), y_pred.min())
    vmax = max(y_test.max(), y_pred.max())

    # (a) Actual vs Predicted
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_test, y_pred, alpha=0.6, s=40, color=MODEL_COLOR, edgecolors="white", linewidths=0.5)
    ax.plot([vmin, vmax], [vmin, vmax], "k--", linewidth=1.5, label="Perfect fit")
    ax.set_xlabel("Actual petal_length (cm)", fontsize=12)
    ax.set_ylabel("Predicted petal_length (cm)", fontsize=12)
    ax.set_title("Linear Regression — Actual vs Predicted", fontsize=13)
    ax.text(0.04, 0.96, f"R²={metrics['R2']:.4f}\nRMSE={metrics['RMSE']:.4f}",
            transform=ax.transAxes, ha="left", va="top", fontsize=10,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8})
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_fig / "iris_lr_actual_vs_predicted.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # (b) Residuals vs Predicted
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(y_pred, residuals, alpha=0.6, s=40, color=MODEL_COLOR, edgecolors="white", linewidths=0.5)
    ax.axhline(0, color="black", linestyle="--", linewidth=1.5)
    ax.set_xlabel("Predicted petal_length (cm)", fontsize=12)
    ax.set_ylabel("Residual (Actual − Predicted)", fontsize=12)
    ax.set_title("Linear Regression — Residuals vs Predicted", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_fig / "iris_lr_residuals_scatter.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # (c) Residuals histogram + KDE
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(residuals, bins=20, density=True, alpha=0.6, color=MODEL_COLOR, edgecolor="black", linewidth=0.5)
    sns.kdeplot(residuals, ax=ax, color="darkblue", linewidth=2)
    ax.axvline(0, color="black", linestyle="--", linewidth=1.5)
    ax.set_xlabel("Residual (Actual − Predicted)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Linear Regression — Residual Distribution", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_fig / "iris_lr_residuals_hist.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # (d) CV scores bar chart
    fig, ax = plt.subplots(figsize=(9, 5))
    fold_labels = [f"Fold {i+1}" for i in range(len(cv_scores))]
    bars = ax.bar(fold_labels, cv_scores, color=MODEL_COLOR, edgecolor="black", linewidth=0.5)
    ax.axhline(cv_scores.mean(), color="red", linestyle="--", linewidth=1.5, label=f"Mean={cv_scores.mean():.4f}")
    ax.set_xlabel("CV Fold", fontsize=12)
    ax.set_ylabel("R² Score", fontsize=12)
    ax.set_title("Linear Regression — 10-Fold CV R² Scores", fontsize=13)
    ax.legend(fontsize=10)
    ax.tick_params(axis="x", rotation=45)
    for bar, val in zip(bars, cv_scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_fig / "iris_lr_cv_scores.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # (e) Feature importance (absolute coefficients)
    coef_abs = np.abs(model.coef_)
    sorted_idx = np.argsort(coef_abs)[::-1]
    sorted_features = [feature_cols[i] for i in sorted_idx]
    sorted_coef = coef_abs[sorted_idx]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(sorted_features, sorted_coef, color=MODEL_COLOR, edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Feature", fontsize=12)
    ax.set_ylabel("|Coefficient|", fontsize=12)
    ax.set_title("Linear Regression — Feature Importance (|Coefficient|)", fontsize=13)
    for bar, val in zip(bars, sorted_coef):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.4f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_fig / "iris_lr_feature_importance.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------
    # 8. Save metrics CSV
    # ------------------------------------------------------------------
    metrics_df = pd.DataFrame([{"Model": "Linear Regression", **metrics,
                                 "CV_R2_mean": cv_scores.mean(), "CV_R2_std": cv_scores.std()}])
    metrics_df.to_csv(out_csv / "iris_lr_metrics.csv", index=False)

    # ------------------------------------------------------------------
    # 9. Print summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("LINEAR REGRESSION — IRIS REGRESSION SUMMARY")
    print("=" * 60)
    print(f"  R²   : {metrics['R2']:.4f}")
    print(f"  RMSE : {metrics['RMSE']:.4f} cm")
    print(f"  MAE  : {metrics['MAE']:.4f} cm")
    print(f"  CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"\nCoefficients:")
    for feat, coef in zip(feature_cols, model.coef_):
        print(f"  {feat:20s}: {coef:+.6f}")
    print(f"  {'intercept':20s}: {model.intercept_:+.6f}")
    print(f"\nSaved figures → {out_fig}")
    print(f"Saved metrics → {out_csv / 'iris_lr_metrics.csv'}")


if __name__ == "__main__":
    main()
