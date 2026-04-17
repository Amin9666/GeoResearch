"""
Master Comparison: Linear Regression vs Symbolic Regression (PySR)
vs Random Forest vs MLP Regressor on the Iris Regression Task.

Predict petal_length (cm) from sepal_length, sepal_width, petal_width.

NOTE: PySR requires Julia to be installed on your system.
See the README for Julia setup instructions.

Outputs (outputs/figures/):
  iris_comparison_metrics_bar.png          – R², RMSE, MAE for all models
  iris_comparison_actual_vs_predicted.png  – 2×2 actual vs predicted
  iris_comparison_error_distribution_boxplot.png – absolute error spread
  iris_comparison_cv_r2_bar.png            – 10-fold CV R² with error bars

Outputs (outputs/csv/):
  iris_comparison_metrics.csv
  iris_comparison_cv_metrics.csv
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
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# Consistent colour palette
PALETTE = {
    "Linear Regression": "#4878CF",
    "Symbolic Regression": "#C4AD66",
    "Random Forest": "#D65F5F",
    "MLP": "#6ACC65",
}
MODEL_NAMES = list(PALETTE.keys())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out_fig = root / "outputs" / "figures"
    out_csv = root / "outputs" / "csv"
    for d in (out_fig, out_csv):
        d.mkdir(parents=True, exist_ok=True)

    sns.set_style("whitegrid")

    # ------------------------------------------------------------------
    # 1. Load iris dataset
    # ------------------------------------------------------------------
    iris = load_iris(as_frame=True)
    df = iris.frame.copy()
    df.columns = [c.replace(" (cm)", "").replace(" ", "_") for c in df.columns]

    feature_cols = ["sepal_length", "sepal_width", "petal_width"]
    target_col = "petal_length"

    X = df[feature_cols]
    y = df[target_col]

    # ------------------------------------------------------------------
    # 2. 80/20 train/test split (same random_state=42)
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    y_train_np = y_train.values
    y_test_np = y_test.values

    # ------------------------------------------------------------------
    # 3. Scale raw features for LR and MLP
    # ------------------------------------------------------------------
    scaler_std = StandardScaler()
    X_train_std = scaler_std.fit_transform(X_train)
    X_test_std = scaler_std.transform(X_test)

    # ------------------------------------------------------------------
    # 4. Train all 4 models
    # ------------------------------------------------------------------
    print("Training models...")

    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train_std, y_train_np)
    y_pred_lr = lr.predict(X_test_std)

    # Symbolic Regression (PySR — true symbolic regression)
    sr = PySRRegressor(
        niterations=40,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["sqrt", "log", "square"],
        random_state=42,
        verbosity=0,
    )
    sr.fit(X_train.values, y_train_np)
    y_pred_sr = sr.predict(X_test.values)

    # Random Forest
    rf = RandomForestRegressor(n_estimators=200, max_depth=None, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train_np)
    y_pred_rf = rf.predict(X_test)

    # MLP
    mlp = MLPRegressor(
        hidden_layer_sizes=(64, 32), activation="relu", max_iter=2000,
        random_state=42, early_stopping=True, validation_fraction=0.1,
    )
    mlp.fit(X_train_std, y_train_np)
    y_pred_mlp = mlp.predict(X_test_std)

    predictions = {
        "Linear Regression": y_pred_lr,
        "Symbolic Regression": y_pred_sr,
        "Random Forest": y_pred_rf,
        "MLP": y_pred_mlp,
    }

    # ------------------------------------------------------------------
    # 5. Compute metrics
    # ------------------------------------------------------------------
    metrics_rows = []
    for name, y_pred in predictions.items():
        m = evaluate(y_test_np, y_pred)
        metrics_rows.append({"Model": name, **m})
        print(f"{name}: R²={m['R2']:.4f}, RMSE={m['RMSE']:.4f}, MAE={m['MAE']:.4f}")
    metrics_df = pd.DataFrame(metrics_rows).sort_values("R2", ascending=False)

    # ------------------------------------------------------------------
    # 6. 10-fold CV for all 4 models
    # ------------------------------------------------------------------
    print("\nRunning 10-fold cross-validation...")
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    pipe_lr = Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())])
    pipe_rf = Pipeline([("model", RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1))])
    pipe_mlp = Pipeline([
        ("scaler", StandardScaler()),
        ("model", MLPRegressor(hidden_layer_sizes=(64, 32), activation="relu",
                               max_iter=500, random_state=42)),
    ])

    cv_scores_lr = cross_val_score(pipe_lr, X, y, cv=kf, scoring="r2")
    cv_scores_rf = cross_val_score(pipe_rf, X, y, cv=kf, scoring="r2")
    cv_scores_mlp = cross_val_score(pipe_mlp, X, y, cv=kf, scoring="r2")

    # SR CV (manual 10-fold, PySR doesn't integrate with cross_val_score)
    cv_scores_sr = []
    for tr_idx, ts_idx in kf.split(X.values):
        m_cv = PySRRegressor(
            niterations=40,
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["sqrt", "log", "square"],
            random_state=42,
            verbosity=0,
        )
        m_cv.fit(X.values[tr_idx], y.values[tr_idx])
        cv_scores_sr.append(r2_score(y.values[ts_idx], m_cv.predict(X.values[ts_idx])))
    cv_scores_sr = np.array(cv_scores_sr)

    cv_all = {
        "Linear Regression": cv_scores_lr,
        "Symbolic Regression": cv_scores_sr,
        "Random Forest": cv_scores_rf,
        "MLP": cv_scores_mlp,
    }
    cv_rows = []
    for name, scores in cv_all.items():
        print(f"  {name}: CV R²={scores.mean():.4f} ± {scores.std():.4f}")
        cv_rows.append({"Model": name, "CV_R2_mean": scores.mean(), "CV_R2_std": scores.std()})
    cv_df = pd.DataFrame(cv_rows).sort_values("CV_R2_mean", ascending=False)

    # ------------------------------------------------------------------
    # 7. Save CSVs
    # ------------------------------------------------------------------
    metrics_df.to_csv(out_csv / "iris_comparison_metrics.csv", index=False)
    cv_df.to_csv(out_csv / "iris_comparison_cv_metrics.csv", index=False)

    # ------------------------------------------------------------------
    # 8. Plots
    # ------------------------------------------------------------------

    # --- (a) Grouped bar chart: R², RMSE, MAE for all models ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, metric in zip(axes, ["R2", "RMSE", "MAE"]):
        bar_colors = [PALETTE[m] for m in metrics_df["Model"]]
        bars = ax.bar(metrics_df["Model"], metrics_df[metric],
                      color=bar_colors, edgecolor="black", linewidth=0.5)
        ax.set_title(f"Model Comparison — {metric}", fontsize=12)
        ax.tick_params(axis="x", rotation=30)
        for bar, val in zip(bars, metrics_df[metric]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle("Iris Regression — Model Comparison Metrics", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_fig / "iris_comparison_metrics_bar.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # --- (b) 2×2 Actual vs Predicted ---
    vmin_global = min(y_test_np.min(), *(p.min() for p in predictions.values()))
    vmax_global = max(y_test_np.max(), *(p.max() for p in predictions.values()))

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
    for ax, name in zip(axes.flat, MODEL_NAMES):
        y_pred = predictions[name]
        m = evaluate(y_test_np, y_pred)
        ax.scatter(y_test_np, y_pred, alpha=0.6, s=30, color=PALETTE[name],
                   edgecolors="white", linewidths=0.4)
        ax.plot([vmin_global, vmax_global], [vmin_global, vmax_global], "k--", linewidth=1.5)
        ax.set_title(name, fontsize=11)
        ax.set_xlabel("Actual (cm)")
        ax.set_ylabel("Predicted (cm)")
        ax.text(0.04, 0.96, f"R²={m['R2']:.4f}\nRMSE={m['RMSE']:.4f}",
                transform=ax.transAxes, ha="left", va="top", fontsize=9,
                bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8})
    fig.suptitle("Iris Regression — Actual vs Predicted (All Models)", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_fig / "iris_comparison_actual_vs_predicted.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # --- (c) Boxplot of absolute errors ---
    abs_errors = {name: np.abs(y_test_np - predictions[name]) for name in MODEL_NAMES}
    fig, ax = plt.subplots(figsize=(9, 6))
    bp = ax.boxplot(
        [abs_errors[n] for n in MODEL_NAMES],
        labels=MODEL_NAMES,
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 2},
    )
    for patch, name in zip(bp["boxes"], MODEL_NAMES):
        patch.set_facecolor(PALETTE[name])
        patch.set_alpha(0.7)
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Absolute Error (cm)", fontsize=12)
    ax.set_title("Iris Regression — Absolute Error Distribution (Boxplot)", fontsize=13)
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(out_fig / "iris_comparison_error_distribution_boxplot.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # --- (d) CV R² bar chart with error bars ---
    fig, ax = plt.subplots(figsize=(9, 5))
    bar_colors = [PALETTE[m] for m in cv_df["Model"]]
    bars = ax.bar(cv_df["Model"], cv_df["CV_R2_mean"], yerr=cv_df["CV_R2_std"],
                  color=bar_colors, edgecolor="black", linewidth=0.5,
                  capsize=5, error_kw={"linewidth": 1.5})
    ax.set_title("Iris Regression — 10-Fold CV R² (Mean ± Std)", fontsize=13)
    ax.set_ylabel("Mean CV R²", fontsize=12)
    ax.tick_params(axis="x", rotation=20)
    for bar, (_, row) in zip(bars, cv_df.iterrows()):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + row["CV_R2_std"] + 0.005,
                f"{row['CV_R2_mean']:.4f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_fig / "iris_comparison_cv_r2_bar.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------
    # 8. Final summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("IRIS REGRESSION COMPARISON — HOLD-OUT TEST RESULTS")
    print("=" * 70)
    print(metrics_df[["Model", "R2", "RMSE", "MAE"]].to_string(index=False))

    print("\n" + "=" * 70)
    print("10-FOLD CROSS-VALIDATION RESULTS")
    print("=" * 70)
    print(cv_df[["Model", "CV_R2_mean", "CV_R2_std"]].to_string(index=False))

    best = metrics_df.iloc[0]
    print(f"\n✓ Best model (test R²): {best['Model']} (R²={best['R2']:.4f}, "
          f"RMSE={best['RMSE']:.4f} cm, MAE={best['MAE']:.4f} cm)")
    best_cv = cv_df.iloc[0]
    print(f"✓ Best model (CV R²): {best_cv['Model']} (CV R²={best_cv['CV_R2_mean']:.4f} "
          f"± {best_cv['CV_R2_std']:.4f})")
    print(f"\nAll figures saved → {out_fig}")
    print(f"All metrics saved → {out_csv}")


if __name__ == "__main__":
    main()
