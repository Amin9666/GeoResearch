"""
Linear Regression on Fault_AE_Combined_Load_Final.xlsx.

Target  : Load  (column C)
Features: RISE, COUN, ENER, DURATION, AMP, A-FRQ, RMS, ASL, PCNTS,
          R-FRQ, I-FRQ, ABS-ENERGY, FRQ-C, P-FRQ  (columns D-Q)

Outputs:
  outputs/figures/fault_ae/lr_actual_vs_predicted_with_residuals.png
  outputs/figures/fault_ae/lr_individual_test_predictions.png
  outputs/figures/fault_ae/lr_cv_scores.png
  outputs/figures/fault_ae/lr_feature_importance.png
  outputs/figures/fault_ae/lr_shap_summary.png
  outputs/csv/fault_ae/lr_metrics.csv
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
import shap
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

MODEL_COLOR = "#4878CF"

# ── Paths ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "Fault_AE_Combined_Load_Final.xlsx"
OUT_FIG = ROOT / "outputs" / "figures" / "fault_ae"
OUT_CSV = ROOT / "outputs" / "csv" / "fault_ae"

FEATURE_COLS = [
    "RISE", "COUN", "ENER", "DURATION", "AMP", "A-FRQ",
    "RMS", "ASL", "PCNTS", "R-FRQ", "I-FRQ", "ABS-ENERGY", "FRQ-C", "P-FRQ",
]
TARGET_COL = "Load"

sns.set_style("whitegrid")


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
    }


def main() -> None:
    for d in (OUT_FIG, OUT_CSV):
        d.mkdir(parents=True, exist_ok=True)

    # ── 1. Load data ────────────────────────────────────────────────────
    df = pd.read_excel(DATA_PATH)
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    # ── 2. 80/20 split ──────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ── 3. Train (with StandardScaler) ──────────────────────────────────
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)

    # ── 4. Evaluate ─────────────────────────────────────────────────────
    metrics = evaluate(y_test.values, y_pred)
    print(f"Linear Regression: R²={metrics['R2']:.4f}, RMSE={metrics['RMSE']:.4f}, MAE={metrics['MAE']:.4f}")

    # ── 5. 10-fold CV ───────────────────────────────────────────────────
    pipe = Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())])
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipe, X, y, cv=kf, scoring="r2")
    print(f"CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    residuals = y_test.values - y_pred
    vmin = min(y_test.min(), y_pred.min())
    vmax = max(y_test.max(), y_pred.max())

    # ── 6a. Predicted vs Actual + Residuals (combined) ──────────────────
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(8, 10),
        gridspec_kw={"height_ratios": [2, 1]},
        sharex=False,
    )

    # Top: predicted vs actual
    ax_top.scatter(
        y_test, y_pred, alpha=0.5, s=20, color=MODEL_COLOR,
        edgecolors="white", linewidths=0.3,
    )
    ax_top.plot([vmin, vmax], [vmin, vmax], "k--", linewidth=1.5, label="Perfect fit")
    ax_top.set_xlabel("Actual Load (kN)", fontsize=11)
    ax_top.set_ylabel("Predicted Load (kN)", fontsize=11)
    ax_top.set_title("Linear Regression — Predicted vs Actual Load", fontsize=13)
    ax_top.text(
        0.04, 0.96,
        f"R²={metrics['R2']:.4f}\nRMSE={metrics['RMSE']:.4f}\nMAE={metrics['MAE']:.4f}",
        transform=ax_top.transAxes, ha="left", va="top", fontsize=10,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
    )
    ax_top.legend(fontsize=10)

    # Bottom: residual errors
    ax_bot.scatter(
        y_pred, residuals, alpha=0.5, s=20, color=MODEL_COLOR,
        edgecolors="white", linewidths=0.3,
    )
    ax_bot.axhline(0, color="black", linestyle="--", linewidth=1.5)
    ax_bot.set_xlabel("Predicted Load (kN)", fontsize=11)
    ax_bot.set_ylabel("Residual (Actual − Predicted)", fontsize=11)
    ax_bot.set_title("Residual Error", fontsize=12)

    fig.tight_layout()
    fig.savefig(
        OUT_FIG / "lr_actual_vs_predicted_with_residuals.png",
        dpi=300, bbox_inches="tight",
    )
    plt.close(fig)

    # ── 6b. Real vs Predicted for individual test samples ───────────────
    n_show = min(100, len(y_test))
    idx = np.arange(n_show)
    y_test_show = y_test.values[:n_show]
    y_pred_show = y_pred[:n_show]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(idx, y_test_show, "o-", color="steelblue", markersize=4,
            linewidth=1, label="Actual Load", alpha=0.8)
    ax.plot(idx, y_pred_show, "s--", color="firebrick", markersize=4,
            linewidth=1, label="Predicted Load", alpha=0.8)
    ax.set_xlabel("Test Sample Index", fontsize=11)
    ax.set_ylabel("Load (kN)", fontsize=11)
    ax.set_title("Linear Regression — Actual vs Predicted Load (Individual Test Samples)", fontsize=13)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(
        OUT_FIG / "lr_individual_test_predictions.png",
        dpi=300, bbox_inches="tight",
    )
    plt.close(fig)

    # ── 6c. CV scores bar chart ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    fold_labels = [f"Fold {i+1}" for i in range(len(cv_scores))]
    bars = ax.bar(fold_labels, cv_scores, color=MODEL_COLOR, edgecolor="black", linewidth=0.5)
    ax.axhline(cv_scores.mean(), color="red", linestyle="--", linewidth=1.5,
               label=f"Mean R²={cv_scores.mean():.4f}")
    ax.set_xlabel("CV Fold", fontsize=11)
    ax.set_ylabel("R² Score", fontsize=11)
    ax.set_title("Linear Regression — 10-Fold CV R² Scores", fontsize=13)
    ax.legend(fontsize=10)
    ax.tick_params(axis="x", rotation=45)
    for bar, val in zip(bars, cv_scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_FIG / "lr_cv_scores.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ── 6d. Feature importance (|coefficients|) ─────────────────────────
    coef_abs = np.abs(model.coef_)
    sorted_idx = np.argsort(coef_abs)
    sorted_features = [FEATURE_COLS[i] for i in sorted_idx]
    sorted_coef = coef_abs[sorted_idx]

    fig, ax = plt.subplots(figsize=(9, 6))
    y_pos = np.arange(len(sorted_features))
    ax.barh(y_pos, sorted_coef, color=MODEL_COLOR, edgecolor="black", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_features, fontsize=10)
    ax.set_xlabel("|Coefficient| (standardised)", fontsize=11)
    ax.set_title("Linear Regression — Feature Importance (|Coefficient|)", fontsize=13)
    for i, val in enumerate(sorted_coef):
        ax.text(val + 0.002, i, f"{val:.4f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_FIG / "lr_feature_importance.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ── 6e. SHAP summary plot ─────────────────────────────────────────────
    print("Computing SHAP values …")
    X_test_df = pd.DataFrame(X_test_s, columns=FEATURE_COLS)
    explainer = shap.LinearExplainer(model, X_train_s, feature_perturbation="interventional")
    shap_values = explainer.shap_values(X_test_df)

    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(
        shap_values, X_test_df,
        feature_names=FEATURE_COLS,
        plot_type="dot",
        show=False,
        max_display=14,
    )
    plt.title("Linear Regression — SHAP Summary Plot", fontsize=13, pad=12)
    plt.tight_layout()
    plt.savefig(OUT_FIG / "lr_shap_summary.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved: lr_shap_summary.png")

    # ── 7. Save metrics ─────────────────────────────────────────────────
    metrics_df = pd.DataFrame([{
        "Model": "Linear Regression",
        **metrics,
        "CV_R2_mean": cv_scores.mean(),
        "CV_R2_std": cv_scores.std(),
    }])
    metrics_df.to_csv(OUT_CSV / "lr_metrics.csv", index=False)

    # ── 8. Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("LINEAR REGRESSION — FAULT AE SUMMARY")
    print("=" * 60)
    print(f"  R²   : {metrics['R2']:.4f}")
    print(f"  RMSE : {metrics['RMSE']:.4f} kN")
    print(f"  MAE  : {metrics['MAE']:.4f} kN")
    print(f"  CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"\nCoefficients (standardised features):")
    for feat, coef in zip(FEATURE_COLS, model.coef_):
        print(f"  {feat:15s}: {coef:+.6f}")
    print(f"  {'intercept':15s}: {model.intercept_:+.6f}")
    print(f"\nSaved figures → {OUT_FIG}")
    print(f"Saved metrics → {OUT_CSV / 'lr_metrics.csv'}")


if __name__ == "__main__":
    main()
