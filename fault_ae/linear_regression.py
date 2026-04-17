"""
Optimised Linear Regression on Fault_AE_Combined_Load_Final.xlsx.

Target  : Load  (column C)
Features: RISE, COUN, ENER, DURATION, AMP, A-FRQ, RMS, ASL, PCNTS,
          R-FRQ, I-FRQ, ABS-ENERGY, FRQ-C, P-FRQ  (columns D-Q)

Optimisations over plain OLS:
  * Log1p-transform + derived AE features (fault_ae/utils.py).
  * Polynomial degree-2 interaction terms capture non-linear feature pairs.
  * SelectPercentile (mutual-information) keeps the most informative 25 %.
  * RidgeCV automatically tunes the L2 regularisation strength.

Outputs:
  outputs/figures/fault_ae/lr_actual_vs_predicted_with_residuals.png
  outputs/figures/fault_ae/lr_individual_test_predictions.png
  outputs/figures/fault_ae/lr_cv_scores.png
  outputs/figures/fault_ae/lr_feature_importance.png
  outputs/figures/fault_ae/lr_shap_summary.png
  outputs/csv/fault_ae/lr_metrics.csv
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.feature_selection import SelectPercentile, mutual_info_regression
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import load_data

warnings.filterwarnings("ignore")

MODEL_COLOR = "#4878CF"
MODEL_LABEL = "Ridge Regression (poly-2, feature-selected)"

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "Fault_AE_Combined_Load_Final.xlsx"
OUT_FIG = ROOT / "outputs" / "figures" / "fault_ae"
OUT_CSV = ROOT / "outputs" / "csv" / "fault_ae"

sns.set_style("whitegrid")


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "R2":   float(r2_score(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE":  float(mean_absolute_error(y_true, y_pred)),
    }


def build_pipeline() -> Pipeline:
    """
    StandardScaler → PolynomialFeatures(degree=2, interactions only)
    → SelectPercentile(mutual_info, 25 %) → RidgeCV
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("poly",   PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
        ("select", SelectPercentile(mutual_info_regression, percentile=25)),
        ("ridge",  RidgeCV(alphas=[0.01, 0.1, 1, 10, 100, 500, 1000])),
    ])


def main() -> None:
    for d in (OUT_FIG, OUT_CSV):
        d.mkdir(parents=True, exist_ok=True)

    # ── 1. Load & engineer features ─────────────────────────────────────
    X, y = load_data(DATA_PATH)
    feat_names = list(X.columns)
    X_arr, y_arr = X.values, y.values

    # ── 2. 80/20 split ──────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X_arr, y_arr, test_size=0.2, random_state=42
    )

    # ── 3. Train ─────────────────────────────────────────────────────────
    pipe = build_pipeline()
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    # ── 4. Evaluate ─────────────────────────────────────────────────────
    metrics = evaluate(y_test, y_pred)
    print(f"{MODEL_LABEL}: R²={metrics['R2']:.4f}, RMSE={metrics['RMSE']:.4f}, MAE={metrics['MAE']:.4f}")

    # ── 5. 10-fold CV ───────────────────────────────────────────────────
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    cv_scores = cross_val_score(build_pipeline(), X_arr, y_arr, cv=kf, scoring="r2")
    print(f"CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    residuals = y_test - y_pred
    vmin = min(y_test.min(), y_pred.min())
    vmax = max(y_test.max(), y_pred.max())

    # ── 6a. Predicted vs Actual + Residuals (combined) ──────────────────
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(8, 10),
        gridspec_kw={"height_ratios": [2, 1]},
    )
    ax_top.scatter(y_test, y_pred, alpha=0.4, s=15, color=MODEL_COLOR,
                   edgecolors="white", linewidths=0.3)
    ax_top.plot([vmin, vmax], [vmin, vmax], "k--", linewidth=1.5, label="Perfect fit")
    ax_top.set_xlabel("Actual Load (kN)", fontsize=11)
    ax_top.set_ylabel("Predicted Load (kN)", fontsize=11)
    ax_top.set_title(f"Linear Regression — Predicted vs Actual Load", fontsize=13)
    ax_top.text(0.04, 0.96,
                f"R²={metrics['R2']:.4f}\nRMSE={metrics['RMSE']:.4f}\nMAE={metrics['MAE']:.4f}",
                transform=ax_top.transAxes, ha="left", va="top", fontsize=10,
                bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8})
    ax_top.legend(fontsize=10)

    ax_bot.scatter(y_pred, residuals, alpha=0.4, s=15, color=MODEL_COLOR,
                   edgecolors="white", linewidths=0.3)
    ax_bot.axhline(0, color="black", linestyle="--", linewidth=1.5)
    ax_bot.set_xlabel("Predicted Load (kN)", fontsize=11)
    ax_bot.set_ylabel("Residual (Actual − Predicted)", fontsize=11)
    ax_bot.set_title("Residual Error", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT_FIG / "lr_actual_vs_predicted_with_residuals.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ── 6b. Individual test predictions ─────────────────────────────────
    n_show = min(100, len(y_test))
    idx = np.arange(n_show)
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(idx, y_test[:n_show], "o-", color="steelblue", markersize=4,
            linewidth=1, label="Actual Load", alpha=0.8)
    ax.plot(idx, y_pred[:n_show], "s--", color="firebrick", markersize=4,
            linewidth=1, label="Predicted Load", alpha=0.8)
    ax.set_xlabel("Test Sample Index", fontsize=11)
    ax.set_ylabel("Load (kN)", fontsize=11)
    ax.set_title("Linear Regression — Actual vs Predicted Load (Individual Test Samples)", fontsize=13)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT_FIG / "lr_individual_test_predictions.png", dpi=300, bbox_inches="tight")
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

    # ── 6d. Top-20 most-important features (|coefficient|) ───────────────
    # The final Ridge coefficients live after feature selection + poly expansion
    ridge = pipe.named_steps["ridge"]
    selector = pipe.named_steps["select"]
    poly = pipe.named_steps["poly"]

    # Reconstruct poly feature names
    scaler = pipe.named_steps["scaler"]
    poly_names = poly.get_feature_names_out(feat_names)
    # After selection
    selected_mask = selector.get_support()
    selected_names = [n for n, m in zip(poly_names, selected_mask) if m]

    coef_abs = np.abs(ridge.coef_)
    n_top = min(20, len(coef_abs))
    top_idx = np.argsort(coef_abs)[-n_top:][::-1]
    top_names = [selected_names[i] for i in top_idx]
    top_coef = coef_abs[top_idx]

    fig, ax = plt.subplots(figsize=(10, 7))
    y_pos = np.arange(n_top)
    ax.barh(y_pos, top_coef[::-1], color=MODEL_COLOR, edgecolor="black", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_names[::-1], fontsize=9)
    ax.set_xlabel("|Ridge Coefficient| (standardised)", fontsize=11)
    ax.set_title("Linear Regression — Top-20 Feature Importance", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT_FIG / "lr_feature_importance.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ── 6e. SHAP summary plot ─────────────────────────────────────────────
    print("Computing SHAP values (LinearExplainer on engineered features) …")
    # Explain at the engineered-feature level using the full pipeline as a
    # single black-box with KernelExplainer on a small background sample.
    bg_size = min(200, len(X_train))
    rng = np.random.RandomState(0)
    bg_idx = rng.choice(len(X_train), size=bg_size, replace=False)
    background = X_train[bg_idx]

    explainer = shap.KernelExplainer(pipe.predict, background)
    shap_sample = X_test[:200]
    shap_values = explainer.shap_values(shap_sample, nsamples=100)

    shap_df = pd.DataFrame(shap_sample, columns=feat_names)
    fig, _ = plt.subplots(figsize=(10, 7))
    shap.summary_plot(shap_values, shap_df, feature_names=feat_names,
                      plot_type="dot", show=False, max_display=20)
    plt.title("Linear Regression — SHAP Summary Plot", fontsize=13, pad=12)
    plt.tight_layout()
    plt.savefig(OUT_FIG / "lr_shap_summary.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved: lr_shap_summary.png")

    # ── 7. Save metrics ─────────────────────────────────────────────────
    metrics_df = pd.DataFrame([{
        "Model": MODEL_LABEL,
        **metrics,
        "CV_R2_mean": cv_scores.mean(),
        "CV_R2_std":  cv_scores.std(),
    }])
    metrics_df.to_csv(OUT_CSV / "lr_metrics.csv", index=False)

    # ── 8. Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("LINEAR REGRESSION — FAULT AE SUMMARY")
    print("=" * 60)
    print(f"  Model: {MODEL_LABEL}")
    print(f"  R²   : {metrics['R2']:.4f}")
    print(f"  RMSE : {metrics['RMSE']:.4f} kN")
    print(f"  MAE  : {metrics['MAE']:.4f} kN")
    print(f"  CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"  Best Ridge α: {ridge.alpha_:.4f}")
    print(f"\nSaved figures → {OUT_FIG}")
    print(f"Saved metrics → {OUT_CSV / 'lr_metrics.csv'}")


if __name__ == "__main__":
    main()
