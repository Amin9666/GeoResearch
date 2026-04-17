"""
Symbolic Regression via gplearn on Fault_AE_Combined_Load_Final.xlsx.

Target  : Load  (column C)
Features: RISE, COUN, ENER, DURATION, AMP, A-FRQ, RMS, ASL, PCNTS,
          R-FRQ, I-FRQ, ABS-ENERGY, FRQ-C, P-FRQ  (columns D-Q)

Uses gplearn's SymbolicRegressor (evolutionary genetic programming) to
discover a compact mathematical expression for Load.

Outputs:
  outputs/figures/fault_ae/sr_actual_vs_predicted_with_residuals.png
  outputs/figures/fault_ae/sr_individual_test_predictions.png
  outputs/figures/fault_ae/sr_cv_scores.png
  outputs/csv/fault_ae/sr_metrics.csv
  outputs/csv/fault_ae/sr_formula.txt
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
from gplearn.genetic import SymbolicRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

MODEL_COLOR = "#C4AD66"

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

    # ── 1. Load & scale data ────────────────────────────────────────────
    df = pd.read_excel(DATA_PATH)
    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale features (z-score normalisation) for better convergence
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # ── 2. Symbolic Regression (gplearn) ─────────────────────────────────
    print("Running Symbolic Regression (gplearn) — this may take a few minutes …")
    sr = SymbolicRegressor(
        population_size=3000,
        generations=30,
        tournament_size=20,
        stopping_criteria=0.005,
        const_range=(-10.0, 10.0),
        init_depth=(2, 6),
        init_method="half and half",
        function_set=("add", "sub", "mul", "div", "sqrt", "abs", "neg", "log"),
        metric="mse",
        parsimony_coefficient=0.0005,
        p_crossover=0.7,
        p_subtree_mutation=0.1,
        p_hoist_mutation=0.05,
        p_point_mutation=0.1,
        max_samples=1.0,
        feature_names=FEATURE_COLS,
        warm_start=False,
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )
    sr.fit(X_train_s, y_train)
    y_pred = sr.predict(X_test_s)

    # ── 3. Evaluate ─────────────────────────────────────────────────────
    metrics = evaluate(y_test, y_pred)
    print(f"Symbolic Regression: R²={metrics['R2']:.4f}, RMSE={metrics['RMSE']:.4f}, MAE={metrics['MAE']:.4f}")

    formula = str(sr._program)
    print(f"Best formula: {formula}")

    # ── 4. 10-fold CV ───────────────────────────────────────────────────
    # Use a Pipeline so the scaler is re-fitted inside each fold (no leakage)
    from sklearn.pipeline import Pipeline

    sr_cv = SymbolicRegressor(
        population_size=1000,
        generations=15,
        tournament_size=20,
        function_set=("add", "sub", "mul", "div", "abs"),
        parsimony_coefficient=0.0005,
        feature_names=FEATURE_COLS,
        random_state=42,
        n_jobs=-1,
        verbose=0,
    )
    pipe_cv = Pipeline([("scaler", StandardScaler()), ("sr", sr_cv)])
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipe_cv, X, y, cv=kf, scoring="r2")
    print(f"CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    residuals = y_test - y_pred
    vmin = min(y_test.min(), y_pred.min())
    vmax = max(y_test.max(), y_pred.max())

    # ── 5a. Predicted vs Actual + Residuals (combined) ──────────────────
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(8, 10),
        gridspec_kw={"height_ratios": [2, 1]},
    )

    ax_top.scatter(
        y_test, y_pred, alpha=0.5, s=20, color=MODEL_COLOR,
        edgecolors="white", linewidths=0.3,
    )
    ax_top.plot([vmin, vmax], [vmin, vmax], "k--", linewidth=1.5, label="Perfect fit")
    ax_top.set_xlabel("Actual Load (kN)", fontsize=11)
    ax_top.set_ylabel("Predicted Load (kN)", fontsize=11)
    ax_top.set_title("Symbolic Regression — Predicted vs Actual Load", fontsize=13)
    ax_top.text(
        0.04, 0.96,
        f"R²={metrics['R2']:.4f}\nRMSE={metrics['RMSE']:.4f}\nMAE={metrics['MAE']:.4f}",
        transform=ax_top.transAxes, ha="left", va="top", fontsize=10,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
    )
    ax_top.legend(fontsize=10)

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
        OUT_FIG / "sr_actual_vs_predicted_with_residuals.png",
        dpi=300, bbox_inches="tight",
    )
    plt.close(fig)

    # ── 5b. Real vs Predicted for individual test samples ───────────────
    n_show = min(100, len(y_test))
    idx = np.arange(n_show)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(idx, y_test[:n_show], "o-", color="steelblue", markersize=4,
            linewidth=1, label="Actual Load", alpha=0.8)
    ax.plot(idx, y_pred[:n_show], "s--", color="firebrick", markersize=4,
            linewidth=1, label="Predicted Load", alpha=0.8)
    ax.set_xlabel("Test Sample Index", fontsize=11)
    ax.set_ylabel("Load (kN)", fontsize=11)
    ax.set_title("Symbolic Regression — Actual vs Predicted Load (Individual Test Samples)", fontsize=13)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(
        OUT_FIG / "sr_individual_test_predictions.png",
        dpi=300, bbox_inches="tight",
    )
    plt.close(fig)

    # ── 5c. CV scores ───────────────────────────────────────────────────
    # Clip display to [-1, 1] for readability; raw values saved in CSV
    cv_display = np.clip(cv_scores, -1.0, 1.0)
    fig, ax = plt.subplots(figsize=(10, 5))
    fold_labels = [f"Fold {i+1}" for i in range(len(cv_scores))]
    bar_colors = [MODEL_COLOR if v >= 0 else "#888888" for v in cv_display]
    bars = ax.bar(fold_labels, cv_display, color=bar_colors, edgecolor="black", linewidth=0.5)
    ax.axhline(cv_scores.mean(), color="red", linestyle="--", linewidth=1.5,
               label=f"Mean R²={cv_scores.mean():.4f}")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("CV Fold", fontsize=11)
    ax.set_ylabel("R² Score (clipped to [−1, 1])", fontsize=11)
    ax.set_title("Symbolic Regression — 10-Fold CV R² Scores", fontsize=13)
    ax.legend(fontsize=10)
    ax.tick_params(axis="x", rotation=45)
    for bar, val in zip(bars, cv_scores):
        offset = 0.01 if val >= 0 else -0.03
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + offset,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_FIG / "sr_cv_scores.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ── 6. Save outputs ─────────────────────────────────────────────────
    with open(OUT_CSV / "sr_formula.txt", "w") as f:
        f.write(f"Best Symbolic Regression Formula:\n{formula}\n")

    metrics_df = pd.DataFrame([{
        "Model": "Symbolic Regression",
        **metrics,
        "CV_R2_mean": cv_scores.mean(),
        "CV_R2_std": cv_scores.std(),
        "Formula": formula,
    }])
    metrics_df.to_csv(OUT_CSV / "sr_metrics.csv", index=False)

    # ── 7. Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SYMBOLIC REGRESSION — FAULT AE SUMMARY")
    print("=" * 60)
    print(f"  R²   : {metrics['R2']:.4f}")
    print(f"  RMSE : {metrics['RMSE']:.4f} kN")
    print(f"  MAE  : {metrics['MAE']:.4f} kN")
    print(f"  CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"\nBest formula:\n  {formula}")
    print(f"\nSaved figures → {OUT_FIG}")
    print(f"Saved metrics → {OUT_CSV / 'sr_metrics.csv'}")


if __name__ == "__main__":
    main()
