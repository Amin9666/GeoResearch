"""
Symbolic Regression via PySR on Fault_AE_Combined_Load_Final.xlsx.

Target  : Load  (column C)
Features: RISE, COUN, ENER, DURATION, AMP, A-FRQ, RMS, ASL, PCNTS,
          R-FRQ, I-FRQ, ABS-ENERGY, FRQ-C, P-FRQ  (columns D-Q)

Uses PySR (Julia-backed evolutionary symbolic regression) to discover an
explicit, interpretable mathematical formula for Load.

Optimisations:
  * Log1p + derived AE features (fault_ae/utils.py) reduce skew so that the
    search space contains simpler, more generalisable formulas.
  * StandardScaler normalises feature magnitudes for the search.
  * Large search budget: 50 iterations, 15 islands, 200 individuals each,
    600-second time cap, operators: +, -, *, /, sqrt, log, abs, square, cube.
  * model_selection="accuracy" picks the most accurate (lowest-loss) formula.
  * The Pareto front (complexity vs loss) is plotted in place of a CV bar
    chart — this is the standard SR model-selection diagnostic.
  * Fast 5-fold CV (tiny SR per fold) gives a cross-validated R² estimate.
  * SHAP values computed via KernelExplainer for the selected formula.

Outputs:
  outputs/figures/fault_ae/sr_actual_vs_predicted_with_residuals.png
  outputs/figures/fault_ae/sr_individual_test_predictions.png
  outputs/figures/fault_ae/sr_pareto_front.png          (replaces cv_scores)
  outputs/figures/fault_ae/sr_shap_summary.png
  outputs/csv/fault_ae/sr_formula.txt
  outputs/csv/fault_ae/sr_metrics.csv
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

# Must be set before importing pysr / juliacall
os.environ.setdefault("PYTHON_JULIAPKG_EXE", "/usr/bin/julia")

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from pysr import PySRRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import load_data

warnings.filterwarnings("ignore")

MODEL_COLOR = "#C4AD66"
MODEL_LABEL = "Symbolic Regression (PySR)"

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "Fault_AE_Combined_Load_Final.xlsx"
OUT_FIG = ROOT / "outputs" / "figures" / "fault_ae"
OUT_CSV = ROOT / "outputs" / "csv" / "fault_ae"

# PySR writes temp files here so the repo root stays clean
PYSR_TEMPDIR = Path("/tmp/pysr_fault_ae")
PYSR_TEMPDIR.mkdir(parents=True, exist_ok=True)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "R2":   float(r2_score(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE":  float(mean_absolute_error(y_true, y_pred)),
    }


def main() -> None:
    sns.set_style("whitegrid")
    for d in (OUT_FIG, OUT_CSV):
        d.mkdir(parents=True, exist_ok=True)

    # ── 1. Load & engineer features ─────────────────────────────────────
    X, y = load_data(DATA_PATH)
    feat_names = list(X.columns)
    X_arr, y_arr = X.values, y.values

    # ── 2. 80/20 split + scale ──────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X_arr, y_arr, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # ── 3. PySR Symbolic Regression ─────────────────────────────────────
    print("Running PySR Symbolic Regression — this may take several minutes …")
    model = PySRRegressor(
        # Search budget
        niterations=50,
        populations=15,
        population_size=200,
        ncycles_per_iteration=550,
        timeout_in_seconds=600,

        # Operators
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["sqrt", "log", "abs", "neg", "square", "cube"],

        # Complexity & parsimony
        maxsize=30,
        parsimony=0.0003,

        # Objective — pick most accurate equation from Pareto front
        elementwise_loss="loss(prediction, target) = (prediction - target)^2",
        model_selection="accuracy",

        # Write temp files outside the repository
        tempdir=str(PYSR_TEMPDIR),
        temp_equation_file=True,

        # Reproducibility (serial required for determinism)
        deterministic=True,
        parallelism="serial",
        random_state=42,

        verbosity=1,
    )
    model.fit(X_train_s, y_train, variable_names=feat_names)
    y_pred = model.predict(X_test_s)

    # ── 4. Evaluate ─────────────────────────────────────────────────────
    metrics = evaluate(y_test, y_pred)
    print(f"{MODEL_LABEL}: R²={metrics['R2']:.4f}, RMSE={metrics['RMSE']:.4f}, MAE={metrics['MAE']:.4f}")

    try:
        formula_str = str(model.sympy())
        latex_str   = str(model.latex())
    except Exception:
        formula_str = str(model.get_best()["equation"])
        latex_str   = formula_str
    print(f"Best formula: {formula_str}")

    # ── 5. Fast 5-fold CV ───────────────────────────────────────────────
    print("Running 5-fold cross-validation …")
    sr_cv = PySRRegressor(
        niterations=15,
        populations=8,
        population_size=100,
        ncycles_per_iteration=200,
        timeout_in_seconds=45,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["sqrt", "log", "abs", "neg"],
        maxsize=20,
        parsimony=0.001,
        model_selection="accuracy",
        tempdir=str(PYSR_TEMPDIR),
        temp_equation_file=True,
        deterministic=True,
        parallelism="serial",
        random_state=42,
        verbosity=0,
    )
    pipe_cv = Pipeline([("scaler", StandardScaler()), ("sr", sr_cv)])
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipe_cv, X_arr, y_arr, cv=kf, scoring="r2")
    print(f"5-Fold CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    residuals = y_test - y_pred
    vmin = min(y_test.min(), y_pred.min())
    vmax = max(y_test.max(), y_pred.max())

    # ── 6a. Predicted vs Actual + Residuals (combined) ──────────────────
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(8, 10), gridspec_kw={"height_ratios": [2, 1]}
    )
    ax_top.scatter(y_test, y_pred, alpha=0.4, s=15, color=MODEL_COLOR,
                   edgecolors="white", linewidths=0.3)
    ax_top.plot([vmin, vmax], [vmin, vmax], "k--", linewidth=1.5, label="Perfect fit")
    ax_top.set_xlabel("Actual Load (kN)", fontsize=11)
    ax_top.set_ylabel("Predicted Load (kN)", fontsize=11)
    ax_top.set_title("Symbolic Regression — Predicted vs Actual Load", fontsize=13)
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
    fig.savefig(OUT_FIG / "sr_actual_vs_predicted_with_residuals.png",
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
    ax.set_title("Symbolic Regression — Actual vs Predicted Load (Individual Test Samples)", fontsize=13)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT_FIG / "sr_individual_test_predictions.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ── 6c. Pareto Front (Complexity vs Loss) ───────────────────────────
    # This is the canonical SR diagnostic — shows the trade-off between
    # equation complexity and predictive accuracy.
    hof = model.equations_
    if hof is not None and len(hof) > 0:
        complexities = hof["complexity"].values
        losses = hof["loss"].values
        # Convert MSE loss to approximate R²
        null_loss = float(np.var(y_train))  # baseline: predict mean
        r2_approx = 1 - losses / null_loss

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(complexities, r2_approx, "o-", color=MODEL_COLOR, markersize=6,
                linewidth=1.5, markeredgecolor="black", markeredgewidth=0.5)
        ax.axhline(metrics["R2"], color="red", linestyle="--", linewidth=1.5,
                   label=f"Selected (test R²={metrics['R2']:.4f})")
        for c, r in zip(complexities, r2_approx):
            if r > 0.01:
                ax.text(c, r + 0.005, f"{r:.3f}", fontsize=7, ha="center")
        ax.set_xlabel("Equation Complexity", fontsize=11)
        ax.set_ylabel("Approximate R² (training)", fontsize=11)
        ax.set_title("Symbolic Regression — Pareto Front (Complexity vs Accuracy)", fontsize=13)
        ax.legend(fontsize=10)
        fig.tight_layout()
        fig.savefig(OUT_FIG / "sr_pareto_front.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        print("  Saved: sr_pareto_front.png")

    # ── 6d. SHAP summary (KernelExplainer) ──────────────────────────────
    print("Computing SHAP values (KernelExplainer) …")

    def scaled_predict(X_in: np.ndarray) -> np.ndarray:
        return model.predict(scaler.transform(X_in))

    bg_size = min(100, len(X_train))
    rng = np.random.RandomState(0)
    bg_idx = rng.choice(len(X_train), size=bg_size, replace=False)
    background = X_train[bg_idx]

    explainer = shap.KernelExplainer(scaled_predict, background)
    shap_sample = X_test[:100]
    shap_values = explainer.shap_values(shap_sample, nsamples=100)

    shap_df = pd.DataFrame(shap_sample, columns=feat_names)
    fig, _ = plt.subplots(figsize=(10, 7))
    shap.summary_plot(shap_values, shap_df, feature_names=feat_names,
                      plot_type="dot", show=False, max_display=20)
    plt.title("Symbolic Regression — SHAP Summary Plot", fontsize=13, pad=12)
    plt.tight_layout()
    plt.savefig(OUT_FIG / "sr_shap_summary.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved: sr_shap_summary.png")

    # ── 7. Save outputs ─────────────────────────────────────────────────
    (OUT_CSV / "sr_formula.txt").write_text(
        f"Best Symbolic Regression Formula (PySR):\n{formula_str}\n\nLaTeX:\n{latex_str}\n"
    )
    metrics_df = pd.DataFrame([{
        "Model": MODEL_LABEL,
        **metrics,
        "CV_R2_mean": cv_scores.mean(),
        "CV_R2_std":  cv_scores.std(),
        "Formula":    formula_str,
    }])
    metrics_df.to_csv(OUT_CSV / "sr_metrics.csv", index=False)

    # ── 8. Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SYMBOLIC REGRESSION — FAULT AE SUMMARY")
    print("=" * 60)
    print(f"  R²   : {metrics['R2']:.4f}")
    print(f"  RMSE : {metrics['RMSE']:.4f} kN")
    print(f"  MAE  : {metrics['MAE']:.4f} kN")
    print(f"  CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"\nBest formula:\n  {formula_str}")
    print(f"\nSaved figures → {OUT_FIG}")
    print(f"Saved metrics → {OUT_CSV / 'sr_metrics.csv'}")


if __name__ == "__main__":
    main()
