"""
Symbolic Regression via LassoCV on Engineered Features — Iris Regression Task.

Predict petal_length (cm) from sepal_length, sepal_width, petal_width.
Builds 24 engineered features (15 unary + 3 products + 6 bidirectional ratios),
then uses LassoCV to discover a sparse formula.

Outputs:
  outputs/figures/iris_sr_actual_vs_predicted.png
  outputs/figures/iris_sr_residuals_scatter.png
  outputs/figures/iris_sr_residuals_hist.png
  outputs/figures/iris_sr_cv_scores.png
  outputs/figures/iris_sr_formula_terms.png
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
from sklearn.datasets import load_iris
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore")

MODEL_COLOR = "#C4AD66"
_OFFSET = 1.0


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
    }


def build_symbolic_features(X: pd.DataFrame) -> pd.DataFrame:
    """Build 24 engineered features from 3 raw features.

    Unary (5 per feature × 3 = 15): raw, log(x+1), sqrt(x), x², 1/(x+1)
    Pairwise products (3 pairs × 1 = 3): x1*x2
    Pairwise ratios (3 pairs × 2 directions = 6): x1/(x2+1)
    Total: 15 + 3 + 6 = 24 features
    """
    cols = list(X.columns)
    Xf: dict[str, np.ndarray] = {}

    # Unary transforms
    for col in cols:
        v = X[col].values.astype(float)
        Xf[col] = v
        Xf[f"log_{col}"] = np.log(v + _OFFSET)
        Xf[f"sqrt_{col}"] = np.sqrt(np.abs(v))
        Xf[f"{col}_sq"] = v ** 2
        Xf[f"{col}_inv"] = _OFFSET / (v + _OFFSET)

    # Pairwise products and ratios (both directions)
    for i, c1 in enumerate(cols):
        for c2 in cols[i + 1:]:
            v1 = X[c1].values.astype(float)
            v2 = X[c2].values.astype(float)
            Xf[f"{c1}x{c2}"] = v1 * v2
            Xf[f"{c1}_div_{c2}"] = v1 / (v2 + _OFFSET)
            Xf[f"{c2}_div_{c1}"] = v2 / (v1 + _OFFSET)

    return pd.DataFrame(Xf)


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

    # 2. Build engineered feature matrix
    Xf = build_symbolic_features(X)
    feature_names = list(Xf.columns)
    print(f"Engineered features: {len(feature_names)}")

    # 3. 80/20 split
    idx = np.arange(len(y))
    idx_train, idx_test = train_test_split(idx, test_size=0.2, random_state=42)

    X_train = Xf.values[idx_train]
    X_test = Xf.values[idx_test]
    y_train = y.values[idx_train]
    y_test = y.values[idx_test]
    # 4. Scale with RobustScaler
    scaler = RobustScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # 5. Train LassoCV
    alphas = np.logspace(-6, 1, 80)
    model = LassoCV(cv=10, max_iter=50000, random_state=42, alphas=alphas)
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)

    # 6. Evaluate
    metrics = evaluate(y_test, y_pred)
    print(f"Symbolic Regression: R²={metrics['R2']:.4f}, RMSE={metrics['RMSE']:.4f}, MAE={metrics['MAE']:.4f}")

    # 7. Extract non-zero coefficients → symbolic formula
    coef = pd.Series(model.coef_, index=feature_names)
    nz_coef = coef[coef != 0].sort_values(key=abs, ascending=False)
    intercept = float(model.intercept_)

    # 8. Plots
    residuals = y_test - y_pred
    vmin = min(y_test.min(), y_pred.min())
    vmax = max(y_test.max(), y_pred.max())

    # (a) Actual vs Predicted
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_test, y_pred, alpha=0.6, s=40, color=MODEL_COLOR, edgecolors="white", linewidths=0.5)
    ax.plot([vmin, vmax], [vmin, vmax], "k--", linewidth=1.5, label="Perfect fit")
    ax.set_xlabel("Actual petal_length (cm)", fontsize=12)
    ax.set_ylabel("Predicted petal_length (cm)", fontsize=12)
    ax.set_title("Symbolic Regression (LassoCV) — Actual vs Predicted", fontsize=13)
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

    # (d) LassoCV alpha path (R² vs log alpha)
    # Reconstruct approximate CV scores from LassoCV alpha path
    # model.mse_path_ has shape (n_alphas, n_folds) — MSE per alpha per fold
    mse_path = model.mse_path_  # shape: (n_alphas, n_folds)
    mean_mse = mse_path.mean(axis=1)
    # Approximate R² = 1 - MSE/Var(y_train)
    var_y = np.var(y_train)
    approx_r2 = 1 - mean_mse / var_y

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.semilogx(model.alphas_, approx_r2, color=MODEL_COLOR, linewidth=2, marker="o", markersize=3)
    ax.axvline(model.alpha_, color="red", linestyle="--", linewidth=1.5, label=f"Best α={model.alpha_:.2e}")
    ax.set_xlabel("Alpha (log scale)", fontsize=12)
    ax.set_ylabel("Approx. CV R²", fontsize=12)
    ax.set_title("Symbolic Regression — LassoCV Alpha Path", fontsize=13)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(out_fig / "iris_sr_cv_scores.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # (e) Non-zero coefficient bar chart
    if len(nz_coef) > 0:
        fig, ax = plt.subplots(figsize=(max(8, len(nz_coef) * 0.6), 5))
        colors_bar = [MODEL_COLOR if c > 0 else "#8B4513" for c in nz_coef.values]
        ax.bar(range(len(nz_coef)), nz_coef.values, color=colors_bar, edgecolor="black", linewidth=0.5)
        ax.set_xticks(range(len(nz_coef)))
        ax.set_xticklabels(nz_coef.index, rotation=45, ha="right", fontsize=9)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Feature Term", fontsize=12)
        ax.set_ylabel("Coefficient", fontsize=12)
        ax.set_title(f"Symbolic Regression — Non-Zero LassoCV Coefficients ({len(nz_coef)} terms)", fontsize=13)
        fig.tight_layout()
        fig.savefig(out_fig / "iris_sr_formula_terms.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    # ------------------------------------------------------------------
    # 9. Save CSVs
    # ------------------------------------------------------------------
    formula_df = pd.DataFrame({"Term": nz_coef.index, "Coefficient": nz_coef.values})
    formula_df.loc[len(formula_df)] = {"Term": "INTERCEPT", "Coefficient": intercept}
    formula_df.to_csv(out_csv / "iris_sr_formula.csv", index=False)

    metrics_df = pd.DataFrame([{"Model": "Symbolic Regression (LassoCV)", **metrics,
                                  "Best_Alpha": model.alpha_, "N_Terms": len(nz_coef)}])
    metrics_df.to_csv(out_csv / "iris_sr_metrics.csv", index=False)

    # 10. Print summary
    print("\n" + "=" * 60)
    print("SYMBOLIC REGRESSION (LassoCV) — IRIS REGRESSION SUMMARY")
    print("=" * 60)
    print(f"  R²         : {metrics['R2']:.4f}")
    print(f"  RMSE       : {metrics['RMSE']:.4f} cm")
    print(f"  MAE        : {metrics['MAE']:.4f} cm")
    print(f"  Best alpha : {model.alpha_:.6e}")
    print(f"  Non-zero terms: {len(nz_coef)}")
    print(f"\nDiscovered Formula:")
    print(f"  petal_length = {intercept:.6f}")
    for term, c in nz_coef.items():
        print(f"    {c:+.6f} × {term}")
    print(f"\nSaved figures → {out_fig}")
    print(f"Saved formula → {out_csv / 'iris_sr_formula.csv'}")
    print(f"Saved metrics → {out_csv / 'iris_sr_metrics.csv'}")


if __name__ == "__main__":
    main()
