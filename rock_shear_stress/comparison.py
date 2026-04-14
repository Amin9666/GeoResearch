"""
Comparison of Four Algorithms on Shear Stress Prediction.

Algorithms compared:
  1. Linear Regression
  2. MLP (Multi-Layer Perceptron)
  3. Random Forest
  4. Symbolic Regression (LassoCV on 45 engineered feature candidates)

Outputs:
  outputs/figures/comparison_bar_metrics.png          – R², RMSE, MAE side-by-side bars
  outputs/figures/comparison_cv_r2.png                – 10-fold CV R² with error bars
  outputs/figures/comparison_actual_vs_predicted.png  – Actual vs Predicted scatter (2×2)
  outputs/figures/comparison_residuals.png            – Residual distributions overlaid
  outputs/figures/comparison_radar.png                – Radar / spider chart of normalised metrics
  outputs/csv/comparison_4_algorithms_metrics.csv
  outputs/csv/comparison_4_algorithms_cv.csv
  outputs/csv/comparison_4_algorithms_predictions.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import RobustScaler

import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FEATURES = ["RISE", "COUN", "ENER", "DURATION", "AMP"]
TARGET = "SHEAR STRESS"
_OFFSET = 1.0

PALETTE = {
    "Linear Regression":   "#4878CF",
    "MLP":                 "#6ACC65",
    "Random Forest":       "#D65F5F",
    "Symbolic Regression": "#C4AD66",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def remove_outliers_iqr(df: pd.DataFrame, cols: list[str], mult: float = 1.0) -> pd.DataFrame:
    mask = np.ones(len(df), dtype=bool)
    for c in cols:
        q1, q3 = df[c].quantile(0.25), df[c].quantile(0.75)
        iqr = q3 - q1
        mask &= (df[c] >= q1 - mult * iqr) & (df[c] <= q3 + mult * iqr)
    return df[mask].reset_index(drop=True)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "R2":   float(r2_score(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE":  float(mean_absolute_error(y_true, y_pred)),
    }


def build_symbolic_features(X: pd.DataFrame) -> pd.DataFrame:
    cols = list(X.columns)
    Xf: dict[str, np.ndarray] = {}
    for col in cols:
        v = X[col].values.astype(float)
        Xf[col] = v
        Xf[f"log_{col}"] = np.log(v + _OFFSET)
        Xf[f"sqrt_{col}"] = np.sqrt(v)
        Xf[f"{col}_sq"] = v ** 2
        Xf[f"{col}_inv"] = _OFFSET / (v + _OFFSET)
    for i, c1 in enumerate(cols):
        for c2 in cols[i + 1:]:
            v1, v2 = X[c1].values.astype(float), X[c2].values.astype(float)
            Xf[f"{c1}x{c2}"] = v1 * v2
            Xf[f"{c1}_div_{c2}"] = v1 / (v2 + _OFFSET)
    return pd.DataFrame(Xf)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_bar_metrics(metrics_df: pd.DataFrame, out_path: Path) -> None:
    """Bar chart with R², RMSE, and MAE for each model side by side."""
    models = list(metrics_df["Model"])
    colors = [PALETTE[m] for m in models]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, metric in zip(axes, ["R2", "RMSE", "MAE"]):
        bars = ax.bar(models, metrics_df[metric], color=colors,
                      edgecolor="black", linewidth=0.6)
        ax.set_title(f"Model Comparison — {metric}", fontsize=13)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=30)
        for bar, val in zip(bars, metrics_df[metric]):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.01,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("Hold-out Test Performance: 4-Algorithm Comparison", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def fig_cv_r2(cv_df: pd.DataFrame, out_path: Path) -> None:
    """Cross-validation R² bar chart with error bars."""
    cv_colors = [PALETTE[m] for m in cv_df["Model"]]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(cv_df["Model"], cv_df["CV_R2_mean"],
                  yerr=cv_df["CV_R2_std"],
                  color=cv_colors, edgecolor="black", linewidth=0.6,
                  capsize=6, error_kw={"linewidth": 1.5})
    ax.set_title("10-Fold Cross-Validation R² — 4-Algorithm Comparison", fontsize=13)
    ax.set_ylabel("Mean CV R²")
    ax.tick_params(axis="x", rotation=30)
    for bar, (_, row) in zip(bars, cv_df.iterrows()):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + row["CV_R2_std"] + 0.001,
                f"{row['CV_R2_mean']:.4f}±{row['CV_R2_std']:.4f}",
                ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def fig_actual_vs_predicted(y_test: np.ndarray,
                            predictions: dict[str, np.ndarray],
                            out_path: Path) -> None:
    """2×2 scatter plots of actual vs predicted for each model."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes_flat = axes.flatten()
    all_vals = np.concatenate([y_test] + list(predictions.values()))
    vmin, vmax = all_vals.min(), all_vals.max()

    for ax, (name, y_pred) in zip(axes_flat, predictions.items()):
        color = PALETTE.get(name, "#999999")
        m = metrics(y_test, y_pred)
        ax.scatter(y_test, y_pred, alpha=0.35, s=12, color=color)
        ax.plot([vmin, vmax], [vmin, vmax], "k--", linewidth=1.5, label="Ideal")
        ax.set_title(name, fontsize=12, fontweight="bold")
        ax.set_xlabel("Actual Shear Stress (MPa)")
        ax.set_ylabel("Predicted Shear Stress (MPa)")
        ax.text(0.04, 0.96,
                f"R²  = {m['R2']:.4f}\nRMSE= {m['RMSE']:.4f} MPa\nMAE = {m['MAE']:.4f} MPa",
                transform=ax.transAxes, ha="left", va="top", fontsize=9,
                bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85})
        ax.legend(fontsize=8, loc="lower right")

    fig.suptitle("Actual vs Predicted Shear Stress — 4-Algorithm Comparison",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def fig_residuals(y_test: np.ndarray,
                  predictions: dict[str, np.ndarray],
                  out_path: Path) -> None:
    """Overlaid residual distribution histograms."""
    fig, ax = plt.subplots(figsize=(12, 6))
    for name, y_pred in predictions.items():
        residuals = y_test - y_pred
        color = PALETTE.get(name, "#999999")
        ax.hist(residuals, bins=50, alpha=0.40, label=name, density=True, color=color)
    ax.axvline(0, color="black", linestyle="--", linewidth=1.5, label="Zero residual")
    ax.set_title("Residual Distributions — 4-Algorithm Comparison", fontsize=13)
    ax.set_xlabel("Residual  (Actual − Predicted, MPa)")
    ax.set_ylabel("Density")
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def fig_radar(metrics_df: pd.DataFrame, out_path: Path) -> None:
    """Radar/spider chart comparing normalised metrics across models."""
    # For radar: R² higher is better; RMSE/MAE lower is better → invert them
    df = metrics_df.copy().reset_index(drop=True)

    # Normalise each metric 0→1 (1 = best)
    r2_norm   = (df["R2"]   - df["R2"].min())   / (df["R2"].max()   - df["R2"].min() + 1e-9)
    rmse_norm = 1 - (df["RMSE"] - df["RMSE"].min()) / (df["RMSE"].max() - df["RMSE"].min() + 1e-9)
    mae_norm  = 1 - (df["MAE"]  - df["MAE"].min())  / (df["MAE"].max()  - df["MAE"].min() + 1e-9)

    categories = ["R²", "1−RMSE\n(normalised)", "1−MAE\n(normalised)"]
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"projection": "polar"})
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=8)
    ax.yaxis.set_tick_params(labelsize=7)

    for _, row in df.iterrows():
        name = row["Model"]
        idx = df.index[df["Model"] == name][0]
        values = [r2_norm[idx], rmse_norm[idx], mae_norm[idx]]
        values += values[:1]
        color = PALETTE.get(name, "#999999")
        ax.plot(angles, values, color=color, linewidth=2.0, label=name)
        ax.fill(angles, values, color=color, alpha=0.15)

    ax.set_title("Radar Chart — Normalised Performance\n(outer = better)",
                 fontsize=12, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def fig_grouped_bar(metrics_df: pd.DataFrame, out_path: Path) -> None:
    """Grouped bar chart showing all three metrics per model together."""
    models = list(metrics_df["Model"])
    x = np.arange(len(models))
    width = 0.25

    r2_vals   = metrics_df["R2"].values
    rmse_vals = metrics_df["RMSE"].values
    mae_vals  = metrics_df["MAE"].values

    fig, ax = plt.subplots(figsize=(12, 6))
    b1 = ax.bar(x - width, r2_vals,   width, label="R²",   color="#4878CF", edgecolor="black", linewidth=0.5)
    b2 = ax.bar(x,         rmse_vals, width, label="RMSE", color="#D65F5F", edgecolor="black", linewidth=0.5)
    b3 = ax.bar(x + width, mae_vals,  width, label="MAE",  color="#6ACC65", edgecolor="black", linewidth=0.5)

    for bars in (b1, b2, b3):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.01,
                    f"{bar.get_height():.3f}",
                    ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right", fontsize=10)
    ax.set_ylabel("Metric value")
    ax.set_title("Grouped Metric Comparison — 4 Algorithms", fontsize=13)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    root = Path(__file__).resolve().parents[1]

    # Locate data
    data_path = root / "data" / "raw" / "Shear_Data_15.csv"
    if not data_path.exists():
        data_path = root / "Shear_Data_15.csv"
    if not data_path.exists():
        sys.exit(f"ERROR: Cannot find Shear_Data_15.csv (looked in {data_path})")

    # Output directories
    out_fig = root / "outputs" / "figures"
    out_csv = root / "outputs" / "csv"
    cg_dir  = root / "comparison_graphs"
    for d in (out_fig, out_csv, cg_dir):
        d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load & clean data
    # ------------------------------------------------------------------
    df = pd.read_csv(data_path)
    df = remove_outliers_iqr(df, FEATURES + [TARGET], mult=1.0)
    print(f"Samples after IQR-1.0 cleaning: {len(df)}")

    X_raw = df[FEATURES]
    y = df[TARGET].values

    idx = np.arange(len(y))
    idx_train, idx_test = train_test_split(idx, test_size=0.2, random_state=42)

    X_raw_train = X_raw.iloc[idx_train].values
    X_raw_test  = X_raw.iloc[idx_test].values
    y_train = y[idx_train]
    y_test  = y[idx_test]

    scaler_raw = RobustScaler()
    X_train_s  = scaler_raw.fit_transform(X_raw_train)
    X_test_s   = scaler_raw.transform(X_raw_test)

    # Symbolic feature matrix
    Xf = build_symbolic_features(X_raw)
    X_sym_train = Xf.values[idx_train]
    X_sym_test  = Xf.values[idx_test]

    scaler_sym  = RobustScaler()
    X_sym_train_s = scaler_sym.fit_transform(X_sym_train)
    X_sym_test_s  = scaler_sym.transform(X_sym_test)

    # ------------------------------------------------------------------
    # 2. Define and train the four models
    # ------------------------------------------------------------------
    print("\nTraining models on hold-out split …")

    # --- Linear Regression ---
    lr = LinearRegression()
    lr.fit(X_train_s, y_train)
    y_pred_lr = lr.predict(X_test_s)

    # --- MLP ---
    mlp = MLPRegressor(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        solver="adam",
        max_iter=2000,
        learning_rate_init=0.001,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
        n_iter_no_change=30,
    )
    mlp.fit(X_train_s, y_train)
    y_pred_mlp = mlp.predict(X_test_s)

    # --- Random Forest ---
    rf = RandomForestRegressor(
        n_estimators=300, max_depth=14, min_samples_leaf=5,
        random_state=42, n_jobs=-1
    )
    rf.fit(X_raw_train, y_train)          # RF does not need scaling
    y_pred_rf = rf.predict(X_raw_test)

    # --- Symbolic Regression (LassoCV on 45 engineered features) ---
    sr = LassoCV(cv=10, max_iter=30000, random_state=42, n_jobs=-1,
                 alphas=np.logspace(-6, 0, 60))
    sr.fit(X_sym_train_s, y_train)
    y_pred_sr = sr.predict(X_sym_test_s)

    predictions: dict[str, np.ndarray] = {
        "Linear Regression":   y_pred_lr,
        "MLP":                 y_pred_mlp,
        "Random Forest":       y_pred_rf,
        "Symbolic Regression": y_pred_sr,
    }

    # ------------------------------------------------------------------
    # 3. Compute hold-out metrics
    # ------------------------------------------------------------------
    metrics_rows = []
    print("\n── Hold-out test results ──")
    for name, y_pred in predictions.items():
        m = metrics(y_test, y_pred)
        metrics_rows.append({"Model": name, **m})
        print(f"  {name:<25}  R²={m['R2']:+.4f}  RMSE={m['RMSE']:.4f}  MAE={m['MAE']:.4f}")

    metrics_df = pd.DataFrame(metrics_rows).sort_values("R2", ascending=False).reset_index(drop=True)
    metrics_df.to_csv(out_csv / "comparison_4_algorithms_metrics.csv", index=False)

    # ------------------------------------------------------------------
    # 4. 10-fold cross-validation
    # ------------------------------------------------------------------
    print("\nRunning 10-fold cross-validation …")
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    cv_rows = []

    def _cv(model_factory, X_arr, y_arr, use_scaler=True):
        scores = []
        for tr_i, ts_i in kf.split(X_arr):
            sc = RobustScaler() if use_scaler else None
            Xtr = sc.fit_transform(X_arr[tr_i]) if sc else X_arr[tr_i]
            Xts = sc.transform(X_arr[ts_i])     if sc else X_arr[ts_i]
            m = model_factory()
            m.fit(Xtr, y_arr[tr_i])
            scores.append(r2_score(y_arr[ts_i], m.predict(Xts)))
        return float(np.mean(scores)), float(np.std(scores))

    # Linear Regression
    mean_r2, std_r2 = _cv(LinearRegression, X_raw.values, y)
    cv_rows.append({"Model": "Linear Regression", "CV_R2_mean": mean_r2, "CV_R2_std": std_r2})
    print(f"  Linear Regression:   CV R²={mean_r2:.4f} ± {std_r2:.4f}")

    # MLP
    mean_r2, std_r2 = _cv(
        lambda: MLPRegressor(hidden_layer_sizes=(128, 64, 32), activation="relu",
                             solver="adam", max_iter=2000, learning_rate_init=0.001,
                             early_stopping=True, validation_fraction=0.1,
                             random_state=42, n_iter_no_change=30),
        X_raw.values, y
    )
    cv_rows.append({"Model": "MLP", "CV_R2_mean": mean_r2, "CV_R2_std": std_r2})
    print(f"  MLP:                 CV R²={mean_r2:.4f} ± {std_r2:.4f}")

    # Random Forest (no scaling needed)
    mean_r2, std_r2 = _cv(
        lambda: RandomForestRegressor(n_estimators=300, max_depth=14,
                                      min_samples_leaf=5, random_state=42, n_jobs=-1),
        X_raw.values, y, use_scaler=False
    )
    cv_rows.append({"Model": "Random Forest", "CV_R2_mean": mean_r2, "CV_R2_std": std_r2})
    print(f"  Random Forest:       CV R²={mean_r2:.4f} ± {std_r2:.4f}")

    # Symbolic Regression (LassoCV on engineered features)
    sr_cv_scores = []
    for tr_i, ts_i in kf.split(Xf.values):
        sc = RobustScaler()
        Xtr_cv = sc.fit_transform(Xf.values[tr_i])
        Xts_cv = sc.transform(Xf.values[ts_i])
        m = LassoCV(cv=10, max_iter=30000, random_state=42, n_jobs=-1,
                    alphas=np.logspace(-6, 0, 60))
        m.fit(Xtr_cv, y[tr_i])
        sr_cv_scores.append(r2_score(y[ts_i], m.predict(Xts_cv)))
    sr_cv_mean = float(np.mean(sr_cv_scores))
    sr_cv_std  = float(np.std(sr_cv_scores))
    cv_rows.append({"Model": "Symbolic Regression", "CV_R2_mean": sr_cv_mean, "CV_R2_std": sr_cv_std})
    print(f"  Symbolic Regression: CV R²={sr_cv_mean:.4f} ± {sr_cv_std:.4f}")

    cv_df = pd.DataFrame(cv_rows).sort_values("CV_R2_mean", ascending=False).reset_index(drop=True)
    cv_df.to_csv(out_csv / "comparison_4_algorithms_cv.csv", index=False)

    # ------------------------------------------------------------------
    # 5. Save predictions CSV
    # ------------------------------------------------------------------
    pred_df = pd.DataFrame({"Actual": y_test})
    for name, y_pred in predictions.items():
        pred_df[name.replace(" ", "_")] = y_pred
    pred_df.to_csv(out_csv / "comparison_4_algorithms_predictions.csv", index=False)

    # ------------------------------------------------------------------
    # 6. Generate comparison graphs
    # ------------------------------------------------------------------
    sns.set_style("whitegrid")
    print("\nGenerating comparison graphs …")

    # a) Bar chart: R², RMSE, MAE
    fig_bar_metrics(metrics_df,
                    out_fig / "comparison_bar_metrics.png")

    # b) CV R² bar with error bars
    fig_cv_r2(cv_df,
              out_fig / "comparison_cv_r2.png")

    # c) Actual vs Predicted (2×2 grid)
    ordered_preds = {n: predictions[n] for n in metrics_df["Model"]}
    fig_actual_vs_predicted(y_test, ordered_preds,
                            out_fig / "comparison_actual_vs_predicted.png")

    # d) Residual distributions
    fig_residuals(y_test, ordered_preds,
                  out_fig / "comparison_residuals.png")

    # e) Radar chart
    fig_radar(metrics_df,
              out_fig / "comparison_radar.png")

    # f) Grouped bar (all metrics per model)
    fig_grouped_bar(metrics_df,
                    out_fig / "comparison_grouped_bar.png")

    # Also copy all comparison graphs to comparison_graphs/ at repo root
    import shutil
    for png_name in [
        "comparison_bar_metrics.png",
        "comparison_cv_r2.png",
        "comparison_actual_vs_predicted.png",
        "comparison_residuals.png",
        "comparison_radar.png",
        "comparison_grouped_bar.png",
    ]:
        src = out_fig / png_name
        if src.exists():
            shutil.copy2(src, cg_dir / png_name)

    print(f"\nAll graphs also copied to: {cg_dir}")

    # ------------------------------------------------------------------
    # 7. Summary table
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("HOLD-OUT TEST RESULTS (sorted by R²)")
    print("=" * 65)
    print(metrics_df[["Model", "R2", "RMSE", "MAE"]].to_string(index=False))

    print("\n" + "=" * 65)
    print("10-FOLD CV RESULTS (sorted by CV R²)")
    print("=" * 65)
    print(cv_df[["Model", "CV_R2_mean", "CV_R2_std"]].to_string(index=False))

    best = metrics_df.iloc[0]
    print(f"\n✓ Best model on hold-out: {best['Model']}  (R²={best['R2']:.4f})")


if __name__ == "__main__":
    main()
