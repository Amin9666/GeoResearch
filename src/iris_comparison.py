"""
Master Comparison: Linear Regression vs Symbolic Regression (LassoCV)
vs Random Forest vs MLP Regressor on the Iris Regression Task.

Predict petal_length (cm) from sepal_length, sepal_width, petal_width.

Outputs (outputs/figures/):
  iris_comparison_metrics_bar.png
  iris_comparison_actual_vs_predicted.png
  iris_comparison_residuals_scatter.png
  iris_comparison_residuals_hist.png
  iris_comparison_error_distribution_boxplot.png
  iris_comparison_cv_r2_bar.png
  iris_comparison_cv_r2_violin.png
  iris_comparison_feature_distributions.png
  iris_comparison_pairplot.png
  iris_comparison_prediction_error_vs_actual.png
  iris_comparison_qq_residuals.png
  iris_comparison_learning_curves.png

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
from scipy.stats import probplot
from sklearn.datasets import load_iris
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, learning_curve, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

warnings.filterwarnings("ignore")

# Consistent colour palette
PALETTE = {
    "Linear Regression": "#4878CF",
    "Symbolic Regression": "#C4AD66",
    "Random Forest": "#D65F5F",
    "MLP": "#6ACC65",
}
MODEL_NAMES = list(PALETTE.keys())

_OFFSET = 1.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
    }


def build_symbolic_features(X: pd.DataFrame) -> pd.DataFrame:
    """Build engineered features for symbolic regression (same as iris_symbolic_regression.py)."""
    cols = list(X.columns)
    Xf: dict[str, np.ndarray] = {}
    for col in cols:
        v = X[col].values.astype(float)
        Xf[col] = v
        Xf[f"log_{col}"] = np.log(v + _OFFSET)
        Xf[f"sqrt_{col}"] = np.sqrt(np.abs(v))
        Xf[f"{col}_sq"] = v ** 2
        Xf[f"{col}_inv"] = _OFFSET / (v + _OFFSET)
    for i, c1 in enumerate(cols):
        for c2 in cols[i + 1:]:
            v1 = X[c1].values.astype(float)
            v2 = X[c2].values.astype(float)
            Xf[f"{c1}x{c2}"] = v1 * v2
            Xf[f"{c1}_div_{c2}"] = v1 / (v2 + _OFFSET)
            Xf[f"{c2}_div_{c1}"] = v2 / (v1 + _OFFSET)
    return pd.DataFrame(Xf)


class SymbolicFeatureTransformer(BaseEstimator, TransformerMixin):
    """Sklearn-compatible transformer that applies build_symbolic_features."""

    def fit(self, X, y=None):  # noqa: N803
        self._feature_names = [f"f{i}" for i in range(X.shape[1])]
        return self

    def transform(self, X, y=None):  # noqa: N803
        df = pd.DataFrame(X, columns=self._feature_names)
        result = build_symbolic_features(df).values
        # Replace any NaN/inf produced by feature engineering with 0
        result = np.where(np.isfinite(result), result, 0.0)
        return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out_fig = root / "outputs" / "figures"
    out_csv = root / "outputs" / "csv"
    out_mdl = root / "outputs" / "models"
    for d in (out_fig, out_csv, out_mdl):
        d.mkdir(parents=True, exist_ok=True)

    sns.set_style("whitegrid")

    # ------------------------------------------------------------------
    # 1. Load iris dataset
    # ------------------------------------------------------------------
    iris = load_iris(as_frame=True)
    df = iris.frame.copy()
    df.columns = [c.replace(" (cm)", "").replace(" ", "_") for c in df.columns]
    species_labels = [iris.target_names[t] for t in df["target"]]

    feature_cols = ["sepal_length", "sepal_width", "petal_width"]
    target_col = "petal_length"
    all_feature_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

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
    # 3. Build symbolic features for SR
    # ------------------------------------------------------------------
    Xf = build_symbolic_features(X)
    idx_train_bool = X_train.index
    idx_test_bool = X_test.index
    Xf_train = Xf.loc[idx_train_bool].values
    Xf_test = Xf.loc[idx_test_bool].values

    scaler_sr = RobustScaler()
    Xf_train_s = scaler_sr.fit_transform(Xf_train)
    Xf_test_s = scaler_sr.transform(Xf_test)

    # Scaled raw features for LR and MLP
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

    # Symbolic Regression (LassoCV)
    alphas = np.logspace(-6, 1, 80)
    sr = LassoCV(cv=10, max_iter=50000, random_state=42, alphas=alphas)
    sr.fit(Xf_train_s, y_train_np)
    y_pred_sr = sr.predict(Xf_test_s)

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

    # SR CV (manual, needs feature engineering inside each fold)
    cv_scores_sr = []
    for tr_idx, ts_idx in kf.split(X.values):
        X_tr_raw = pd.DataFrame(X.values[tr_idx], columns=feature_cols)
        X_ts_raw = pd.DataFrame(X.values[ts_idx], columns=feature_cols)
        Xf_tr = build_symbolic_features(X_tr_raw).values
        Xf_ts = build_symbolic_features(X_ts_raw).values
        sc = RobustScaler()
        Xf_tr_s = sc.fit_transform(Xf_tr)
        Xf_ts_s = sc.transform(Xf_ts)
        m_cv = LassoCV(cv=5, max_iter=10000, random_state=42, alphas=np.logspace(-6, 1, 40))
        m_cv.fit(Xf_tr_s, y.values[tr_idx])
        cv_scores_sr.append(r2_score(y.values[ts_idx], m_cv.predict(Xf_ts_s)))
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

    # --- (c) 2×2 Residuals vs Predicted scatter ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, name in zip(axes.flat, MODEL_NAMES):
        y_pred = predictions[name]
        res = y_test_np - y_pred
        ax.scatter(y_pred, res, alpha=0.6, s=30, color=PALETTE[name],
                   edgecolors="white", linewidths=0.4)
        ax.axhline(0, color="black", linestyle="--", linewidth=1.5)
        ax.set_title(name, fontsize=11)
        ax.set_xlabel("Predicted (cm)")
        ax.set_ylabel("Residual")
    fig.suptitle("Iris Regression — Residuals vs Predicted (All Models)", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_fig / "iris_comparison_residuals_scatter.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # --- (d) Overlaid residuals histogram ---
    fig, ax = plt.subplots(figsize=(10, 6))
    for name in MODEL_NAMES:
        res = y_test_np - predictions[name]
        ax.hist(res, bins=20, alpha=0.45, density=True, label=name, color=PALETTE[name])
        sns.kdeplot(res, ax=ax, color=PALETTE[name], linewidth=1.5)
    ax.axvline(0, color="black", linestyle="--", linewidth=1.5)
    ax.set_xlabel("Residual (Actual − Predicted, cm)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Iris Regression — Residual Distributions (All Models)", fontsize=13)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(out_fig / "iris_comparison_residuals_hist.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # --- (e) Boxplot of absolute errors ---
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

    # --- (f) CV R² bar chart with error bars ---
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

    # --- (g) Violin plot of 10-fold CV R² scores ---
    cv_df_long = pd.DataFrame({
        name: cv_all[name] for name in MODEL_NAMES
    })
    cv_df_melt = cv_df_long.melt(var_name="Model", value_name="CV R²")

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.violinplot(
        data=cv_df_melt, x="Model", y="CV R²", ax=ax,
        palette=PALETTE, inner="quartile", cut=0,
    )
    ax.set_title("Iris Regression — CV R² Violin Plot (10 Folds)", fontsize=13)
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("CV R²", fontsize=12)
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(out_fig / "iris_comparison_cv_r2_violin.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # --- (h) KDE distributions of 3 input features coloured by species ---
    species_col = pd.Series(species_labels, index=df.index, name="species")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    species_colors = {"setosa": "#E74C3C", "versicolor": "#3498DB", "virginica": "#2ECC71"}
    for ax, feat in zip(axes, feature_cols):
        for sp, grp in df.groupby(species_col):
            sns.kdeplot(grp[feat], ax=ax, label=sp, color=species_colors.get(sp, "grey"),
                        linewidth=2, fill=True, alpha=0.25)
        ax.set_title(f"{feat} distribution", fontsize=11)
        ax.set_xlabel(f"{feat} (cm)", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.legend(fontsize=9)
    fig.suptitle("Iris — Input Feature KDE Distributions by Species", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_fig / "iris_comparison_feature_distributions.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # --- (i) Pairplot coloured by species ---
    df_plot = df[all_feature_cols].copy()
    df_plot["species"] = species_labels
    pair_palette = {"setosa": "#E74C3C", "versicolor": "#3498DB", "virginica": "#2ECC71"}
    pairfig = sns.pairplot(df_plot, hue="species", palette=pair_palette, plot_kws={"alpha": 0.5, "s": 20})
    pairfig.figure.suptitle("Iris — Pairplot of All Features by Species", y=1.02, fontsize=13)
    pairfig.savefig(out_fig / "iris_comparison_pairplot.png", dpi=300, bbox_inches="tight")
    plt.close(pairfig.figure)

    # --- (j) abs_error vs y_actual, all models overlaid ---
    fig, ax = plt.subplots(figsize=(9, 6))
    for name in MODEL_NAMES:
        abs_err = np.abs(y_test_np - predictions[name])
        ax.scatter(y_test_np, abs_err, alpha=0.5, s=25, color=PALETTE[name], label=name)
    ax.set_xlabel("Actual petal_length (cm)", fontsize=12)
    ax.set_ylabel("Absolute Error (cm)", fontsize=12)
    ax.set_title("Iris Regression — Prediction Error vs Actual (All Models)", fontsize=13)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(out_fig / "iris_comparison_prediction_error_vs_actual.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # --- (k) Q-Q plots of residuals (2×2) ---
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for ax, name in zip(axes.flat, MODEL_NAMES):
        res = y_test_np - predictions[name]
        (osm, osr), (slope, intercept_qq, r_val) = probplot(res, dist="norm")
        ax.scatter(osm, osr, s=20, alpha=0.7, color=PALETTE[name])
        ax.plot(osm, slope * np.array(osm) + intercept_qq, color="black",
                linewidth=1.5, linestyle="--", label="Normal line")
        ax.set_title(f"{name} — Q-Q Plot of Residuals", fontsize=10)
        ax.set_xlabel("Theoretical Quantiles")
        ax.set_ylabel("Sample Quantiles")
        ax.legend(fontsize=8)
    fig.suptitle("Iris Regression — Q-Q Plots of Residuals (All Models)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_fig / "iris_comparison_qq_residuals.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # --- (l) Learning curves for all 4 models ---
    train_sizes_frac = np.linspace(0.1, 1.0, 10)
    lc_kf = KFold(n_splits=5, shuffle=True, random_state=42)

    pipe_lr_lc = Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())])
    pipe_rf_lc = Pipeline([("model", RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))])
    pipe_mlp_lc = Pipeline([
        ("scaler", StandardScaler()),
        ("model", MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)),
    ])

    # For SR, build a pipeline that applies feature engineering then scales and fits LassoCV
    pipe_sr_lc = Pipeline([
        ("engineer", SymbolicFeatureTransformer()),
        ("scaler", RobustScaler()),
        ("model", LassoCV(cv=5, max_iter=5000, random_state=42, alphas=np.logspace(-6, 1, 40))),
    ])

    lc_models = {
        "Linear Regression": pipe_lr_lc,
        "Symbolic Regression": pipe_sr_lc,
        "Random Forest": pipe_rf_lc,
        "MLP": pipe_mlp_lc,
    }

    fig, ax = plt.subplots(figsize=(10, 7))
    for name, pipe_lc in lc_models.items():
        train_sz, train_sc, val_sc = learning_curve(
            pipe_lc, X, y, cv=lc_kf,
            train_sizes=train_sizes_frac,
            scoring="r2", n_jobs=-1,
        )
        train_mean = train_sc.mean(axis=1)
        val_mean = val_sc.mean(axis=1)
        val_std = val_sc.std(axis=1)
        color = PALETTE[name]
        ax.plot(train_sz, train_mean, "--", color=color, linewidth=1.5, alpha=0.7)
        ax.plot(train_sz, val_mean, "-", color=color, linewidth=2, label=name)
        ax.fill_between(train_sz, val_mean - val_std, val_mean + val_std, alpha=0.12, color=color)

    ax.set_xlabel("Training Set Size", fontsize=12)
    ax.set_ylabel("R² Score", fontsize=12)
    ax.set_title("Iris Regression — Learning Curves (solid=CV, dashed=Train)", fontsize=13)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(out_fig / "iris_comparison_learning_curves.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------
    # 9. Final summary
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
