"""
Definitive Comparison: Linear Regression vs Three ML Models vs Symbolic Regression.

Methodology:
- Data: Acoustic emission parameters from rock shear testing
- Preprocessing: IQR 1.0 aggressive outlier removal
- ML models (Linear, SVR, RF, GB): trained on 5 raw AE features
- Symbolic Regression: trained on 45 engineered feature candidates;
  LassoCV selects a sparse interpretable formula automatically
- Evaluation: 80/20 train-test split + 10-fold cross-validation

Outputs:
  outputs/csv/final_comparison_metrics.csv
  outputs/csv/final_comparison_cv_metrics.csv
  outputs/csv/final_comparison_predictions.csv
  outputs/figures/final_comparison_metrics.png
  outputs/figures/final_comparison_cv_r2.png
  outputs/figures/final_comparison_actual_vs_predicted.png
  outputs/figures/final_comparison_residuals.png
  outputs/models/sr_lasso_model.pkl
  outputs/models/sr_scaler.pkl
  outputs/csv/sr_formula_terms.csv
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVR

import warnings

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def remove_outliers_iqr(data: pd.DataFrame, features: list[str], mult: float = 1.0) -> pd.DataFrame:
    mask = np.ones(len(data), dtype=bool)
    for feat in features:
        q1, q3 = data[feat].quantile(0.25), data[feat].quantile(0.75)
        iqr = q3 - q1
        mask &= (data[feat] >= q1 - mult * iqr) & (data[feat] <= q3 + mult * iqr)
    return data[mask].reset_index(drop=True)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
    }


# Numerical stability offset added before log() and division to avoid log(0)/div-by-zero
_OFFSET = 1.0


def build_symbolic_features(X: pd.DataFrame) -> pd.DataFrame:
    """Create 45 candidate symbolic features from 5 raw AE parameters."""
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
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data_path = root / "data" / "raw" / "Shear_Data_15.csv"
    if not data_path.exists():
        data_path = root / "Shear_Data_15.csv"

    out_csv = root / "outputs" / "csv"
    out_fig = root / "outputs" / "figures"
    out_mdl = root / "outputs" / "models"
    for d in (out_csv, out_fig, out_mdl):
        d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load & clean
    # ------------------------------------------------------------------
    df = pd.read_csv(data_path)
    features = ["RISE", "COUN", "ENER", "DURATION", "AMP"]
    target = "SHEAR STRESS"

    df = remove_outliers_iqr(df, features + [target], mult=1.0)
    print(f"Samples after IQR 1.0 cleaning: {len(df)}")

    X_raw = df[features]
    y = df[target]

    idx = np.arange(len(y))
    idx_train, idx_test = train_test_split(idx, test_size=0.2, random_state=42)

    X_raw_train = X_raw.iloc[idx_train].values
    X_raw_test = X_raw.iloc[idx_test].values
    y_train = y.values[idx_train]
    y_test = y.values[idx_test]

    # Scale raw features for ML models
    scaler_raw = RobustScaler()
    X_raw_train_s = scaler_raw.fit_transform(X_raw_train)
    X_raw_test_s = scaler_raw.transform(X_raw_test)

    # ------------------------------------------------------------------
    # 2. Build symbolic feature matrix
    # ------------------------------------------------------------------
    Xf = build_symbolic_features(X_raw)
    feature_names = list(Xf.columns)
    X_sym_train = Xf.values[idx_train]
    X_sym_test = Xf.values[idx_test]

    scaler_sym = RobustScaler()
    X_sym_train_s = scaler_sym.fit_transform(X_sym_train)
    X_sym_test_s = scaler_sym.transform(X_sym_test)

    # ------------------------------------------------------------------
    # 3. Train models
    # ------------------------------------------------------------------
    ml_models: dict[str, object] = {
        "Linear Regression": LinearRegression(),
        "Support Vector Regression": SVR(kernel="rbf", C=10, epsilon=0.05, gamma="scale"),
        "Random Forest": RandomForestRegressor(
            n_estimators=300, max_depth=14, min_samples_leaf=5, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.03, max_depth=4, subsample=0.9,
            min_samples_leaf=5, random_state=42
        ),
    }

    predictions: dict[str, np.ndarray] = {}
    metrics_rows: list[dict] = []

    for name, model in ml_models.items():
        model.fit(X_raw_train_s, y_train)
        y_pred = model.predict(X_raw_test_s)
        predictions[name] = y_pred
        m = evaluate(y_test, y_pred)
        metrics_rows.append({"Model": name, "Type": "ML / Baseline", **m})
        print(f"{name}: R²={m['R2']:.4f}, RMSE={m['RMSE']:.4f}, MAE={m['MAE']:.4f}")

    # Symbolic regression via LassoCV on engineered features
    sr_model = LassoCV(
        cv=10, max_iter=30000, random_state=42, n_jobs=-1,
        alphas=np.logspace(-6, 0, 60)
    )
    sr_model.fit(X_sym_train_s, y_train)
    y_pred_sr = sr_model.predict(X_sym_test_s)
    predictions["Symbolic Regression"] = y_pred_sr
    m_sr = evaluate(y_test, y_pred_sr)
    metrics_rows.append({"Model": "Symbolic Regression", "Type": "Symbolic", **m_sr})
    print(f"Symbolic Regression: R²={m_sr['R2']:.4f}, RMSE={m_sr['RMSE']:.4f}, MAE={m_sr['MAE']:.4f}")

    # ------------------------------------------------------------------
    # 4. Extract and save the symbolic formula
    # ------------------------------------------------------------------
    coef = pd.Series(sr_model.coef_, index=feature_names)
    nz_coef = coef[coef != 0].sort_values(key=abs, ascending=False)
    intercept = float(sr_model.intercept_)

    formula_df = pd.DataFrame({"Term": nz_coef.index, "Coefficient": nz_coef.values})
    formula_df.loc[len(formula_df)] = {"Term": "INTERCEPT", "Coefficient": intercept}
    formula_df.to_csv(out_csv / "sr_formula_terms.csv", index=False)

    # Save models
    with open(out_mdl / "sr_lasso_model.pkl", "wb") as f:
        pickle.dump(sr_model, f)
    with open(out_mdl / "sr_scaler.pkl", "wb") as f:
        pickle.dump(scaler_sym, f)
    with open(out_mdl / "sr_feature_names.pkl", "wb") as f:
        pickle.dump(feature_names, f)

    # ------------------------------------------------------------------
    # 5. 10-fold cross-validation
    # ------------------------------------------------------------------
    print("\nRunning 10-fold cross-validation...")
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    cv_rows: list[dict] = []

    def _cv_r2(model_fn: object, X_arr: np.ndarray, y_arr: np.ndarray) -> tuple[float, float]:
        scores = []
        for tr_i, ts_i in kf.split(X_arr):
            sc = RobustScaler()
            Xtr_cv = sc.fit_transform(X_arr[tr_i])
            Xts_cv = sc.transform(X_arr[ts_i])
            m = model_fn()
            m.fit(Xtr_cv, y_arr[tr_i])
            scores.append(r2_score(y_arr[ts_i], m.predict(Xts_cv)))
        return float(np.mean(scores)), float(np.std(scores))

    cv_model_fns: dict[str, object] = {
        "Linear Regression": lambda: LinearRegression(),
        "Support Vector Regression": lambda: SVR(kernel="rbf", C=10, epsilon=0.05, gamma="scale"),
        "Random Forest": lambda: RandomForestRegressor(
            n_estimators=300, max_depth=14, min_samples_leaf=5, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": lambda: GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.03, max_depth=4, subsample=0.9,
            min_samples_leaf=5, random_state=42
        ),
    }
    for name, fn in cv_model_fns.items():
        mean_r2, std_r2 = _cv_r2(fn, X_raw.values, y.values)
        cv_rows.append({"Model": name, "CV_R2_mean": mean_r2, "CV_R2_std": std_r2})
        print(f"  {name}: CV R²={mean_r2:.4f} ± {std_r2:.4f}")

    # SR cross-validation (on engineered features)
    sr_cv_scores: list[float] = []
    for tr_i, ts_i in kf.split(Xf.values):
        sc = RobustScaler()
        Xtr_cv = sc.fit_transform(Xf.values[tr_i])
        Xts_cv = sc.transform(Xf.values[ts_i])
        m = LassoCV(cv=10, max_iter=30000, random_state=42, n_jobs=-1, alphas=np.logspace(-6, 0, 60))
        m.fit(Xtr_cv, y.values[tr_i])
        sr_cv_scores.append(r2_score(y.values[ts_i], m.predict(Xts_cv)))
    sr_cv_mean = float(np.mean(sr_cv_scores))
    sr_cv_std = float(np.std(sr_cv_scores))
    cv_rows.append({"Model": "Symbolic Regression", "CV_R2_mean": sr_cv_mean, "CV_R2_std": sr_cv_std})
    print(f"  Symbolic Regression: CV R²={sr_cv_mean:.4f} ± {sr_cv_std:.4f}")

    cv_df = pd.DataFrame(cv_rows).sort_values("CV_R2_mean", ascending=False)
    cv_df.to_csv(out_csv / "final_comparison_cv_metrics.csv", index=False)

    # ------------------------------------------------------------------
    # 6. Save metrics and predictions CSVs
    # ------------------------------------------------------------------
    metrics_df = pd.DataFrame(metrics_rows).sort_values("R2", ascending=False)
    metrics_df.to_csv(out_csv / "final_comparison_metrics.csv", index=False)

    pred_df = pd.DataFrame({"Actual": y_test})
    for name, p in predictions.items():
        pred_df[name.replace(" ", "_")] = p
    pred_df.to_csv(out_csv / "final_comparison_predictions.csv", index=False)

    # ------------------------------------------------------------------
    # 7. Figures
    # ------------------------------------------------------------------
    sns.set_style("whitegrid")
    palette_map = {
        "Linear Regression": "#4878CF",
        "Support Vector Regression": "#6ACC65",
        "Random Forest": "#D65F5F",
        "Gradient Boosting": "#B47CC7",
        "Symbolic Regression": "#C4AD66",
    }
    colors = [palette_map[m] for m in metrics_df["Model"]]

    # Bar chart: hold-out test metrics
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, metric in zip(axes, ["R2", "RMSE", "MAE"]):
        bars = ax.bar(metrics_df["Model"], metrics_df[metric], color=colors, edgecolor="black", linewidth=0.5)
        ax.set_title(f"Model Comparison — {metric}", fontsize=13)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=30)
        for bar, val in zip(bars, metrics_df[metric]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_fig / "final_comparison_metrics.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Cross-validation R² bar chart with error bars
    cv_colors = [palette_map[m] for m in cv_df["Model"]]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(cv_df["Model"], cv_df["CV_R2_mean"], yerr=cv_df["CV_R2_std"],
                  color=cv_colors, edgecolor="black", linewidth=0.5,
                  capsize=5, error_kw={"linewidth": 1.5})
    ax.set_title("10-Fold Cross-Validation R² Comparison", fontsize=13)
    ax.set_ylabel("Mean CV R²")
    ax.tick_params(axis="x", rotation=30)
    for bar, (_, row) in zip(bars, cv_df.iterrows()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + row["CV_R2_std"] + 0.001,
                f"{row['CV_R2_mean']:.4f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_fig / "final_comparison_cv_r2.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Actual vs predicted scatter
    n = len(predictions)
    cols_per_row = 3
    rows = int(np.ceil(n / cols_per_row))
    fig, axes = plt.subplots(rows, cols_per_row, figsize=(15, 5 * rows))
    axes_flat = np.array(axes).reshape(-1)
    all_vals = np.concatenate([y_test] + list(predictions.values()))
    vmin, vmax = all_vals.min(), all_vals.max()

    for idx_plot, (name, y_pred) in enumerate(predictions.items()):
        ax = axes_flat[idx_plot]
        color = palette_map.get(name, "#999999")
        ax.scatter(y_test, y_pred, alpha=0.4, s=12, color=color)
        ax.plot([vmin, vmax], [vmin, vmax], "k--", linewidth=1.5)
        m = evaluate(y_test, y_pred)
        ax.set_title(name, fontsize=11)
        ax.set_xlabel("Actual (MPa)")
        ax.set_ylabel("Predicted (MPa)")
        ax.text(0.04, 0.96, f"R²={m['R2']:.4f}\nRMSE={m['RMSE']:.4f}",
                transform=ax.transAxes, ha="left", va="top", fontsize=9,
                bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8})

    for idx_plot in range(n, len(axes_flat)):
        axes_flat[idx_plot].axis("off")

    fig.suptitle("Actual vs Predicted Shear Stress — All Models", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(out_fig / "final_comparison_actual_vs_predicted.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Residual distributions
    fig, ax = plt.subplots(figsize=(12, 6))
    for name, y_pred in predictions.items():
        residuals = y_test - y_pred
        color = palette_map.get(name, "#999999")
        ax.hist(residuals, bins=50, alpha=0.45, label=name, density=True, color=color)
    ax.axvline(0, color="black", linestyle="--", linewidth=1.5)
    ax.set_title("Residual Distribution — All Models", fontsize=13)
    ax.set_xlabel("Residual (Actual − Predicted, MPa)")
    ax.set_ylabel("Density")
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(out_fig / "final_comparison_residuals.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------
    # 8. Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("HOLD-OUT TEST RESULTS (sorted by R²)")
    print("=" * 72)
    print(metrics_df[["Model", "R2", "RMSE", "MAE"]].to_string(index=False))

    print("\n" + "=" * 72)
    print("10-FOLD CROSS-VALIDATION RESULTS (sorted by CV R²)")
    print("=" * 72)
    print(cv_df[["Model", "CV_R2_mean", "CV_R2_std"]].to_string(index=False))

    print(f"\n--- Discovered Symbolic Formula ({len(nz_coef)} terms) ---")
    print(f"SHEAR_STRESS = {intercept:.6f}")
    for term, c in nz_coef.items():
        print(f"  {c:+.6f} × {term}")

    best = metrics_df.iloc[0]
    print(f"\n✓ Best model: {best['Model']} (R²={best['R2']:.4f})")


if __name__ == "__main__":
    main()
