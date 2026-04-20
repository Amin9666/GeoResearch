"""
Exploratory Data Analysis for Fault_AE_Combined_Load_Final.xlsx.

Target  : Load  (column C)
Features: RISE, COUN, ENER, DURATION, AMP, A-FRQ, RMS, ASL, PCNTS,
          R-FRQ, I-FRQ, ABS-ENERGY, FRQ-C, P-FRQ  (columns D-Q)

Outputs:
  outputs/figures/fault_ae/eda_correlation_heatmap.png
  outputs/figures/fault_ae/eda_scatter_matrix.png
  outputs/figures/fault_ae/eda_boxplots.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ── Paths ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "Fault_AE_Combined_Load_Final.xlsx"
OUT_FIG = ROOT / "outputs" / "figures" / "fault_ae"
OUT_FIG.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
    "RISE", "COUN", "ENER", "DURATION", "AMP", "A-FRQ",
    "RMS", "ASL", "PCNTS", "R-FRQ", "I-FRQ", "ABS-ENERGY", "FRQ-C", "P-FRQ",
]
TARGET_COL = "Load"

sns.set_style("whitegrid")


def load_data() -> pd.DataFrame:
    df = pd.read_excel(DATA_PATH)
    return df[[TARGET_COL] + FEATURE_COLS].copy()


# ── 1. Correlation Heatmap ──────────────────────────────────────────────
def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    corr = df.corr()

    fig, ax = plt.subplots(figsize=(14, 11))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(
        corr, mask=mask, cmap=cmap, vmin=-1, vmax=1, center=0,
        annot=True, fmt=".2f", linewidths=0.4, ax=ax,
        annot_kws={"size": 7},
    )
    ax.set_title("Fault AE — Correlation Matrix (Lower Triangle)", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT_FIG / "eda_correlation_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: eda_correlation_heatmap.png")


# ── 2. Scatter Matrix (pairplot) for Load + top-5 correlated features ──
def plot_scatter_matrix(df: pd.DataFrame) -> None:
    # Select Load + 5 features most correlated with Load for readability
    corr_with_target = df.corr()[TARGET_COL].drop(TARGET_COL).abs().sort_values(ascending=False)
    top_features = corr_with_target.head(5).index.tolist()
    cols = [TARGET_COL] + top_features

    g = sns.pairplot(
        df[cols], diag_kind="kde", plot_kws={"alpha": 0.3, "s": 10},
        height=2.2,
    )
    g.figure.suptitle(
        "Fault AE — Scatter Matrix (Load + Top-5 Correlated Features)",
        y=1.02, fontsize=13,
    )
    g.figure.savefig(OUT_FIG / "eda_scatter_matrix.png", dpi=200, bbox_inches="tight")
    plt.close(g.figure)
    print(f"  Saved: eda_scatter_matrix.png  (features: {top_features})")


# ── 3. Boxplots ─────────────────────────────────────────────────────────
def plot_boxplots(df: pd.DataFrame) -> None:
    """
    Two panels: one for the target, one for all features.
    Features with high variance are shown on log-scale.
    """
    all_cols = [TARGET_COL] + FEATURE_COLS
    n = len(all_cols)
    ncols = 5
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(18, nrows * 3.5))
    axes = axes.flatten()

    for i, col in enumerate(all_cols):
        ax = axes[i]
        data = df[col].dropna()
        color = "#4878CF" if col == TARGET_COL else "#60BD68"
        ax.boxplot(data, vert=True, patch_artist=True,
                   boxprops=dict(facecolor=color, alpha=0.6),
                   medianprops=dict(color="black", linewidth=2),
                   whiskerprops=dict(linewidth=1.2),
                   capprops=dict(linewidth=1.2),
                   flierprops=dict(marker="o", markersize=2, alpha=0.3))
        ax.set_title(col, fontsize=10, fontweight="bold")
        ax.set_xticks([])
        ax.set_ylabel("Value", fontsize=8)
        # Use log scale for highly skewed columns
        if data.max() > 1000 * (data.median() + 1e-9):
            ax.set_yscale("log")
            ax.set_ylabel("Value (log)", fontsize=8)

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Fault AE — Boxplots of Target and Features", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(OUT_FIG / "eda_boxplots.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: eda_boxplots.png")


def main() -> None:
    print("=" * 60)
    print("EDA — Fault AE Dataset")
    print("=" * 60)
    df = load_data()
    print(f"  Rows: {len(df)}, Features: {len(FEATURE_COLS)}, Target: {TARGET_COL}")

    plot_correlation_heatmap(df)
    plot_scatter_matrix(df)
    plot_boxplots(df)

    print(f"\nAll EDA figures saved → {OUT_FIG}")


if __name__ == "__main__":
    main()
