"""
Bootstrap 95% CI, p-values, and FDR-corrected significance for
Δp = p(Specific) − p(Generic) by strategy (Fig. 4).

Condition-level bootstrap: each resample draws n_conditions conditions with
replacement, recomputes macro-average Δp per strategy (Method A logic).

Outputs
-------
  final_results/summary/specific_generic_delta_ci.csv
  final_results/plots/eval_fr_directionality_bars__ALL.png

Usage
-----
  python -m result_analysis.specific_generic_delta_ci
"""

from __future__ import annotations

import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from result_analysis.fr_directionality_overall import build_fr_directionality_summary_overall
    from result_analysis.model_behavioral_profile import (
        FIXED_CONDITION_ID_COLS,
        load_profile_data,
        valid_strategies,
        _strategy_distribution,
    )
    from result_analysis.rationale_analysis import _bh_fdr
except ImportError:
    from fr_directionality_overall import build_fr_directionality_summary_overall
    from model_behavioral_profile import (
        FIXED_CONDITION_ID_COLS,
        load_profile_data,
        valid_strategies,
        _strategy_distribution,
    )
    from rationale_analysis import _bh_fdr

SUMMARY_DIR = "./final_results/summary"
PLOTS_DIR = "./final_results/plots"
BOOTSTRAP_N = 10_000
BOOTSTRAP_SEED = 42


def build_condition_level_deltas(df: pd.DataFrame) -> np.ndarray:
    """Return (n_conditions, n_strategies) matrix of Δp vectors."""
    cond_cols = ["Model", "Temperature", *FIXED_CONDITION_ID_COLS, "context_variant"]
    per_cond: List[np.ndarray] = []
    for _, g in df.groupby(cond_cols, dropna=False):
        gg = g[g["problem_type"] == "generic"]["Standard Mapping"]
        ss = g[g["problem_type"] == "specific"]["Standard Mapping"]
        if len(gg) == 0 or len(ss) == 0:
            continue
        pg = _strategy_distribution(gg)
        ps = _strategy_distribution(ss)
        per_cond.append(ps - pg)
    if not per_cond:
        raise ValueError("No matched generic/specific conditions found.")
    return np.vstack(per_cond)


def _bootstrap_pvalue_two_sided(observed: float, boot_samples: np.ndarray) -> float:
    """Two-sided bootstrap p-value for H0: mean Δp = 0."""
    if observed >= 0:
        p = 2.0 * float(np.mean(boot_samples <= 0))
    else:
        p = 2.0 * float(np.mean(boot_samples >= 0))
    return min(p, 1.0)


def _sig_stars(q_fdr: float) -> str:
    if q_fdr < 0.001:
        return "***"
    if q_fdr < 0.01:
        return "**"
    if q_fdr < 0.05:
        return "*"
    return ""


def compute_specific_generic_delta_ci(
    df: pd.DataFrame,
    *,
    n_boot: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """Bootstrap condition-level Δp; return strategy-level CI, p, and FDR q."""
    D = build_condition_level_deltas(df)
    n_cond, n_strat = D.shape
    observed = np.nanmean(D, axis=0)

    rng = np.random.default_rng(seed)
    boot_means = np.empty((n_boot, n_strat), dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n_cond, n_cond)
        boot_means[b] = np.nanmean(D[idx], axis=0)

    ci_lower = np.percentile(boot_means, 2.5, axis=0)
    ci_upper = np.percentile(boot_means, 97.5, axis=0)
    p_values = np.array([
        _bootstrap_pvalue_two_sided(observed[j], boot_means[:, j])
        for j in range(n_strat)
    ])
    q_fdr = _bh_fdr(p_values)

    rows = []
    for j, strat in enumerate(valid_strategies):
        rows.append({
            "Strategy": strat,
            "mean_delta_specific_minus_generic": float(observed[j]),
            "ci_lower": float(ci_lower[j]),
            "ci_upper": float(ci_upper[j]),
            "p_value": float(p_values[j]),
            "q_value_fdr": float(q_fdr[j]),
            "sig_stars": _sig_stars(float(q_fdr[j])),
            "n_conditions": int(n_cond),
        })
    return pd.DataFrame(rows)


def plot_specific_generic_delta_bars(
    ci_df: pd.DataFrame,
    save_dir: str = PLOTS_DIR,
    filename: str = "eval_fr_directionality_bars__ALL.png",
) -> str:
    """Horizontal bar chart with 95% CI error bars and FDR significance stars."""
    os.makedirs(save_dir, exist_ok=True)
    order_idx = {s: i for i, s in enumerate(valid_strategies)}
    gg = ci_df[ci_df["Strategy"].isin(order_idx)].copy()
    gg["__order"] = gg["Strategy"].map(order_idx)
    gg = gg.sort_values("__order").reset_index(drop=True)

    vals = gg["mean_delta_specific_minus_generic"].astype(float).to_numpy()
    ci_lo = gg["ci_lower"].astype(float).to_numpy()
    ci_hi = gg["ci_upper"].astype(float).to_numpy()
    labels = gg["Strategy"].astype(str).to_numpy()
    stars = gg["sig_stars"].astype(str).to_numpy()

    xerr = np.vstack([vals - ci_lo, ci_hi - vals])

    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    y = np.arange(len(labels))
    colors = np.where(vals >= 0, "#2c7fb8", "#d95f0e")
    ax.barh(
        y, vals, xerr=xerr, color=colors, alpha=0.85,
        ecolor="#333333", capsize=3, error_kw={"linewidth": 1.0, "alpha": 0.9},
    )
    ax.axvline(0, color="#444", linewidth=1.0)

    span = float(np.nanmax(np.abs(np.r_[ci_lo, ci_hi, vals]))) if len(vals) else 1.0
    if not np.isfinite(span) or span <= 0:
        span = 1.0
    left = min(0.0, float(np.nanmin(ci_lo))) - 0.45 * span
    right = max(0.0, float(np.nanmax(ci_hi))) + 0.40 * span
    ax.set_xlim(left, right)

    gap = 0.03 * span
    for i, (v, star) in enumerate(zip(vals, stars)):
        # Place a single combined label just beyond the error-bar cap so the
        # value and stars never overlap the bar or the CI whiskers.
        text = f"{v:+.2f}{(' ' + star) if star else ''}"
        if v >= 0:
            label_x = ci_hi[i] + gap
            ha = "left"
        else:
            label_x = ci_lo[i] - gap
            ha = "right"
        ax.text(label_x, i, text, va="center", ha=ha, fontsize=8, color="#222")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Δp  (Specific − Generic)")
    ax.grid(True, axis="x", linestyle="--", alpha=0.3)
    plt.tight_layout()

    out = os.path.join(save_dir, filename)
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def run(input_dir: str = "infer_results") -> pd.DataFrame:
    os.makedirs(SUMMARY_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    print("Loading profile data...")
    df = load_profile_data(input_dir=input_dir)
    print(f"Rows loaded: {len(df):,}")

    ci_df = compute_specific_generic_delta_ci(df)
    # Cross-check point estimates against existing macro-average helper
    ref = build_fr_directionality_summary_overall(df).set_index("Strategy")
    merged = ci_df.set_index("Strategy")
    max_diff = (
        merged["mean_delta_specific_minus_generic"] - ref["mean_delta_specific_minus_generic"]
    ).abs().max()
    if max_diff > 1e-9:
        print(f"Warning: point estimate mismatch vs macro-average (max |Δ| = {max_diff:.2e})")

    out_csv = os.path.join(SUMMARY_DIR, "specific_generic_delta_ci.csv")
    ci_df.to_csv(out_csv, index=False)
    print(f"Saved → {out_csv}")

    out_plot = plot_specific_generic_delta_bars(ci_df)
    print(f"Saved → {out_plot}")

    display_cols = [
        "Strategy", "mean_delta_specific_minus_generic", "ci_lower", "ci_upper",
        "p_value", "q_value_fdr", "sig_stars", "n_conditions",
    ]
    print("\n" + ci_df[display_cols].to_string(index=False))
    return ci_df


if __name__ == "__main__":
    run()
