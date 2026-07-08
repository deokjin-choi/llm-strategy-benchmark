"""
Bootstrap 95% CI, p-values, and FDR for mean RDS by strategy (§5.5, Fig. 7).

Method A (condition × strategy cells): each cell is
(Model, Temperature, scenario, Num Context, context_variant, strategy).
Within each cell, RDS is averaged over matched pairs; strategy-level means
macro-average cell means with equal weight per cell. Bootstrap resamples
cells with replacement (10,000 replicates).

Overall mean RDS uses the same cell macro-average (all strategies pooled).

H0 for strategy tests: mean RDS = 0 (two-sided bootstrap p-value; FDR
across seven strategies).

Outputs
-------
  final_results/summary/rationale_rds_overall_ci.csv
  final_results/summary/rationale_rds_by_strategy_ci.csv
  final_results/plots/eval_rationale_rds_strategy_boxplot.png   (boxplot; stats in CSV/text only)

Usage
-----
  python -m result_analysis.rds_by_strategy_ci
"""

from __future__ import annotations

import os
from typing import List, Tuple

import numpy as np
import pandas as pd

try:
    from result_analysis.model_behavioral_profile import valid_strategies
    from result_analysis.rationale_analysis import load_rationale_data, _bh_fdr
    from result_analysis.rationale_rds_analysis import (
        PLOTS_DIR,
        SUMMARY_DIR,
        _build_rds_pair_df,
        _legacy_heatmap_csv_to_matrix,
        _plot_rds_strategy_boxplot,
        _prepare_df,
    )
except ImportError:
    from model_behavioral_profile import valid_strategies
    from rationale_analysis import load_rationale_data, _bh_fdr
    from rationale_rds_analysis import (
        CONTEXT_COLORS,
        CONTEXT_ORDER,
        PLOTS_DIR,
        STRATEGY_ORDER,
        SUMMARY_DIR,
        _build_rds_heatmap_df,
        _build_rds_pair_df,
        _legacy_heatmap_csv_to_matrix,
        _prepare_df,
    )

BOOTSTRAP_N = 10_000
BOOTSTRAP_SEED = 42
NPZ_PATH = os.path.join(SUMMARY_DIR, "rationale_rds_calibration_distances.npz")
CELL_COLS = [
    "Model",
    "Temperature",
    "scenario",
    "Num Context",
    "context_variant",
    "strategy",
]


def load_rds_pairs_with_distances(input_dir: str = "infer_results") -> pd.DataFrame:
    """Rebuild matched pair table and attach saved RDS distances (no re-embedding)."""
    if not os.path.exists(NPZ_PATH):
        raise FileNotFoundError(
            f"Missing {NPZ_PATH}. Run: python -m result_analysis.rationale_rds_analysis"
        )
    df = _prepare_df(load_rationale_data(input_dir=input_dir))
    pairs = _build_rds_pair_df(df)
    rds = np.load(NPZ_PATH)["rds"]
    if len(rds) != len(pairs):
        raise ValueError(
            f"Pair count mismatch: pairs={len(pairs):,}, npz={len(rds):,}. "
            "Re-run rationale_rds_analysis."
        )
    pairs = pairs.copy()
    pairs["rds"] = rds
    return pairs


def build_cell_level_rds(pairs: pd.DataFrame) -> pd.DataFrame:
    """One row per condition×strategy cell with mean RDS."""
    return (
        pairs.groupby(CELL_COLS, dropna=False)["rds"]
        .mean()
        .reset_index(name="mean_rds")
    )


def _bootstrap_pvalue_two_sided(observed: float, boot_samples: np.ndarray) -> float:
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


def _bootstrap_cell_means(
    cell_df: pd.DataFrame,
    value_col: str = "mean_rds",
    *,
    n_boot: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_SEED,
) -> Tuple[float, float, float, np.ndarray, int]:
    """Macro mean over cells with bootstrap resampling."""
    vals = cell_df[value_col].astype(float).to_numpy()
    n_cells = len(vals)
    observed = float(np.mean(vals))
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n_cells, n_cells)
        boot[b] = float(np.mean(vals[idx]))
    ci_lo = float(np.percentile(boot, 2.5))
    ci_hi = float(np.percentile(boot, 97.5))
    return observed, ci_lo, ci_hi, boot, n_cells


def compute_rds_overall_ci(
    cell_df: pd.DataFrame,
    *,
    n_boot: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_SEED,
) -> pd.DataFrame:
    observed, ci_lo, ci_hi, boot, n_cells = _bootstrap_cell_means(
        cell_df, n_boot=n_boot, seed=seed,
    )
    p = _bootstrap_pvalue_two_sided(observed, boot)
    return pd.DataFrame([{
        "mean_rds_macro": observed,
        "ci_lower": ci_lo,
        "ci_upper": ci_hi,
        "p_value": p,
        "n_cells": n_cells,
    }])


def compute_rds_by_strategy_ci(
    cell_df: pd.DataFrame,
    *,
    n_boot: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """Bootstrap cell-level macro means per strategy."""
    rows: List[dict] = []
    for i, strat in enumerate(valid_strategies):
        sub = cell_df[cell_df["strategy"] == strat]
        if sub.empty:
            continue
        observed, ci_lo, ci_hi, boot, n_cells = _bootstrap_cell_means(
            sub, n_boot=n_boot, seed=seed + i,
        )
        rows.append({
            "Strategy": strat,
            "mean_rds": observed,
            "ci_lower": ci_lo,
            "ci_upper": ci_hi,
            "p_value": _bootstrap_pvalue_two_sided(observed, boot),
            "n_cells": n_cells,
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["q_value_fdr"] = _bh_fdr(out["p_value"].to_numpy())
    out["sig_stars"] = out["q_value_fdr"].map(lambda q: _sig_stars(float(q)))
    return out


def plot_rds_strategy_boxplot_fig7(save_dir: str = PLOTS_DIR) -> str:
    """Paper Fig. 7: boxplot of context-variant cell means (no CI/stars on figure)."""
    path = os.path.join(SUMMARY_DIR, "rationale_rds_heatmap.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing {path}. Run: python -m result_analysis.rationale_rds_analysis"
        )
    heatmap_df = _legacy_heatmap_csv_to_matrix(pd.read_csv(path))
    out = os.path.join(save_dir, "eval_rationale_rds_strategy_boxplot.png")
    _plot_rds_strategy_boxplot(heatmap_df, out)
    return out


def run(input_dir: str = "infer_results") -> Tuple[pd.DataFrame, pd.DataFrame]:
    os.makedirs(SUMMARY_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    print("Loading RDS pairs (cached distances)...")
    pairs = load_rds_pairs_with_distances(input_dir=input_dir)
    print(f"  Pairs: {len(pairs):,}")

    cell_df = build_cell_level_rds(pairs)
    print(f"  Condition×strategy cells: {len(cell_df):,}")

    overall_df = compute_rds_overall_ci(cell_df)
    overall_df["mean_rds_pairs"] = float(pairs["rds"].mean())
    overall_df["median_rds_pairs"] = float(pairs["rds"].median())
    overall_df["n_pairs"] = int(len(pairs))

    ci_df = compute_rds_by_strategy_ci(cell_df)

    out_overall = os.path.join(SUMMARY_DIR, "rationale_rds_overall_ci.csv")
    overall_df.to_csv(out_overall, index=False)
    print(f"Saved → {out_overall}")

    out_strat = os.path.join(SUMMARY_DIR, "rationale_rds_by_strategy_ci.csv")
    ci_df.to_csv(out_strat, index=False)
    print(f"Saved → {out_strat}")

    out_plot = plot_rds_strategy_boxplot_fig7()
    print(f"Saved → {out_plot}")

    print("\nOverall (macro cell mean):")
    print(overall_df.to_string(index=False))
    print("\nBy strategy:")
    print(ci_df.to_string(index=False))
    return overall_df, ci_df


if __name__ == "__main__":
    run()
