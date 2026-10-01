"""
RQ2 (JIIS revision): firm-identity shift and its stability across context variants.

Same estimators as the initial version (condition-level macro Δp, 10,000 bootstrap
resamples, BH FDR), with competitive_dynamics removed. Existing scripts and their
output files are left unchanged.

Outputs
-------
  final_results/summary/rq2_firm_delta_ci.csv
  final_results/summary/rq2_firm_delta_by_variant.csv
  final_results/summary/rq2_firm_interaction_ci.csv
  final_results/summary/rq2_firm_delta_spearman.csv
  final_results/plots/eval_rq2_firm_delta_bars.png
  final_results/plots/eval_rq2_firm_delta_heatmap.png

Usage
-----
  python -m result_analysis.rq2_firm_identity
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

try:
    from result_analysis.fr_directionality_overall import build_fr_directionality_by_variant
    from result_analysis.framing_context_interaction_ci import (
        build_condition_level_deltas_by_variant,
        compute_framing_context_interaction_ci,
    )
    from result_analysis.model_behavioral_profile import load_profile_data, valid_strategies
    from result_analysis.specific_generic_delta_ci import (
        compute_specific_generic_delta_ci,
        plot_specific_generic_delta_bars,
    )
except ImportError:
    from fr_directionality_overall import build_fr_directionality_by_variant
    from framing_context_interaction_ci import (
        build_condition_level_deltas_by_variant,
        compute_framing_context_interaction_ci,
    )
    from model_behavioral_profile import load_profile_data, valid_strategies
    from specific_generic_delta_ci import (
        compute_specific_generic_delta_ci,
        plot_specific_generic_delta_bars,
    )

SUMMARY_DIR = "./final_results/summary"
PLOTS_DIR = "./final_results/plots"
EXCLUDED = "competitive_dynamics"


def _spearman_between_variants(by_variant: pd.DataFrame) -> pd.DataFrame:
    mat = by_variant.pivot(
        index="context_variant",
        columns="Strategy",
        values="mean_delta_specific_minus_generic",
    )
    mat = mat.reindex(columns=[s for s in valid_strategies if s in mat.columns])
    rows = []
    variants = list(mat.index)
    for i, a in enumerate(variants):
        for b in variants[i + 1:]:
            rho, _ = spearmanr(mat.loc[a].to_numpy(), mat.loc[b].to_numpy())
            rows.append({"variant_a": a, "variant_b": b, "spearman_rho": float(rho)})
    return pd.DataFrame(rows)


def _plot_heatmap(by_variant: pd.DataFrame, out_path: str) -> None:
    order = ["base", "opp_focus", "count_fact", "randomized_numbers"]
    mat = by_variant.pivot(
        index="context_variant",
        columns="Strategy",
        values="mean_delta_specific_minus_generic",
    )
    mat = mat.reindex(
        index=[v for v in order if v in mat.index],
        columns=[s for s in valid_strategies if s in mat.columns],
    )
    vals = mat.to_numpy(dtype=float)
    vmax = float(np.nanmax(np.abs(vals))) if np.isfinite(vals).any() else 1.0

    fig, ax = plt.subplots(figsize=(10, 3.6))
    im = ax.imshow(vals, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(np.arange(mat.shape[1]))
    ax.set_xticklabels(mat.columns.tolist(), rotation=25, ha="right", fontsize=9)
    ax.set_yticks(np.arange(mat.shape[0]))
    ax.set_yticklabels(mat.index.tolist(), fontsize=9)
    ax.set_xlabel("Strategy option")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = vals[i, j]
            if np.isfinite(v):
                ax.text(
                    j, i, f"{v:+.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(v) / vmax > 0.55 else "#222",
                )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r"$\Delta p$ (Specific $-$ Generic)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    os.makedirs(SUMMARY_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    df = load_profile_data(input_dir="infer_results")
    df = df[df["context_variant"] != EXCLUDED].copy()
    print(f"Rows after dropping {EXCLUDED}: {len(df):,}")

    ci_df = compute_specific_generic_delta_ci(df)
    ci_path = os.path.join(SUMMARY_DIR, "rq2_firm_delta_ci.csv")
    ci_df.to_csv(ci_path, index=False)
    bar_path = plot_specific_generic_delta_bars(
        ci_df, save_dir=PLOTS_DIR, filename="eval_rq2_firm_delta_bars.png"
    )

    by_variant = build_fr_directionality_by_variant(df)
    by_variant = by_variant[by_variant["context_variant"] != EXCLUDED]
    var_path = os.path.join(SUMMARY_DIR, "rq2_firm_delta_by_variant.csv")
    by_variant.to_csv(var_path, index=False)
    heat_path = os.path.join(PLOTS_DIR, "eval_rq2_firm_delta_heatmap.png")
    _plot_heatmap(by_variant, heat_path)

    deltas = build_condition_level_deltas_by_variant(df)
    deltas.pop(EXCLUDED, None)
    interaction = compute_framing_context_interaction_ci(deltas)
    interaction = interaction[interaction["context_variant"] != EXCLUDED]
    inter_path = os.path.join(SUMMARY_DIR, "rq2_firm_interaction_ci.csv")
    interaction.to_csv(inter_path, index=False)

    spearman = _spearman_between_variants(by_variant)
    spear_path = os.path.join(SUMMARY_DIR, "rq2_firm_delta_spearman.csv")
    spearman.to_csv(spear_path, index=False)

    print(f"Saved → {ci_path}")
    print(f"Saved → {bar_path}")
    print(f"Saved → {heat_path}")
    print(f"Saved → {inter_path}")
    print(f"Saved → {spear_path}")
    cols = [
        "Strategy", "mean_delta_specific_minus_generic", "ci_lower", "ci_upper",
        "q_value_fdr", "sig_stars",
    ]
    print(ci_df[cols].to_string(index=False))
    print(spearman.to_string(index=False))
    show = interaction[interaction["sig_stars"] != ""][
        ["context_variant", "Strategy", "interaction_delta", "ci_lower", "ci_upper", "q_value_fdr", "sig_stars"]
    ]
    print(show.to_string(index=False) if len(show) else "no FDR-significant interactions")


if __name__ == "__main__":
    main()
