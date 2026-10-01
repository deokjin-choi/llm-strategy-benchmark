"""
RQ3 (JIIS revision): rationale divergence under matched choice.

Same estimators as the initial version, with competitive_dynamics removed.
Log-odds is recomputed. RDS distances are the saved embeddings, reattached to
the same pairs and then filtered, so the embedding model is not rerun.
Existing scripts and their output files are left unchanged.

Outputs
-------
  final_results/summary/rq3_logodds_global.csv
  final_results/summary/rq3_logodds_keywords.csv
  final_results/summary/rq3_rds_calibration.csv
  final_results/summary/rq3_rds_overall_ci.csv
  final_results/summary/rq3_rds_by_strategy_ci.csv
  final_results/summary/rq3_rds_by_variant_strategy.csv
  final_results/plots/eval_rq3_rds_histogram.png
  final_results/plots/eval_rq3_rds_strategy_boxplot.png

Usage
-----
  python -m result_analysis.rq3_rationale
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from result_analysis.rationale_analysis import (
        load_rationale_data,
        permutation_test_discriminative_keywords,
        run_tesla_generic_audit,
    )
    from result_analysis.rationale_rds_analysis import (
        _build_ceiling_pair_df,
        _build_noise_pair_df,
        _build_rds_heatmap_df,
        _plot_rds_strategy_boxplot,
        _prepare_df,
    )
    from result_analysis.rds_by_strategy_ci import (
        build_cell_level_rds,
        compute_rds_by_strategy_ci,
        compute_rds_overall_ci,
        load_rds_pairs_with_distances,
    )
except ImportError:
    from rationale_analysis import (
        load_rationale_data,
        permutation_test_discriminative_keywords,
        run_tesla_generic_audit,
    )
    from rationale_rds_analysis import (
        _build_ceiling_pair_df,
        _build_noise_pair_df,
        _build_rds_heatmap_df,
        _plot_rds_strategy_boxplot,
        _prepare_df,
    )
    from rds_by_strategy_ci import (
        build_cell_level_rds,
        compute_rds_by_strategy_ci,
        compute_rds_overall_ci,
        load_rds_pairs_with_distances,
    )

SUMMARY_DIR = "./final_results/summary"
PLOTS_DIR = "./final_results/plots"
EXCLUDED = "competitive_dynamics"
TABLE_WORDS = [
    "mission lead", "identity leader", "goal technological", "position platform",
    "world transition", "leader capturing", "prestige demonstrate", "mission showcase",
    "trust balanced", "goals standards", "enables integration", "optimization scale",
    "rushed quality", "delaying significant", "funding invest", "make feasible",
]


def _attach(pairs: pd.DataFrame, distances: np.ndarray, col: str) -> pd.DataFrame:
    if len(pairs) != len(distances):
        raise ValueError(f"{col}: pairs={len(pairs):,}, saved={len(distances):,}")
    out = pairs.copy()
    out[col] = distances
    return out


def _plot_histogram(rds, noise, ceiling, summary: pd.DataFrame, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.2))
    x_max = max(0.50, float(np.max(rds)), float(np.max(noise)), float(np.max(ceiling)))
    bins = np.linspace(0, x_max, 50)
    series = [
        ("noise", noise, "#95a5a6", "Repeat noise"),
        ("rds", rds, "#e74c3c", "Firm identity (RDS)"),
        ("ceiling", ceiling, "#2980b9", "Cross-strategy ceiling"),
    ]
    for z, (name, data, color, label) in enumerate(series, start=1):
        med = float(summary.loc[summary.distribution == name, "median"].iloc[0])
        ax.hist(
            data, bins=bins, weights=np.full(len(data), 100.0 / len(data)),
            color=color, edgecolor="white", linewidth=0.3, alpha=0.55,
            label=f"{label} (median={med:.3f})", zorder=z,
        )
        ax.axvline(med, color=color, linestyle="--", linewidth=1.1, zorder=z + 3)
    ax.set_xlim(0, x_max)
    ax.set_xlabel("Cosine distance between rationale embeddings")
    ax.set_ylabel("Percent of pairs (%)")
    ax.legend(fontsize=8.5, frameon=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _summarize(dist: pd.Series, name: str) -> dict:
    x = dist.astype(float)
    return {
        "distribution": name,
        "n_pairs": int(len(x)),
        "mean": float(x.mean()),
        "median": float(x.median()),
        "std": float(x.std()),
    }


def main() -> None:
    os.makedirs(SUMMARY_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    print("Loading rationales...")
    raw = load_rationale_data("infer_results")
    df = raw[raw["context_variant"] != EXCLUDED].copy()
    print(f"Rows after dropping {EXCLUDED}: {len(df):,}")

    print("Log-odds audit...")
    audit = run_tesla_generic_audit(
        df, ngram_range=(2, 2), min_df=2, top_k=30,
        homogeneous_only=True, mask_brand=True,
    )
    perm = permutation_test_discriminative_keywords(
        audit["df_used"], ngram_range=(2, 2), min_df=2, n_perm=300,
        top_k=20, random_state=42, mask_brand=True, show_progress=True,
    )
    global_row = pd.DataFrame([{
        "n_pairs": perm["n_pairs"],
        "n_vocab": perm["n_vocab"],
        "obs_global_mean_abs_delta": perm["obs_global_mean_abs_delta"],
        "perm_global_mean_abs_delta_mean": perm["perm_global_mean_abs_delta_mean"],
        "perm_global_p_value": perm["perm_global_p_value"],
    }])
    global_row.to_csv(os.path.join(SUMMARY_DIR, "rq3_logodds_global.csv"), index=False)
    keywords = perm["keyword_stats"]
    keywords.to_csv(os.path.join(SUMMARY_DIR, "rq3_logodds_keywords.csv"), index=False)
    tops = pd.concat([
        audit["top_specific"].assign(side="specific"),
        audit["top_generic"].assign(side="generic"),
    ], ignore_index=True)
    tops.to_csv(os.path.join(SUMMARY_DIR, "rq3_logodds_top.csv"), index=False)
    present = set(tops.iloc[:, 0].astype(str))
    if "keyword" in tops.columns:
        present = set(tops["keyword"].astype(str))
    print("Table words still listed:", sorted(w for w in TABLE_WORDS if w in present))
    print("Table words missing:", sorted(w for w in TABLE_WORDS if w not in present))
    print(global_row.to_string(index=False))

    print("RDS from saved distances...")
    pairs = load_rds_pairs_with_distances("infer_results")
    pairs = pairs[pairs["context_variant"] != EXCLUDED].copy()
    prepared = _prepare_df(raw)
    saved = np.load(os.path.join(SUMMARY_DIR, "rationale_rds_calibration_distances.npz"))
    noise = _attach(_build_noise_pair_df(prepared), saved["noise"], "distance")
    ceiling = _attach(_build_ceiling_pair_df(prepared), saved["ceiling"], "distance")
    noise = noise[noise["context_variant"] != EXCLUDED]
    ceiling = ceiling[ceiling["context_variant"] != EXCLUDED]

    cal = pd.DataFrame([
        _summarize(pairs["rds"], "rds"),
        _summarize(noise["distance"], "noise"),
        _summarize(ceiling["distance"], "ceiling"),
    ])
    cal.to_csv(os.path.join(SUMMARY_DIR, "rq3_rds_calibration.csv"), index=False)

    cells = build_cell_level_rds(pairs)
    overall = compute_rds_overall_ci(cells)
    overall["median_rds_pairs"] = float(pairs["rds"].median())
    overall["mean_rds_pairs"] = float(pairs["rds"].mean())
    overall["n_pairs"] = int(len(pairs))
    overall.to_csv(os.path.join(SUMMARY_DIR, "rq3_rds_overall_ci.csv"), index=False)
    by_strat = compute_rds_by_strategy_ci(cells)
    by_strat.to_csv(os.path.join(SUMMARY_DIR, "rq3_rds_by_strategy_ci.csv"), index=False)

    heat = _build_rds_heatmap_df(pairs)
    long = (
        pairs.groupby(["context_variant", "strategy"])["rds"]
        .mean()
        .reset_index()
    )
    long.to_csv(os.path.join(SUMMARY_DIR, "rq3_rds_by_variant_strategy.csv"), index=False)
    box_path = os.path.join(PLOTS_DIR, "eval_rq3_rds_strategy_boxplot.png")
    _plot_rds_strategy_boxplot(heat, box_path)
    hist_path = os.path.join(PLOTS_DIR, "eval_rq3_rds_histogram.png")
    _plot_histogram(pairs["rds"].to_numpy(), noise["distance"].to_numpy(), ceiling["distance"].to_numpy(), cal, hist_path)

    print(cal.to_string(index=False))
    print(overall.to_string(index=False))
    print(by_strat.to_string(index=False))
    for strat, g in long.groupby("strategy"):
        span = float(g["rds"].max() - g["rds"].min())
        print(f"{strat}: span={span:.3f} min={g['rds'].min():.3f} max={g['rds'].max():.3f}")
    print(f"Saved plots → {hist_path}, {box_path}")


if __name__ == "__main__":
    main()
