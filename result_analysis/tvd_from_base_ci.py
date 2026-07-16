"""
Bootstrap 95% CI for Total Variation Distance (TVD) from base
(scenario-level context variants vs base).

TVD(p, q) = 0.5 * sum_i |p_i - q_i| is the minimum fraction of strategy mass
that must be reallocated to turn distribution q into p. Unlike JSD (nats,
see jsd_from_base_ci.py), TVD is directly interpretable as a percentage-point
share of reallocated choices, matching the Δp language used for firm-identity
sensitivity elsewhere in the paper.

Companion metrics (entropy, spearman_vs_base) are recomputed alongside TVD
using the same per-scenario strategy-ratio table; jsd_from_base_ci.py is left
untouched.

Outputs
-------
  final_results/summary/tvd_from_base_ci.csv

Usage
-----
  python -m result_analysis.tvd_from_base_ci
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

try:
    from result_analysis.overall_results import entropy, load_overall_ratio
    from result_analysis.jsd_from_base_ci import _dist_from_codes, _load_pooled_strategy_codes
except ImportError:
    from overall_results import entropy, load_overall_ratio
    from jsd_from_base_ci import _dist_from_codes, _load_pooled_strategy_codes

SUMMARY_DIR = "./final_results/summary"
BOOTSTRAP_N = 10_000
BOOTSTRAP_SEED = 42


def total_variation_distance(p: np.ndarray, q: np.ndarray) -> float:
    """TVD(p, q) = 0.5 * sum |p_i - q_i|; both vectors normalized to sum 1."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = p / p.sum()
    q = q / q.sum()
    return 0.5 * float(np.sum(np.abs(p - q)))


def compute_ratio_metrics_tvd(df: pd.DataFrame) -> pd.DataFrame:
    """Per-scenario entropy, TVD from base, and Spearman rank correlation vs base."""
    base = df.loc["base"]
    rows = []
    for scen, row in df.iterrows():
        H = entropy(row.values)
        tvd = total_variation_distance(row.values, base.values)
        rho = pd.Series(row.values).rank().corr(pd.Series(base.values).rank(), method="spearman")
        rows.append({"scenario": scen, "entropy": H, "tvd_from_base": tvd, "spearman_vs_base": rho})
    return pd.DataFrame(rows).set_index("scenario")


def bootstrap_tvd_ci(
    base_codes: np.ndarray,
    variant_codes: np.ndarray,
    *,
    n_boot: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float, float]:
    """Nonparametric bootstrap 95% CI for TVD(base, variant) from row resampling."""
    rng = np.random.default_rng(seed)
    nb, nv = len(base_codes), len(variant_codes)
    tvd_samples = np.empty(n_boot, dtype=float)

    for b in range(n_boot):
        bc = base_codes[rng.integers(0, nb, nb)]
        vc = variant_codes[rng.integers(0, nv, nv)]
        tvd_samples[b] = total_variation_distance(_dist_from_codes(bc), _dist_from_codes(vc))

    return (
        float(np.mean(tvd_samples)),
        float(np.percentile(tvd_samples, 2.5)),
        float(np.percentile(tvd_samples, 97.5)),
    )


def compute_tvd_from_base_ci(input_dir: str = "infer_results") -> pd.DataFrame:
    """Point estimates from pooled ratios; TVD 95% CI from bootstrap resampling."""
    ratio_path = os.path.join(SUMMARY_DIR, "analysis_overall_ratio.csv")
    if not os.path.exists(ratio_path):
        raise FileNotFoundError(
            f"{ratio_path} not found. Run make_summary or run_all_analysis first."
        )

    metrics = compute_ratio_metrics_tvd(load_overall_ratio(ratio_path)).reset_index()
    pooled, _ = _load_pooled_strategy_codes(input_dir)
    base_codes = pooled.loc[pooled["scenario_type"] == "base", "code"].to_numpy()

    ci_rows = []
    for scenario in metrics["scenario"]:
        if scenario == "base":
            ci_rows.append({"scenario": scenario, "tvd_ci_lower": 0.0, "tvd_ci_upper": 0.0})
            continue
        var_codes = pooled.loc[pooled["scenario_type"] == scenario, "code"].to_numpy()
        _, lo, hi = bootstrap_tvd_ci(base_codes, var_codes)
        ci_rows.append({"scenario": scenario, "tvd_ci_lower": lo, "tvd_ci_upper": hi})

    out = metrics.merge(pd.DataFrame(ci_rows), on="scenario")
    out["tvd_95ci"] = out.apply(
        lambda r: "—" if r["scenario"] == "base"
        else f"[{r['tvd_ci_lower']:.4f}, {r['tvd_ci_upper']:.4f}]",
        axis=1,
    )
    return out


def run(input_dir: str = "infer_results") -> pd.DataFrame:
    os.makedirs(SUMMARY_DIR, exist_ok=True)
    table = compute_tvd_from_base_ci(input_dir)
    out_path = os.path.join(SUMMARY_DIR, "tvd_from_base_ci.csv")
    table.to_csv(out_path, index=False)
    print(f"Saved → {out_path}")
    print(table[["scenario", "entropy", "tvd_from_base", "tvd_95ci", "spearman_vs_base"]].to_string(index=False))
    return table


if __name__ == "__main__":
    run()
