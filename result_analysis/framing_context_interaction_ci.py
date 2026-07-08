"""
Bootstrap 95% CI, p-values, and FDR for context × framing interaction effects.

Interaction (difference-in-differences vs. base):
    Δp_interaction(variant, strategy) = Δp(variant) − Δp(base)

where each Δp is the condition-macro-average of p(Specific) − p(Generic) within
that context variant (Method A logic from §5.3–5.4).

Outputs
-------
  final_results/summary/framing_context_interaction_ci.csv          (long)
  final_results/summary/framing_context_interaction_matrix.csv    (wide, for Table 5)

Usage
-----
  python -m result_analysis.framing_context_interaction_ci
"""

from __future__ import annotations

import os
from typing import Dict, List

import numpy as np
import pandas as pd

try:
    from result_analysis.fr_directionality_overall import build_fr_directionality_by_variant
    from result_analysis.model_behavioral_profile import (
        CONTEXT_VARIANTS,
        FIXED_CONDITION_ID_COLS,
        load_profile_data,
        valid_strategies,
        _strategy_distribution,
    )
    from result_analysis.rationale_analysis import _bh_fdr
except ImportError:
    from fr_directionality_overall import build_fr_directionality_by_variant
    from model_behavioral_profile import (
        CONTEXT_VARIANTS,
        FIXED_CONDITION_ID_COLS,
        load_profile_data,
        valid_strategies,
        _strategy_distribution,
    )
    from rationale_analysis import _bh_fdr

SUMMARY_DIR = "./final_results/summary"
BOOTSTRAP_N = 10_000
BOOTSTRAP_SEED = 42
BASE_VARIANT = "base"
INTERACTION_VARIANTS = [v for v in CONTEXT_VARIANTS if v != BASE_VARIANT]


def build_condition_level_deltas_by_variant(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Return {context_variant: (n_conditions, n_strategies)} Δp matrices."""
    cond_cols = ["Model", "Temperature", *FIXED_CONDITION_ID_COLS, "context_variant"]
    buckets: Dict[str, List[np.ndarray]] = {v: [] for v in CONTEXT_VARIANTS}

    for _, g in df.groupby(cond_cols, dropna=False):
        variant = g["context_variant"].iloc[0]
        if variant not in buckets:
            continue
        gg = g[g["problem_type"] == "generic"]["Standard Mapping"]
        ss = g[g["problem_type"] == "specific"]["Standard Mapping"]
        if len(gg) == 0 or len(ss) == 0:
            continue
        pg = _strategy_distribution(gg)
        ps = _strategy_distribution(ss)
        buckets[variant].append(ps - pg)

    out: Dict[str, np.ndarray] = {}
    for variant, rows in buckets.items():
        if rows:
            out[variant] = np.vstack(rows)
    return out


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


def compute_framing_context_interaction_ci(
    deltas_by_variant: Dict[str, np.ndarray],
    *,
    n_boot: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """Bootstrap interaction Δp(variant) − Δp(base) per strategy."""
    if BASE_VARIANT not in deltas_by_variant:
        raise ValueError(f"Missing base variant in deltas: {BASE_VARIANT}")

    D_base = deltas_by_variant[BASE_VARIANT]
    n_base = D_base.shape[0]
    rng = np.random.default_rng(seed)
    rows: List[dict] = []

    for variant in INTERACTION_VARIANTS:
        if variant not in deltas_by_variant:
            continue
        D_var = deltas_by_variant[variant]
        n_var = D_var.shape[0]

        observed = np.nanmean(D_var, axis=0) - np.nanmean(D_base, axis=0)
        boot_inter = np.empty((n_boot, len(valid_strategies)), dtype=float)

        for b in range(n_boot):
            idx_b = rng.integers(0, n_base, n_base)
            idx_v = rng.integers(0, n_var, n_var)
            boot_inter[b] = np.nanmean(D_var[idx_v], axis=0) - np.nanmean(D_base[idx_b], axis=0)

        ci_lower = np.percentile(boot_inter, 2.5, axis=0)
        ci_upper = np.percentile(boot_inter, 97.5, axis=0)
        p_values = np.array([
            _bootstrap_pvalue_two_sided(observed[j], boot_inter[:, j])
            for j in range(len(valid_strategies))
        ])

        for j, strat in enumerate(valid_strategies):
            rows.append({
                "context_variant": variant,
                "Strategy": strat,
                "interaction_delta": float(observed[j]),
                "ci_lower": float(ci_lower[j]),
                "ci_upper": float(ci_upper[j]),
                "p_value": float(p_values[j]),
                "n_conditions_variant": int(n_var),
                "n_conditions_base": int(n_base),
            })

    out = pd.DataFrame(rows)
    if len(out) == 0:
        return out

    out["q_value_fdr"] = _bh_fdr(out["p_value"].to_numpy())
    out["sig_stars"] = out["q_value_fdr"].map(_sig_stars)
    out["interaction_95ci"] = out.apply(
        lambda r: f"[{r['ci_lower']:.3f}, {r['ci_upper']:.3f}]",
        axis=1,
    )
    return out


def interaction_long_to_matrix(long_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot interaction effects to context_variant × strategy matrix."""
    mat = long_df.pivot(
        index="context_variant",
        columns="Strategy",
        values="interaction_delta",
    )
    row_order = [v for v in INTERACTION_VARIANTS if v in mat.index]
    col_order = [s for s in valid_strategies if s in mat.columns]
    return mat.reindex(index=row_order, columns=col_order)


def interaction_long_to_ci_matrix(long_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot formatted 95% CI strings for paper table."""
    mat = long_df.pivot(
        index="context_variant",
        columns="Strategy",
        values="interaction_95ci",
    )
    row_order = [v for v in INTERACTION_VARIANTS if v in mat.index]
    col_order = [s for s in valid_strategies if s in mat.columns]
    return mat.reindex(index=row_order, columns=col_order)


def run(input_dir: str = "infer_results") -> pd.DataFrame:
    os.makedirs(SUMMARY_DIR, exist_ok=True)

    print("Loading profile data...")
    df = load_profile_data(input_dir=input_dir)
    print(f"Rows loaded: {len(df):,}")

    deltas_by_variant = build_condition_level_deltas_by_variant(df)
    for v in CONTEXT_VARIANTS:
        if v in deltas_by_variant:
            print(f"  {v}: {deltas_by_variant[v].shape[0]} conditions")

    interaction_df = compute_framing_context_interaction_ci(deltas_by_variant)

    # Cross-check against existing by-variant summary
    ref = build_fr_directionality_by_variant(df)
    if len(ref) > 0 and len(interaction_df) > 0:
        base_ref = ref[ref["context_variant"] == BASE_VARIANT].set_index("Strategy")[
            "mean_delta_specific_minus_generic"
        ]
        checks = []
        for variant in INTERACTION_VARIANTS:
            var_ref = ref[ref["context_variant"] == variant].set_index("Strategy")[
                "mean_delta_specific_minus_generic"
            ]
            expected = var_ref - base_ref
            got = interaction_df[interaction_df["context_variant"] == variant].set_index(
                "Strategy"
            )["interaction_delta"]
            checks.append((expected - got).abs().max())
        max_diff = max(checks) if checks else 0.0
        if max_diff > 1e-9:
            print(f"Warning: interaction point estimate mismatch (max |Δ| = {max_diff:.2e})")

    long_path = os.path.join(SUMMARY_DIR, "framing_context_interaction_ci.csv")
    interaction_df.to_csv(long_path, index=False)
    print(f"Saved → {long_path}")

    matrix_path = os.path.join(SUMMARY_DIR, "framing_context_interaction_matrix.csv")
    interaction_long_to_matrix(interaction_df).to_csv(matrix_path)
    print(f"Saved → {matrix_path}")

    display_cols = [
        "context_variant", "Strategy", "interaction_delta", "interaction_95ci",
        "p_value", "q_value_fdr", "sig_stars",
    ]
    print("\n" + interaction_df[display_cols].to_string(index=False))
    return interaction_df


if __name__ == "__main__":
    run()
