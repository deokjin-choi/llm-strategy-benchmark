"""
Context sensitivity case deep-dive plots for §5.6.3 Case I (Figs. 9-10).

This module is intentionally separate from ``deep_dive_fr_cr_ds_plots.py``: the
FR/CR/DS-named functions there are legacy scaffolding kept for reference but no
longer used to produce paper figures. Context-sensitivity visualizations (strategy
mix under context perturbations + framing, and mean TVD from *base* by perturbation
variant) live here so figure titles, labels, and defaults are owned by this axis
rather than inherited from the old CR terminology.

Outputs (default paths under final_results/plots):
  eval_deepdive_context_strategy_stacks_framing__{model}__{scenario}.png  (Fig. 9)
  eval_deepdive_context_tvd_by_variant__{model}__{scenario}.png          (Fig. 10)
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from result_analysis.model_behavioral_profile import (
        PERTURBATION_VARIANTS,
        load_profile_data,
        valid_strategies,
        _safe_filename,
        _short_model_name,
        _strategy_distribution,
    )
    from result_analysis.tvd_from_base_ci import total_variation_distance
except ImportError:
    from model_behavioral_profile import (
        PERTURBATION_VARIANTS,
        load_profile_data,
        valid_strategies,
        _safe_filename,
        _short_model_name,
        _strategy_distribution,
    )
    from tvd_from_base_ci import total_variation_distance

TEMPERATURES_DEFAULT: Tuple[float, float] = (0.0, 0.7)

# Full context-variant names (no abbreviation), matching how they are written
# in the paper prose (e.g. *competitive_dynamics*, *count_fact*, *opp_focus*,
# *randomized_numbers*).
CONTEXT_VARIANTS_WITH_BASE: Tuple[str, ...] = ("base", *PERTURBATION_VARIANTS)

CONTEXT_VARIANT_COLORS: Dict[str, str] = {
    "base": "#7f7f7f",
    "competitive_dynamics": "#3182bd",
    "count_fact": "#31a354",
    "opp_focus": "#fd8d3c",
    "randomized_numbers": "#984ea3",
}

STRATEGY_COLORS = plt.cm.tab10(np.linspace(0, 0.9, len(valid_strategies)))


def _same_temp(x: float, t: float, eps: float = 1e-6) -> bool:
    return abs(float(x) - float(t)) < eps


def _safe_tvd(p: np.ndarray, q: np.ndarray) -> float:
    """Total Variation Distance; NaN if either side has zero mass."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    if p.sum() == 0 or q.sum() == 0:
        return np.nan
    return total_variation_distance(p, q)


def _stacked_bars(
    ax: plt.Axes,
    x_labels: Sequence[str],
    prop_rows: np.ndarray,
    title: str,
    y_max: Optional[float] = None,
) -> None:
    """prop_rows: shape (n_groups, n_strategies), each row sums to ~1."""
    n_g, _ = prop_rows.shape
    x = np.arange(n_g)
    bottom = np.zeros(n_g)
    for j, strat in enumerate(valid_strategies):
        h = np.nan_to_num(prop_rows[:, j], nan=0.0)
        ax.bar(x, h, bottom=bottom, label=strat, color=STRATEGY_COLORS[j], width=0.65)
        bottom = bottom + h
    ax.set_xticks(x)
    ax.set_xticklabels(list(x_labels), rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("proportion")
    ax.set_title(title, fontsize=10)
    ax.set_ylim(0.0, 1.0 if y_max is None else y_max)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)


def build_context_tvd_from_base_long(
    df: pd.DataFrame,
    *,
    model: Optional[str] = None,
    scenario: Optional[str] = None,
) -> pd.DataFrame:
    """
    Mean TVD(base, v) for each perturbation variant v (including randomized_numbers),
    mean over (Num Context, problem_type).

    If model is None: cohort-wide mean for each (Model, Temperature, variant).
    If model and scenario set: restrict to that slice (still mean over Num Context, problem_type).
    """
    rows: List[dict] = []
    d = df.copy()
    if model is not None:
        d = d[d["Model"] == model]
    if scenario is not None:
        d = d[d["scenario"] == scenario]

    group_keys = ["Model", "Temperature"]
    if scenario is None:
        group_keys.append("scenario")

    for keys, g in d.groupby(group_keys, dropna=False):
        if scenario is None:
            mod, temp, scen = keys
        else:
            mod, temp = keys
            scen = scenario

        for v in PERTURBATION_VARIANTS:
            tvals = []
            for _, gp in g.groupby(["Num Context", "problem_type"], dropna=False):
                base = gp[gp["context_variant"] == "base"]["Standard Mapping"]
                dv = gp[gp["context_variant"] == v]["Standard Mapping"]
                if len(base) == 0 or len(dv) == 0:
                    continue
                tvals.append(
                    _safe_tvd(_strategy_distribution(base), _strategy_distribution(dv))
                )
            tm = float(np.nanmean(tvals)) if len(tvals) else np.nan
            rows.append(
                {
                    "Model": mod,
                    "Temperature": float(temp),
                    "scenario": scen,
                    "perturbation_variant": v,
                    "mean_tvd_from_base": tm,
                    "n_conditions": len(tvals),
                }
            )

    out = pd.DataFrame(rows)
    if len(out) == 0:
        return out
    return out.sort_values(
        ["Model", "Temperature", "scenario", "perturbation_variant"]
    ).reset_index(drop=True)


def plot_context_strategy_stacks_framing_split(
    df: pd.DataFrame,
    *,
    model: str,
    scenario: str,
    temperatures: Sequence[float] = TEMPERATURES_DEFAULT,
    variants_for_plot: Sequence[str] = CONTEXT_VARIANTS_WITH_BASE,
    save_path: str,
) -> None:
    """Fig. 9: strategy mix across all context perturbations (incl. randomized_numbers)
    and framing types, 2 rows per temperature (generic / specific)."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    n_t = len(temperatures)
    fig, axes = plt.subplots(n_t * 2, 1, figsize=(8.0, 2.8 * n_t * 2), sharex=True)

    row = 0
    for temp in temperatures:
        sub = df[
            (df["Model"] == model)
            & (df["scenario"] == scenario)
            & (df["Temperature"].apply(lambda x: _same_temp(x, temp)))
        ]
        for ptype in ("generic", "specific"):
            ax = axes[row]
            rows_m = []
            for v in variants_for_plot:
                g = sub[(sub["context_variant"] == v) & (sub["problem_type"] == ptype)]
                if len(g) == 0:
                    rows_m.append(np.full(len(valid_strategies), np.nan))
                else:
                    rows_m.append(_strategy_distribution(g["Standard Mapping"]))
            mat = np.vstack(rows_m)
            _stacked_bars(
                ax,
                list(variants_for_plot),
                mat,
                title=f"T={float(temp):g} — {ptype}",
            )
            row += 1

    fig.suptitle(
        f"Context sensitivity case: strategy mix by context variant (Generic vs Specific)\n"
        f"{_short_model_name(model)} — {scenario}",
        fontsize=11,
        y=1.005,
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_context_tvd_by_variant_bars(
    tvd_long: pd.DataFrame,
    *,
    model: str,
    scenario: str,
    temperatures: Sequence[float] = TEMPERATURES_DEFAULT,
    save_path: str,
) -> None:
    """Fig. 10: grouped bars, mean TVD(base, v) for each perturbation variant, per temperature."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(1, len(temperatures), figsize=(5.2 * len(temperatures), 4.0), sharey=True)
    if len(temperatures) == 1:
        axes = np.array([axes])

    for ax, temp in zip(axes, temperatures):
        sub = tvd_long[
            (tvd_long["Model"] == model)
            & (tvd_long["scenario"] == scenario)
            & (tvd_long["Temperature"].apply(lambda x: _same_temp(x, temp)))
        ]
        order = PERTURBATION_VARIANTS
        ys = [
            float(sub[sub["perturbation_variant"] == v]["mean_tvd_from_base"].iloc[0])
            if len(sub[sub["perturbation_variant"] == v])
            else np.nan
            for v in order
        ]
        x = np.arange(len(order))
        colors = [CONTEXT_VARIANT_COLORS.get(v, "#888888") for v in order]
        ax.bar(x, ys, color=colors, width=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(list(order), rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("mean TVD from base")
        ax.set_title(f"T={float(temp):g}")
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)

    fig.suptitle(
        f"Context sensitivity: perturbation strength (mean TVD from base)\n"
        f"{_short_model_name(model)} — {scenario}",
        fontsize=11,
        y=1.05,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_context_sensitivity_case_plots(
    input_dir: str = "infer_results",
    plots_dir: str = "./final_results/plots",
    *,
    context_cell: Tuple[str, str] = ("meta-llama/Meta-Llama-3.1-8B-Instruct", "4_model_x_launch"),
    temperatures: Sequence[float] = TEMPERATURES_DEFAULT,
) -> None:
    """Regenerate Figs. 9-10 (§5.6.3 Case I) for the given (model, scenario) cell."""
    os.makedirs(plots_dir, exist_ok=True)
    df = load_profile_data(input_dir=input_dir)
    m, s = context_cell

    plot_context_strategy_stacks_framing_split(
        df,
        model=m,
        scenario=s,
        temperatures=temperatures,
        save_path=os.path.join(
            plots_dir,
            f"eval_deepdive_context_strategy_stacks_framing__{_safe_filename(_short_model_name(m))}__{_safe_filename(s)}.png",
        ),
    )

    tvd_scen = build_context_tvd_from_base_long(df, model=m, scenario=s)
    plot_context_tvd_by_variant_bars(
        tvd_scen,
        model=m,
        scenario=s,
        temperatures=temperatures,
        save_path=os.path.join(
            plots_dir,
            f"eval_deepdive_context_tvd_by_variant__{_safe_filename(_short_model_name(m))}__{_safe_filename(s)}.png",
        ),
    )
    print("Context sensitivity case plots (Figs. 9-10) written to:", plots_dir)


if __name__ == "__main__":
    run_context_sensitivity_case_plots()
