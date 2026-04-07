"""
Deep-dive plots for Table-3 style (model, scenario) cells: FR, CR, DS.

Design notes (logic / alignment with main axes)
-----------------------------------------------
1) FR — strategy mix under Generic vs Specific
   - Aggregate FR is 1 − mean JSD(Generic, Specific) over (Num Context, context_variant).
   - If you pool *all* context_variant rows into one Generic and one Specific histogram, semantic
     variants (CR manipulations) are mixed with framing; that can still illustrate a *marginal*
     scenario-level shift but confounds FR with CR.
   - Default here: context_variant == "base" only, so the plot isolates **firm-identify framing**
     on the **neutral** narrative (aligned with how readers think about “brand on/off”).

2) CR — JSD(base → v) and strategy mixes
   - Matches `compute_cr` / scenario CR: within (scenario, Num Context, problem_type), JSD between
     base and each semantic variant; we report **means over Num Context × problem_type** for the
     chosen scenario (and optionally cohort-wide means for a compact bar chart).
   - Showing **base + all three semantic variants** (stacked strategy bars) is usually better than
     only the single largest-JSD variant: readers see *which* cue moves mass and whether the pattern
     is asymmetric (e.g. opp_focus vs count_fact).

3) DS — entropy vs Num Context
   - Uses `build_condition_level_ds_diagnostics`: per cell entropy of the repeat-level strategy
     distribution (bits). Repeat order is irrelevant; only the multiset of outcomes matters.
   - Boxplots summarize how that entropy spreads **across** (context_variant, problem_type) cells
     at each Num Context, for the chosen scenario.

Outputs (default paths under final_results/plots and final_results/summary):
  eval_deepdive_fr_framing_stacks__{model}__{scenario}__.png
  eval_deepdive_cr_jsd_by_variant__{model}__{scenario}__.png
  eval_deepdive_cr_strategy_stacks__{model}__{scenario}__.png
  eval_deepdive_ds_entropy_numcontext_box__{model}__{scenario}__.png
  deepdive_cr_jsd_by_variant_long.csv (optional cohort / scenario long table)

Optional (``rationale_audit=True``): FR cell **generic vs specific** rationale keyword tables
and permutation sidecar CSVs via ``rationale_analysis.run_framing_keyword_slice_audit`` (same
``infer_results`` frame as plots; ``context_variant`` aligns with ``fr_context_variants``, default ``base``).
"""

from __future__ import annotations

import importlib
import os
import sys
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from result_analysis.model_behavioral_profile import (
        SEMANTIC_VARIANTS,
        build_condition_level_ds_diagnostics,
        load_profile_data,
        _safe_filename,
        _safe_jsd,
        _short_model_name,
        _strategy_distribution,
    )
except ImportError:
    from model_behavioral_profile import (
        SEMANTIC_VARIANTS,
        build_condition_level_ds_diagnostics,
        load_profile_data,
        _safe_filename,
        _safe_jsd,
        _short_model_name,
        _strategy_distribution,
    )

# Always load project rationale via package path — avoid bare `import rationale_analysis`
# (can resolve to an unrelated PyPI package missing our helpers).
def _load_rationale_submodule():
    try:
        return importlib.import_module("result_analysis.rationale_analysis")
    except ImportError:
        _root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        if _root not in sys.path:
            sys.path.insert(0, _root)
        return importlib.import_module("result_analysis.rationale_analysis")


_ra_mod = _load_rationale_submodule()
valid_strategies = _ra_mod.valid_strategies

TEMPERATURES_DEFAULT: Tuple[float, float] = (0.0, 0.7)
STRATEGY_COLORS = plt.cm.tab10(np.linspace(0, 0.9, len(valid_strategies)))


def _same_temp(x: float, t: float, eps: float = 1e-6) -> bool:
    return abs(float(x) - float(t)) < eps


def _rationale_context_variant(fr_context_variants: Sequence[str]) -> str:
    """Match rationale audit to FR plots: prefer `base` when present."""
    if not fr_context_variants:
        return "base"
    if "base" in fr_context_variants:
        return "base"
    return str(fr_context_variants[0])


def pool_strategy_proportions(
    df: pd.DataFrame,
    *,
    problem_type: str,
    context_variants: Optional[Sequence[str]] = None,
) -> np.ndarray:
    """Empirical p over valid_strategies for rows satisfying filters."""
    g = df[df["problem_type"] == problem_type]
    if context_variants is not None:
        g = g[g["context_variant"].isin(list(context_variants))]
    if len(g) == 0:
        return np.full(len(valid_strategies), np.nan)
    return _strategy_distribution(g["Standard Mapping"])


def build_jsd_base_vs_semantic_long(
    df: pd.DataFrame,
    *,
    model: Optional[str] = None,
    scenario: Optional[str] = None,
) -> pd.DataFrame:
    """
    Mean JSD(base, v) for each semantic variant v, mean over (Num Context, problem_type).

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

        for v in SEMANTIC_VARIANTS:
            jvals = []
            for _, gp in g.groupby(["Num Context", "problem_type"], dropna=False):
                base = gp[gp["context_variant"] == "base"]["Standard Mapping"]
                dv = gp[gp["context_variant"] == v]["Standard Mapping"]
                if len(base) == 0 or len(dv) == 0:
                    continue
                jvals.append(
                    _safe_jsd(_strategy_distribution(base), _strategy_distribution(dv))
                )
            jm = float(np.nanmean(jvals)) if len(jvals) else np.nan
            row = {
                "Model": mod,
                "Temperature": float(temp),
                "scenario": scen,
                "semantic_variant": v,
                "mean_jsd_from_base": jm,
                "n_conditions": len(jvals),
            }
            rows.append(row)

    out = pd.DataFrame(rows)
    if len(out) == 0:
        return out
    return out.sort_values(
        ["Model", "Temperature", "scenario", "semantic_variant"]
    ).reset_index(drop=True)


def _stacked_bars(
    ax: plt.Axes,
    x_labels: Sequence[str],
    prop_rows: np.ndarray,
    title: str,
    y_max: Optional[float] = None,
) -> None:
    """prop_rows: shape (n_groups, n_strategies), each row sums to ~1."""
    n_g, n_s = prop_rows.shape
    x = np.arange(n_g)
    bottom = np.zeros(n_g)
    for j, strat in enumerate(valid_strategies):
        h = prop_rows[:, j]
        h = np.nan_to_num(h, nan=0.0)
        ax.bar(x, h, bottom=bottom, label=strat, color=STRATEGY_COLORS[j], width=0.65)
        bottom = bottom + h
    ax.set_xticks(x)
    ax.set_xticklabels(list(x_labels), rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("proportion")
    ax.set_title(title, fontsize=10)
    ax.set_ylim(0.0, 1.0 if y_max is None else y_max)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)


def plot_fr_framing_strategy_stacks(
    df: pd.DataFrame,
    *,
    model: str,
    scenario: str,
    temperatures: Sequence[float] = TEMPERATURES_DEFAULT,
    context_variants_for_framing: Sequence[str] = ("base",),
    save_path: str,
) -> None:
    """
    Two panels (T=0, T=0.7): stacked bars for Generic vs Specific strategy proportions.
    Default: only `base` context_variant so framing is not confounded by semantic variants.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(1, len(temperatures), figsize=(5.2 * len(temperatures), 4.4), sharey=True)
    if len(temperatures) == 1:
        axes = np.array([axes])

    for ax, temp in zip(axes, temperatures):
        sub = df[
            (df["Model"] == model)
            & (df["scenario"] == scenario)
            & (df["Temperature"].apply(lambda x: _same_temp(x, temp)))
        ]
        p_gen = pool_strategy_proportions(
            sub, problem_type="generic", context_variants=context_variants_for_framing
        )
        p_spec = pool_strategy_proportions(
            sub, problem_type="specific", context_variants=context_variants_for_framing
        )
        mat = np.vstack([p_gen, p_spec])
        _stacked_bars(
            ax,
            ["Generic", "Specific"],
            mat,
            title=f"T={float(temp):g}",
        )

    fig.suptitle(
        f"FR deep-dive: strategy mix (Generic vs Specific)\n"
        f"{_short_model_name(model)} — {scenario}\n"
        f"context_variant in {{{', '.join(context_variants_for_framing)}}}",
        fontsize=11,
        y=1.02,
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_cr_jsd_by_variant_bars(
    jsd_long: pd.DataFrame,
    *,
    model: str,
    scenario: str,
    temperatures: Sequence[float] = TEMPERATURES_DEFAULT,
    save_path: str,
) -> None:
    """Grouped bars: mean JSD(base, v) for each semantic variant, per temperature."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(1, len(temperatures), figsize=(4.2 * len(temperatures), 3.8), sharey=True)
    if len(temperatures) == 1:
        axes = np.array([axes])

    for ax, temp in zip(axes, temperatures):
        sub = jsd_long[
            (jsd_long["Model"] == model)
            & (jsd_long["scenario"] == scenario)
            & (jsd_long["Temperature"].apply(lambda x: _same_temp(x, temp)))
        ]
        order = SEMANTIC_VARIANTS
        ys = [float(sub[sub["semantic_variant"] == v]["mean_jsd_from_base"].iloc[0]) if len(sub[sub["semantic_variant"] == v]) else np.nan for v in order]
        x = np.arange(len(order))
        ax.bar(x, ys, color=["#3182bd", "#31a354", "#fd8d3c"], width=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(order, rotation=25, ha="right", fontsize=8)
        ax.set_ylabel("mean JSD from base")
        ax.set_title(f"T={float(temp):g}")
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)

    fig.suptitle(
        f"CR deep-dive: semantic cue strength (mean JSD from base)\n"
        f"{_short_model_name(model)} — {scenario}",
        fontsize=11,
        y=1.05,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_cr_strategy_mix_by_variant(
    df: pd.DataFrame,
    *,
    model: str,
    scenario: str,
    temperatures: Sequence[float] = TEMPERATURES_DEFAULT,
    variants_for_plot: Sequence[str] = ("base", "competitive_dynamics", "count_fact", "opp_focus"),
    save_path: str,
) -> None:
    """
    Stacked strategy distributions for base + semantic variants (pooled over Num Context
    within each problem_type — we pool generic+specific to show cue effect on overall mix).

    If you need framing held fixed, split into two figures or filter problem_type.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(len(temperatures), 1, figsize=(7.0, 3.8 * len(temperatures)), sharex=True)
    if len(temperatures) == 1:
        axes = np.array([axes])

    labels_short = {
        "base": "base",
        "competitive_dynamics": "comp_dyn",
        "count_fact": "count_fact",
        "opp_focus": "opp_focus",
    }

    for ax, temp in zip(axes, temperatures):
        sub = df[
            (df["Model"] == model)
            & (df["scenario"] == scenario)
            & (df["Temperature"].apply(lambda x: _same_temp(x, temp)))
        ]
        rows = []
        for v in variants_for_plot:
            g = sub[sub["context_variant"] == v]
            if len(g) == 0:
                rows.append(np.full(len(valid_strategies), np.nan))
            else:
                rows.append(_strategy_distribution(g["Standard Mapping"]))
        mat = np.vstack(rows)
        _stacked_bars(
            ax,
            [labels_short.get(v, v) for v in variants_for_plot],
            mat,
            title=f"T={float(temp):g}  (pooled over Num Context; generic+specific combined)",
        )

    fig.suptitle(
        f"CR deep-dive: strategy mix by context_variant\n{_short_model_name(model)} — {scenario}",
        fontsize=11,
        y=1.01,
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_cr_strategy_mix_by_variant_framing_split(
    df: pd.DataFrame,
    *,
    model: str,
    scenario: str,
    temperatures: Sequence[float] = TEMPERATURES_DEFAULT,
    variants_for_plot: Sequence[str] = ("base", "competitive_dynamics", "count_fact", "opp_focus"),
    save_path: str,
) -> None:
    """Same as mix-by-variant but 2 rows per temperature: generic vs specific (wider figure)."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    n_t = len(temperatures)
    fig, axes = plt.subplots(n_t * 2, 1, figsize=(7.2, 2.8 * n_t * 2), sharex=True)
    labels_short = {
        "base": "base",
        "competitive_dynamics": "comp_dyn",
        "count_fact": "count_fact",
        "opp_focus": "opp_focus",
    }

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
                [labels_short.get(v, v) for v in variants_for_plot],
                mat,
                title=f"T={float(temp):g} — {ptype}",
            )
            row += 1

    fig.suptitle(
        f"CR deep-dive: strategy mix by variant (Generic vs Specific)\n"
        f"{_short_model_name(model)} — {scenario}",
        fontsize=11,
        y=1.005,
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_ds_entropy_num_context_boxplots(
    ds_cond_df: pd.DataFrame,
    *,
    model: str,
    scenario: str,
    temperatures: Sequence[float] = TEMPERATURES_DEFAULT,
    save_path: str,
) -> None:
    """Boxplot of repeat-level choice entropy (bits) vs Num Context; one column per T."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    sub_all = ds_cond_df[
        (ds_cond_df["Model"] == model) & (ds_cond_df["scenario"] == scenario)
    ]
    if len(sub_all) == 0:
        return

    num_levels = sorted(sub_all["Num Context"].dropna().unique(), key=lambda x: float(x))
    fig, axes = plt.subplots(1, len(temperatures), figsize=(4.5 * len(temperatures), 4.2), sharey=True)
    if len(temperatures) == 1:
        axes = np.array([axes])

    for ax, temp in zip(axes, temperatures):
        sub = sub_all[sub_all["Temperature"].apply(lambda x: _same_temp(x, temp))]
        data = [
            sub[sub["Num Context"] == k]["top1_entropy_bits"].dropna().to_numpy()
            for k in num_levels
        ]
        ax.boxplot(data, labels=[str(int(k)) if float(k).is_integer() else str(k) for k in num_levels])
        ax.set_xlabel("Num Context")
        ax.set_ylabel("Entropy of strategy choices across repeats [bits]")
        ax.set_title(f"T={float(temp):g}")
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)

    fig.suptitle(
        f"DS deep-dive: entropy vs context count (all variant×framing cells)\n"
        f"{_short_model_name(model)} — {scenario}",
        fontsize=11,
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_deep_dive_fr_cr_ds(
    input_dir: str = "infer_results",
    summary_dir: str = "./final_results/summary",
    plots_dir: str = "./final_results/plots",
    *,
    fr_cell: Tuple[str, str] = ("Qwen/Qwen2.5-14B-Instruct", "5_model_3_mass_market"),
    cr_cell: Tuple[str, str] = ("Qwen/Qwen2.5-14B-Instruct", "4_model_x_launch"),
    ds_cell: Tuple[str, str] = ("deepseek-ai/deepseek-llm-7b-chat", "2_roadster_launch"),
    temperatures: Sequence[float] = TEMPERATURES_DEFAULT,
    fr_context_variants: Sequence[str] = ("base",),
    cr_plot_framing_split: bool = True,
    rationale_audit: bool = True,
    rationale_n_perm: int = 200,
) -> None:
    """
    Generate FR / CR / DS deep-dive figures for three (model, scenario) pairs.

    Parameters
    ----------
    cr_plot_framing_split
        If True, also save a second CR figure with generic/specific rows per temperature.
    rationale_audit
        If True, run ``run_framing_keyword_slice_audit`` on the FR cell (same model/scenario as
        stacked bars), one audit per temperature, using ``_rationale_context_variant(fr_context_variants)``.
    """
    os.makedirs(summary_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    df = load_profile_data(input_dir=input_dir)
    ds_cond_df = build_condition_level_ds_diagnostics(df)

    # --- FR
    m, s = fr_cell
    plot_fr_framing_strategy_stacks(
        df,
        model=m,
        scenario=s,
        temperatures=temperatures,
        context_variants_for_framing=fr_context_variants,
        save_path=os.path.join(
            plots_dir,
            f"eval_deepdive_fr_framing_stacks__{_safe_filename(_short_model_name(m))}__{_safe_filename(s)}.png",
        ),
    )

    # --- CR: JSD table + plots for cr_cell; also save cohort-long for appendix-style tables
    jsd_long = build_jsd_base_vs_semantic_long(df)
    jsd_path = os.path.join(summary_dir, "deepdive_cr_jsd_by_variant_long.csv")
    jsd_long.to_csv(jsd_path, index=False)

    m, s = cr_cell
    jsd_scen = build_jsd_base_vs_semantic_long(df, model=m, scenario=s)
    plot_cr_jsd_by_variant_bars(
        jsd_scen,
        model=m,
        scenario=s,
        temperatures=temperatures,
        save_path=os.path.join(
            plots_dir,
            f"eval_deepdive_cr_jsd_by_variant__{_safe_filename(_short_model_name(m))}__{_safe_filename(s)}.png",
        ),
    )
    plot_cr_strategy_mix_by_variant(
        df,
        model=m,
        scenario=s,
        temperatures=temperatures,
        save_path=os.path.join(
            plots_dir,
            f"eval_deepdive_cr_strategy_stacks__{_safe_filename(_short_model_name(m))}__{_safe_filename(s)}.png",
        ),
    )
    if cr_plot_framing_split:
        plot_cr_strategy_mix_by_variant_framing_split(
            df,
            model=m,
            scenario=s,
            temperatures=temperatures,
            save_path=os.path.join(
                plots_dir,
                f"eval_deepdive_cr_strategy_stacks_framing__{_safe_filename(_short_model_name(m))}__{_safe_filename(s)}.png",
            ),
        )

    # --- DS
    m, s = ds_cell
    plot_ds_entropy_num_context_boxplots(
        ds_cond_df,
        model=m,
        scenario=s,
        temperatures=temperatures,
        save_path=os.path.join(
            plots_dir,
            f"eval_deepdive_ds_entropy_numcontext_box__{_safe_filename(_short_model_name(m))}__{_safe_filename(s)}.png",
        ),
    )

    print(f"Saved CR JSD long table: {jsd_path}")
    print("Deep-dive plots written to:", plots_dir)

    if rationale_audit:
        run_audit = getattr(_ra_mod, "run_framing_keyword_slice_audit", None)
        if run_audit is None:
            raise ImportError(
                "result_analysis.rationale_analysis has no run_framing_keyword_slice_audit; "
                "pull the latest rationale_analysis.py from this repo."
            )
        m, scen = fr_cell
        cv = _rationale_context_variant(fr_context_variants)
        print(f"\nRationale audit (FR cell): model={m}, scenario={scen}, context_variant={cv}")
        for temp in temperatures:
            run_audit(
                df,
                scenario=scen,
                model=m,
                temperature=float(temp),
                context_variant=cv,
                homogeneous_only=False,
                pair_group_cols=["scenario", "repeat", "Model", "Temperature", "Num Context"],
                ngram_range=(2, 2),
                min_df=1,
                top_k=40,
                n_perm=int(rationale_n_perm),
                summary_dir=summary_dir,
                show_progress=False,
            )
        print("Rationale keyword CSVs written under:", summary_dir)


if __name__ == "__main__":
    run_deep_dive_fr_cr_ds()
