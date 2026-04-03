"""
result_analysis/model_behavioral_profile.py

Independent analysis script for model-level behavioral profiling.

It computes five metrics per (Model, Temperature) (higher is better on each axis):
  1) Framing Robustness (FR) = 1 - E[JSD(Generic, Specific)]
  2) Context Responsiveness (CR) = E[JSD(Base, v)] for v in semantic variants
  3) Numerical Sensitivity (NS) = E[JSD(Base, Randomized)]
  4) Decision Stability (DS) = 1 - E_r[JSD(P_r, centroid)]
  5) Explanatory Framing Invariance (EFI) = 1 / (1 + EFD_raw)

Also stores EFD_raw (mean |Δ log-odds|) and permutation / FDR side statistics for rationale analysis.

Outputs:
  - final_results/summary/model_profile_by_model_temp.csv
  - final_results/summary/model_profile_temperature_delta.csv
  - final_results/plots/eval_model_profile_radar.png
"""

import glob
import math
import os
import re
from typing import Dict, List, Tuple, Optional

from matplotlib.lines import Line2D

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from result_analysis.rationale_analysis import (
        permutation_test_discriminative_keywords,
        valid_strategies,
    )
except ImportError:
    # fallback for direct script execution from result_analysis/
    from rationale_analysis import permutation_test_discriminative_keywords, valid_strategies


# Context variants expected from infer_results/scenarios_*.csv
CONTEXT_VARIANTS = [
    "base",
    "competitive_dynamics",
    "count_fact",
    "opp_focus",
    "randomized_numbers",
]

SEMANTIC_VARIANTS = ["competitive_dynamics", "count_fact", "opp_focus"]
# In this file, repeat is treated as stochastic replicate.
# Fixed-condition keys intentionally exclude `Chosen Option` because it can move
# together with `Standard Mapping` (label leakage risk for stability/profiling).
FIXED_CONDITION_ID_COLS = ["scenario", "Num Context"]


def _parse_variant_from_filename(path: str) -> str:
    base = os.path.basename(path)
    name = base.replace(".csv", "")
    # scenarios_base / scenarios_competitive_dynamics / ...
    if name.startswith("scenarios_"):
        return name.replace("scenarios_", "")
    return name


def load_profile_data(input_dir: str = "infer_results") -> pd.DataFrame:
    paths = sorted(glob.glob(os.path.join(input_dir, "scenarios_*.csv")))
    if not paths:
        raise FileNotFoundError(f"No scenario CSV files found in: {input_dir}")

    dfs = []
    use_cols = [
        "scenario",
        "problem_type",
        "repeat",
        "Model",
        "Temperature",
        "Num Context",
        "Chosen Option",
        "Standard Mapping",
        "Rationale",
    ]
    for p in paths:
        variant = _parse_variant_from_filename(p)
        d = pd.read_csv(p, usecols=use_cols)
        d["context_variant"] = variant
        dfs.append(d)

    df = pd.concat(dfs, ignore_index=True)
    df["Standard Mapping"] = df["Standard Mapping"].fillna("N/A")
    df["Rationale"] = df["Rationale"].fillna("")
    df = df[df["Standard Mapping"].isin(valid_strategies)].copy()

    # normalize key columns
    df["Temperature"] = pd.to_numeric(df["Temperature"], errors="coerce")
    df["repeat"] = pd.to_numeric(df["repeat"], errors="coerce")
    df["Num Context"] = pd.to_numeric(df["Num Context"], errors="coerce")

    # keep only known variants (safety)
    df = df[df["context_variant"].isin(CONTEXT_VARIANTS)].copy()
    return df


def _strategy_distribution(series: pd.Series) -> np.ndarray:
    counts = series.value_counts(normalize=True).reindex(valid_strategies, fill_value=0.0)
    return counts.values.astype(float)


def _safe_jsd(p: np.ndarray, q: np.ndarray) -> float:
    """
    Jensen-Shannon divergence with log2 base (bounded in [0,1]).
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)

    if p.sum() == 0 or q.sum() == 0:
        return np.nan

    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    eps = 1e-12

    def _kl(a, b):
        idx = a > 0
        return np.sum(a[idx] * np.log2((a[idx] + eps) / (b[idx] + eps)))

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def compute_mean_jsd_framing_generic_specific(df_mt: pd.DataFrame) -> float:
    """
    Raw mean JSD between Generic and Specific strategy distributions
    under the same fixed condition
    (scenario, Num Context, context_variant).
    FR = 1 - this quantity.
    """
    vals = []
    keys = [*FIXED_CONDITION_ID_COLS, "context_variant"]
    for _, g in df_mt.groupby(keys, dropna=False):
        gg = g[g["problem_type"] == "generic"]["Standard Mapping"]
        ss = g[g["problem_type"] == "specific"]["Standard Mapping"]
        if len(gg) == 0 or len(ss) == 0:
            continue
        pg = _strategy_distribution(gg)
        ps = _strategy_distribution(ss)
        vals.append(_safe_jsd(pg, ps))
    return float(np.nanmean(vals)) if vals else np.nan


def compute_fr(df_mt: pd.DataFrame) -> float:
    """Framing Robustness: 1 - mean JSD(Generic, Specific)."""
    j = compute_mean_jsd_framing_generic_specific(df_mt)
    if not np.isfinite(j):
        return np.nan
    return float(1.0 - j)


def compute_cr(df_mt: pd.DataFrame) -> float:
    """
    Context Responsiveness:
    Mean JSD between Base and semantic variants
    (competitive/count_fact/opp_focus), within the same fixed condition
    (scenario, Num Context, problem_type).
    """
    vals = []
    keys = [*FIXED_CONDITION_ID_COLS, "problem_type"]
    for _, g in df_mt.groupby(keys, dropna=False):
        base = g[g["context_variant"] == "base"]["Standard Mapping"]
        if len(base) == 0:
            continue
        p_base = _strategy_distribution(base)
        for v in SEMANTIC_VARIANTS:
            dv = g[g["context_variant"] == v]["Standard Mapping"]
            if len(dv) == 0:
                continue
            p_v = _strategy_distribution(dv)
            vals.append(_safe_jsd(p_base, p_v))
    return float(np.nanmean(vals)) if vals else np.nan


def compute_ns(df_mt: pd.DataFrame) -> float:
    """
    Numerical Sensitivity:
    Mean JSD between Base and randomized_numbers
    within the same fixed condition
    (scenario, Num Context, problem_type).
    Higher = stronger response to numerical perturbation.
    """
    vals = []
    keys = [*FIXED_CONDITION_ID_COLS, "problem_type"]
    for _, g in df_mt.groupby(keys, dropna=False):
        base = g[g["context_variant"] == "base"]["Standard Mapping"]
        rnd = g[g["context_variant"] == "randomized_numbers"]["Standard Mapping"]
        if len(base) == 0 or len(rnd) == 0:
            continue
        p_base = _strategy_distribution(base)
        p_rnd = _strategy_distribution(rnd)
        vals.append(_safe_jsd(p_base, p_rnd))
    return float(np.nanmean(vals)) if vals else np.nan


def compute_ds(df_mt: pd.DataFrame) -> float:
    """
    Decision Stability:
    For each fixed condition (scenario, Num Context, context_variant, problem_type),
    treat each repeat as a single draw (one chosen strategy) and compute stability
    from the empirical choice distribution across repeats.

    Let p be the empirical distribution of `Standard Mapping` over repeats under
    a fixed condition. Define:

        DS_condition = 1 - H(p) / log2(|A|)

    where H is Shannon entropy (bits) and |A| is number of strategies.

    Intuition:
      - DS_condition = 1 if all repeats pick the same strategy (perfectly predictable)
      - DS_condition = 0 if repeats are uniform over strategies (maximally unpredictable)

    Then aggregate DS as the mean of DS_condition over all fixed conditions.
    """
    vals = []
    keys = [*FIXED_CONDITION_ID_COLS, "context_variant", "problem_type"]
    for _, g in df_mt.groupby(keys, dropna=False):
        # With the current logging layout, each repeat is typically one decision (one row).
        # If multiple rows exist for a repeat, they still contribute as samples here.
        series = g["Standard Mapping"]
        if series.shape[0] < 2:
            continue
        p = _strategy_distribution(series)
        eps = 1e-12
        p_safe = p + eps
        p_safe = p_safe / p_safe.sum()
        H = float(-(p_safe * np.log2(p_safe)).sum())
        Hmax = float(np.log2(len(valid_strategies)))
        ds_cond = 1.0 - (H / Hmax if Hmax > 0 else np.nan)
        vals.append(ds_cond)

    return float(np.nanmean(vals)) if vals else np.nan


def compute_efd_raw_and_tests(
    df_mt: pd.DataFrame,
    n_perm: int = 200,
    top_k: int = 20,
    random_state: int = 42,
) -> Tuple[float, float, float]:
    """
    EFD_raw: observed mean |Δ log-odds| over vocabulary (paired Generic vs Specific rationales).
    Also returns permutation p-value and FDR-significant keyword ratio (q<0.05).
    EFI = 1 / (1 + EFD_raw).
    """
    # Pair generic vs specific rationales only when the *strategy outcome* is the same.
    # `Chosen Option` is excluded (can be a proxy for Standard Mapping), so we pair on
    # Standard Mapping directly to avoid label leakage and enforce same-decision comparison.
    group_cols = ["scenario", "repeat", "Model", "Temperature", "Num Context", "Standard Mapping"]
    res = permutation_test_discriminative_keywords(
        df_mt,
        group_cols=group_cols,
        ngram_range=(2, 2),
        min_df=2,
        n_perm=n_perm,
        top_k=top_k,
        random_state=random_state,
        mask_brand=True,
        show_progress=False,
    )
    kw = res["keyword_stats"]
    sig_ratio = float((kw["q_value_fdr"] < 0.05).mean()) if len(kw) > 0 else np.nan
    return (
        float(res["obs_global_mean_abs_delta"]),
        float(res["perm_global_p_value"]),
        sig_ratio,
    )


def build_model_profiles(
    df: pd.DataFrame,
    n_perm_efd: int = 200,
    top_k_efd: int = 20,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Compute five-axis profile per (Model, Temperature).
    Aggregation method:
      - FR/CR/NS/DS: mean over condition-wise statistics (FR, DS use complements of mean JSD)
      - EFD_raw + EFI: paired permutation setup; EFI is a monotone map of EFD_raw
    """
    rows = []
    keys = ["Model", "Temperature"]
    for (model, temp), d in df.groupby(keys, dropna=False):
        d = d.copy()
        fr = compute_fr(d)
        cr = compute_cr(d)
        ns = compute_ns(d)
        ds = compute_ds(d)

        try:
            efd_raw, efd_p, efd_sig_ratio = compute_efd_raw_and_tests(
                d, n_perm=n_perm_efd, top_k=top_k_efd, random_state=random_state
            )
        except Exception:
            efd_raw, efd_p, efd_sig_ratio = np.nan, np.nan, np.nan

        if np.isfinite(efd_raw):
            efi = float(1.0 / (1.0 + efd_raw))
        else:
            efi = np.nan

        rows.append(
            {
                "Model": model,
                "Temperature": float(temp),
                "FR_framing_robustness": fr,
                "CR_context_responsiveness": cr,
                "NS_numerical_sensitivity": ns,
                "DS_decision_stability": ds,
                "EFI_explanatory_framing_invariance": efi,
                "EFD_raw": efd_raw,
                "EFD_perm_p_value": efd_p,
                "EFD_sig_keyword_ratio_q05": efd_sig_ratio,
            }
        )

    out = pd.DataFrame(rows).sort_values(["Model", "Temperature"]).reset_index(drop=True)
    return out


def _minmax_scale(series: pd.Series) -> pd.Series:
    s = series.astype(float)
    mn, mx = np.nanmin(s.values), np.nanmax(s.values)
    if not np.isfinite(mn) or not np.isfinite(mx) or abs(mx - mn) < 1e-12:
        return pd.Series(np.full(len(s), 0.5), index=s.index)
    return (s - mn) / (mx - mn)


def build_profile_key_tables(profile_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build two tables:
      1) raw_table: raw metric values (for interpretation/reporting)
      2) scaled_table: min-max scaled values used in radar (relative comparison)
    """
    metrics = [
        "FR_framing_robustness",
        "CR_context_responsiveness",
        "NS_numerical_sensitivity",
        "DS_decision_stability",
        "EFI_explanatory_framing_invariance",
    ]
    raw_cols = [
        "Model",
        "Temperature",
        *metrics,
        "EFD_raw",
        "EFD_perm_p_value",
        "EFD_sig_keyword_ratio_q05",
    ]
    raw_table = profile_df[raw_cols].copy()

    scaled_table = profile_df[["Model", "Temperature"]].copy()
    for m in metrics:
        scaled_table[m + "_scaled_for_radar"] = _minmax_scale(profile_df[m])

    return raw_table, scaled_table


def _short_model_name(name: str) -> str:
    short = str(name).split("/")[-1]
    short = short.replace("-Instruct", "").replace("-Chat", "")
    return short


def plot_radar_by_temperature(
    profile_df: pd.DataFrame,
    save_path: str,
    facet_by: str = "model",
):
    """
    Radar charts on five behavioral axes (min–max scaled within the table).

    facet_by:
      - "model" (default): one small polar subplot per model; each overlays
        all available temperatures (color + linestyle). If the subplot grid has
        a spare cell (e.g. five models on 3×2), it shows the mean of the same
        min–max scaled scores across all models per temperature ("Mean profile").
      - "temperature": one panel per temperature; all models overlaid (legacy).
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    metrics = [
        "FR_framing_robustness",
        "CR_context_responsiveness",
        "NS_numerical_sensitivity",
        "DS_decision_stability",
        "EFI_explanatory_framing_invariance",
    ]

    d = profile_df.copy()
    d["TempKey"] = d["Temperature"].round(3)
    for m in metrics:
        d[m + "_scaled"] = _minmax_scale(d[m])

    temps = sorted([float(t) for t in d["Temperature"].dropna().unique().tolist()])
    if len(temps) == 0:
        return

    n_axes = len(metrics)
    angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False).tolist()
    angles += angles[:1]

    label_map = {
        "FR_framing_robustness": "FR",
        "CR_context_responsiveness": "CR",
        "NS_numerical_sensitivity": "NS",
        "DS_decision_stability": "DS",
        "EFI_explanatory_framing_invariance": "EFI",
    }
    axis_labels = [label_map[m] for m in metrics]

    temp_linestyles = ["-", "--", "-.", ":"]
    cmap_t = plt.get_cmap("coolwarm")
    temp_colors = [cmap_t(0.15 + 0.7 * i / max(1, len(temps) - 1)) for i in range(len(temps))]

    def _style_polar_ax(ax):
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(axis_labels, fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=7)

    if facet_by == "model":
        model_order = sorted(d["Model"].dropna().unique().tolist())
        n_models = len(model_order)
        ncols = min(3, max(1, n_models))
        nrows = max(1, math.ceil(n_models / ncols))
        fig, axes = plt.subplots(
            nrows,
            ncols,
            subplot_kw={"projection": "polar"},
            figsize=(4.0 * ncols + 0.6, 3.75 * nrows + 1.0),
        )
        if nrows == 1 and ncols == 1:
            axes_flat = [axes]
        else:
            axes_flat = np.atleast_1d(axes).ravel().tolist()

        legend_handles = [
            Line2D(
                [0],
                [0],
                color=temp_colors[ti],
                linestyle=temp_linestyles[ti % len(temp_linestyles)],
                linewidth=2.2,
                label=f"T = {temps[ti]:g}",
            )
            for ti in range(len(temps))
        ]

        for idx, model in enumerate(model_order):
            ax = axes_flat[idx]
            _style_polar_ax(ax)
            ax.set_title(_short_model_name(model), y=1.12, fontsize=11, fontweight="600")

            for ti, t in enumerate(temps):
                tk = round(float(t), 3)
                sub = d[(d["Model"] == model) & (d["TempKey"] == tk)]
                if len(sub) == 0:
                    continue
                r = sub.iloc[0]
                vals = [float(r[m + "_scaled"]) for m in metrics]
                vals += vals[:1]
                c = temp_colors[ti]
                ls = temp_linestyles[ti % len(temp_linestyles)]
                ax.plot(angles, vals, linewidth=2.0, color=c, linestyle=ls)
                ax.fill(angles, vals, color=c, alpha=0.14)

        # Use first spare subplot (e.g. bottom-right for 5 models on a 3×2 grid)
        # as a baseline: per-temperature mean of the same min–max scaled values.
        n_slots = len(axes_flat)
        if n_models < n_slots:
            ax_avg = axes_flat[n_models]
            _style_polar_ax(ax_avg)
            ax_avg.set_title(
                "Mean profile\n(all models)",
                y=1.12,
                fontsize=11,
                fontweight="600",
            )
            for ti, t in enumerate(temps):
                tk = round(float(t), 3)
                block = d[d["TempKey"] == tk]
                if len(block) == 0:
                    continue
                vals = [float(np.nanmean(block[m + "_scaled"].to_numpy())) for m in metrics]
                vals += vals[:1]
                c = temp_colors[ti]
                ls = temp_linestyles[ti % len(temp_linestyles)]
                ax_avg.plot(angles, vals, linewidth=2.0, color=c, linestyle=ls)
                ax_avg.fill(angles, vals, color=c, alpha=0.14)

        for j in range(n_models + (1 if n_models < n_slots else 0), n_slots):
            axes_flat[j].set_visible(False)

        fig.legend(
            handles=legend_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.02),
            ncol=min(4, len(legend_handles)),
            fontsize=9,
            frameon=False,
        )
        fig.suptitle(
            "Behavioral footprints per model (min–max scaled across models and temperatures)",
            y=0.995,
            fontsize=13,
        )
        plt.subplots_adjust(top=0.88, bottom=0.14, wspace=0.32, hspace=0.42)

    elif facet_by == "temperature":
        ncols = len(temps)
        fig, axes = plt.subplots(
            1, ncols, subplot_kw={"projection": "polar"}, figsize=(6.2 * ncols, 6.2)
        )
        if ncols == 1:
            axes = [axes]

        model_order = sorted(d["Model"].dropna().unique().tolist())
        cmap = plt.get_cmap("tab10")
        color_map = {m: cmap(i % 10) for i, m in enumerate(model_order)}

        for ax, t in zip(axes, temps):
            tk = round(float(t), 3)
            dt = d[d["TempKey"] == tk].copy()
            _style_polar_ax(ax)
            ax.set_title(f"Temperature = {t:g}", y=1.06, fontsize=12)

            for _, r in dt.iterrows():
                vals = [float(r[m + "_scaled"]) for m in metrics]
                vals += vals[:1]
                c = color_map[r["Model"]]
                ax.plot(
                    angles,
                    vals,
                    linewidth=2.0,
                    color=c,
                    label=_short_model_name(r["Model"]),
                )
                ax.fill(angles, vals, color=c, alpha=0.06)

        handles, labels = axes[0].get_legend_handles_labels()
        unique = {}
        for h, l in zip(handles, labels):
            if l not in unique:
                unique[l] = h

        fig.legend(
            unique.values(),
            unique.keys(),
            loc="lower center",
            bbox_to_anchor=(0.5, 0.02),
            ncol=min(3, len(unique)),
            fontsize=9,
            frameon=False,
        )
        fig.suptitle(
            "Model Behavioral Profiles Across Five Axes (Relative: Min-Max Scaled)",
            y=0.98,
            fontsize=14,
        )
        plt.subplots_adjust(top=0.82, bottom=0.16, wspace=0.28)
    else:
        raise ValueError(f"facet_by must be 'model' or 'temperature', got {facet_by!r}")

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_temperature_delta_table(profile_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build per-model temperature comparison table:
      metric@T0, metric@T0.7, delta(T0.7-T0)
    """
    metrics = [
        "FR_framing_robustness",
        "CR_context_responsiveness",
        "NS_numerical_sensitivity",
        "DS_decision_stability",
        "EFI_explanatory_framing_invariance",
    ]
    d = profile_df.copy()

    # Use rounded temp keys to avoid floating drift
    d["TempKey"] = d["Temperature"].round(3)

    rows = []
    for model, g in d.groupby("Model", dropna=False):
        row = {"Model": model}
        for m in metrics:
            v0 = g.loc[g["TempKey"] == 0.0, m]
            v7 = g.loc[g["TempKey"] == 0.7, m]
            x0 = float(v0.iloc[0]) if len(v0) > 0 else np.nan
            x7 = float(v7.iloc[0]) if len(v7) > 0 else np.nan
            row[f"{m}_T0"] = x0
            row[f"{m}_T07"] = x7
            row[f"{m}_delta_T07_minus_T0"] = x7 - x0 if np.isfinite(x0) and np.isfinite(x7) else np.nan
        rows.append(row)

    return pd.DataFrame(rows).sort_values("Model").reset_index(drop=True)


def build_condition_level_ds_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Condition-level DS diagnostics to reveal instability hotspots and failure modes.

    One row per
    (Model, Temperature, scenario, Num Context, context_variant, problem_type).
    Columns:
      - DS_condition: 1 - H(p) / log2(|A|), where p is empirical choice distribution across repeats
      - n_repeats: number of repeats available for this condition
      - toggle_rate_top1: fraction of adjacent repeats whose chosen strategy differs
      - top1_entropy_bits: entropy of chosen strategy across repeats (bits)
      - top1_strategy / top2_strategy and their shares across repeats
    """
    keys = [
        "Model",
        "Temperature",
        *FIXED_CONDITION_ID_COLS,
        "context_variant",
        "problem_type",
    ]
    rows: List[Dict[str, object]] = []

    for (model, temp, scenario, num_context, variant, ptype), g in df.groupby(
        keys, dropna=False
    ):
        rep_top1: List[str] = []
        rep_ids: List[float] = []

        for rep, gr in g.groupby("repeat", dropna=False):
            # In current logs, each repeat is typically a single decision.
            # If multiple rows exist, use the mode as the repeat-level label.
            vc_mode = gr["Standard Mapping"].value_counts()
            rep_top1.append(str(vc_mode.index[0]) if len(vc_mode) > 0 else "N/A")
            rep_ids.append(rep)

        n_repeats = len(rep_top1)
        if n_repeats < 2:
            continue

        # Empirical distribution across repeats (repeat-level labels)
        p = _strategy_distribution(pd.Series(rep_top1))
        eps = 1e-12
        p_safe = p + eps
        p_safe = p_safe / p_safe.sum()
        top1_entropy = float(-(p_safe * np.log2(p_safe)).sum())
        Hmax = float(np.log2(len(valid_strategies)))
        ds_cond = 1.0 - (top1_entropy / Hmax if Hmax > 0 else np.nan)

        # toggle rate needs a stable repeat order
        order = np.argsort(np.array(rep_ids, dtype=float))
        top1_seq = [rep_top1[i] for i in order]
        toggle_rate = (
            float(np.mean([top1_seq[i] != top1_seq[i - 1] for i in range(1, len(top1_seq))]))
            if len(top1_seq) >= 2
            else np.nan
        )

        # frequency summary over repeat-level labels (for reporting top-1/top-2 shares)
        vc = pd.Series(top1_seq).value_counts(normalize=True)

        top1_strategy = str(vc.index[0]) if len(vc) > 0 else "N/A"
        top1_share = float(vc.iloc[0]) if len(vc) > 0 else np.nan
        top2_strategy = str(vc.index[1]) if len(vc) > 1 else ""
        top2_share = float(vc.iloc[1]) if len(vc) > 1 else np.nan

        rows.append(
            {
                "Model": model,
                "Temperature": float(temp),
                "scenario": scenario,
                "Num Context": num_context,
                "context_variant": variant,
                "problem_type": ptype,
                "DS_condition": ds_cond,
                "n_repeats": int(n_repeats),
                "toggle_rate_top1": toggle_rate,
                "top1_entropy_bits": top1_entropy,
                "top1_strategy": top1_strategy,
                "top1_share": top1_share,
                "top2_strategy": top2_strategy,
                "top2_share": top2_share,
            }
        )

    out = pd.DataFrame(rows)
    if len(out) == 0:
        return out
    return out.sort_values(["Model", "Temperature", "DS_condition"]).reset_index(drop=True)


def build_fr_directionality_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Directional framing effect: Δp = p(Specific) - p(Generic) per strategy.

    Returns a long table with one row per (Model, Temperature, Strategy),
    containing:
      - mean_delta: average Δp across
        (scenario, Num Context, context_variant)
      - consistency: fraction of conditions where sign(Δp) matches sign(mean_delta)
      - n_conditions: number of conditions contributing
    """
    cond_keys = ["Model", "Temperature", "scenario", "context_variant"]
    deltas: List[Dict[str, object]] = []

    for (model, temp), dmt in df.groupby(["Model", "Temperature"], dropna=False):
        per_cond: List[np.ndarray] = []
        for _, g in dmt.groupby([*FIXED_CONDITION_ID_COLS, "context_variant"], dropna=False):
            gg = g[g["problem_type"] == "generic"]["Standard Mapping"]
            ss = g[g["problem_type"] == "specific"]["Standard Mapping"]
            if len(gg) == 0 or len(ss) == 0:
                continue
            pg = _strategy_distribution(gg)
            ps = _strategy_distribution(ss)
            per_cond.append(ps - pg)

        if len(per_cond) == 0:
            continue

        D = np.vstack(per_cond)  # shape: (n_conditions, n_strategies)
        mean_delta = np.nanmean(D, axis=0)
        sign_mean = np.sign(mean_delta)
        sign_per = np.sign(D)
        with np.errstate(invalid="ignore"):
            consistency = np.nanmean(sign_per == sign_mean, axis=0)

        for i, strat in enumerate(valid_strategies):
            deltas.append(
                {
                    "Model": model,
                    "Temperature": float(temp),
                    "Strategy": strat,
                    "mean_delta_specific_minus_generic": float(mean_delta[i]),
                    "consistency_sign": float(consistency[i]),
                    "n_conditions": int(D.shape[0]),
                }
            )

    out = pd.DataFrame(deltas)
    if len(out) == 0:
        return out
    return out.sort_values(
        ["Model", "Temperature", "Strategy"]
    ).reset_index(drop=True)


def build_temperature_fr_efi_delta_table(profile_df: pd.DataFrame) -> pd.DataFrame:
    """
    Focused temperature asymmetry table for (ΔFR, ΔEFI) at T=0.0 vs T=0.7.
    """
    d = profile_df.copy()
    d["TempKey"] = d["Temperature"].round(3)

    rows = []
    for model, g in d.groupby("Model", dropna=False):
        row = {"Model": model}
        fr0 = g.loc[g["TempKey"] == 0.0, "FR_framing_robustness"]
        fr7 = g.loc[g["TempKey"] == 0.7, "FR_framing_robustness"]
        e0 = g.loc[g["TempKey"] == 0.0, "EFI_explanatory_framing_invariance"]
        e7 = g.loc[g["TempKey"] == 0.7, "EFI_explanatory_framing_invariance"]

        FR_T0 = float(fr0.iloc[0]) if len(fr0) else np.nan
        FR_T07 = float(fr7.iloc[0]) if len(fr7) else np.nan
        EFI_T0 = float(e0.iloc[0]) if len(e0) else np.nan
        EFI_T07 = float(e7.iloc[0]) if len(e7) else np.nan

        row.update(
            {
                "FR_T0": FR_T0,
                "FR_T07": FR_T07,
                "delta_FR_T07_minus_T0": (FR_T07 - FR_T0)
                if np.isfinite(FR_T0) and np.isfinite(FR_T07)
                else np.nan,
                "EFI_T0": EFI_T0,
                "EFI_T07": EFI_T07,
                "delta_EFI_T07_minus_T0": (EFI_T07 - EFI_T0)
                if np.isfinite(EFI_T0) and np.isfinite(EFI_T07)
                else np.nan,
            }
        )
        rows.append(row)

    return pd.DataFrame(rows).sort_values("Model").reset_index(drop=True)


def plot_temp_delta_fr_efi(delta_df: pd.DataFrame, save_path: str):
    """
    Scatter plot of (ΔFR, ΔEFI) per model to visualize asymmetry and trade-offs.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    d = delta_df.copy()
    x = d["delta_FR_T07_minus_T0"].astype(float)
    y = d["delta_EFI_T07_minus_T0"].astype(float)

    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    ax.axhline(0, color="#888", linewidth=1.0)
    ax.axvline(0, color="#888", linewidth=1.0)
    ax.scatter(x, y, s=80, alpha=0.85)
    for _, r in d.iterrows():
        ax.text(
            float(r["delta_FR_T07_minus_T0"]),
            float(r["delta_EFI_T07_minus_T0"]),
            _short_model_name(r["Model"]),
            fontsize=9,
            ha="left",
            va="bottom",
        )
    ax.set_xlabel("ΔFR (T=0.7 − T=0.0)")
    ax.set_ylabel("ΔEFI (T=0.7 − T=0.0)")
    ax.set_title("Temperature effect asymmetry: (ΔFR, ΔEFI) per model")
    ax.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _safe_filename(text: str) -> str:
    s = str(text)
    s = s.replace("/", "_").replace("\\", "_")
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", s).strip("_")
    return s[:160] if len(s) > 160 else s


def plot_ds_condition_heatmaps(ds_cond_df: pd.DataFrame, save_dir: str):
    """
    Experiment (1): Visualize DS hotspots as heatmaps per (Model, Temperature).
    Heatmap cells are mean DS_condition aggregated over scenarios.
    """
    os.makedirs(save_dir, exist_ok=True)
    if ds_cond_df is None or len(ds_cond_df) == 0:
        return

    for (model, temp), g in ds_cond_df.groupby(["Model", "Temperature"], dropna=False):
        pivot = (
            g.groupby(["context_variant", "problem_type"], dropna=False)["DS_condition"]
            .mean()
            .unstack("problem_type")
        )
        # fixed row order for readability
        row_order = [v for v in CONTEXT_VARIANTS if v in pivot.index]
        pivot = pivot.reindex(row_order)

        fig, ax = plt.subplots(figsize=(7.2, 3.6))
        mat = pivot.to_numpy(dtype=float)
        im = ax.imshow(mat, vmin=0.0, vmax=1.0, cmap="viridis", aspect="auto")

        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels(pivot.index.tolist(), fontsize=9)
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns.tolist(), fontsize=9)
        ax.set_title(f"DS hotspots heatmap (mean over scenarios) — {_short_model_name(model)} @ T={float(temp):g}")
        ax.set_xlabel("framing / problem_type")
        ax.set_ylabel("context_variant")

        # annotate values
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8, color="white" if v < 0.45 else "black")

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("DS_condition (higher = more stable)", fontsize=9)
        plt.tight_layout()

        out = os.path.join(save_dir, f"eval_ds_hotspots_heatmap__{_safe_filename(_short_model_name(model))}__T{float(temp):g}.png")
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)


def plot_ds_failure_mode_scatter(ds_cond_df: pd.DataFrame, save_dir: str):
    """
    Experiment (1): Scatter (toggle_rate_top1, top1_entropy_bits) per condition.
    Helps separate 'two-strategy toggling' vs 'diffuse randomness'.
    """
    os.makedirs(save_dir, exist_ok=True)
    if ds_cond_df is None or len(ds_cond_df) == 0:
        return

    for (model, temp), g in ds_cond_df.groupby(["Model", "Temperature"], dropna=False):
        fig, ax = plt.subplots(figsize=(7.0, 5.8))
        x = g["toggle_rate_top1"].astype(float)
        y = g["top1_entropy_bits"].astype(float)
        c = g["DS_condition"].astype(float)
        sc = ax.scatter(x, y, c=c, cmap="viridis", vmin=0.0, vmax=1.0, s=42, alpha=0.85)
        ax.set_xlabel("ToggleRate(top-1) across repeats")
        ax.set_ylabel("Entropy(top-1) across repeats [bits]")
        ax.set_title(f"DS failure modes — {_short_model_name(model)} @ T={float(temp):g}")
        ax.grid(True, linestyle="--", alpha=0.35)
        cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("DS_condition (higher = more stable)", fontsize=9)
        plt.tight_layout()

        out = os.path.join(save_dir, f"eval_ds_failure_modes_scatter__{_safe_filename(_short_model_name(model))}__T{float(temp):g}.png")
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)


def plot_fr_directionality_bars(fr_dir_df: pd.DataFrame, save_dir: str, top_k: int = 8):
    """
    Experiment (2): Bar plot for Δp = p(Specific) - p(Generic), showing largest-magnitude strategies.
    Bars are colored by sign; marker encodes sign-consistency across conditions.
    """
    os.makedirs(save_dir, exist_ok=True)
    if fr_dir_df is None or len(fr_dir_df) == 0:
        return

    for (model, temp), g in fr_dir_df.groupby(["Model", "Temperature"], dropna=False):
        gg = g.copy()
        gg["abs_delta"] = gg["mean_delta_specific_minus_generic"].abs()
        gg = gg.sort_values("abs_delta", ascending=False).head(int(top_k)).iloc[::-1]  # small-to-large for nicer bars

        fig, ax = plt.subplots(figsize=(7.4, 4.4))
        vals = gg["mean_delta_specific_minus_generic"].astype(float).to_numpy()
        labels = gg["Strategy"].astype(str).to_numpy()
        y = np.arange(len(labels))
        colors = np.where(vals >= 0, "#2c7fb8", "#d95f0e")
        ax.barh(y, vals, color=colors, alpha=0.85)
        ax.axvline(0, color="#444", linewidth=1.0)

        # annotate consistency
        cons = gg["consistency_sign"].astype(float).to_numpy()
        for i, (v, cs) in enumerate(zip(vals, cons)):
            ax.text(v + (0.002 if v >= 0 else -0.002), i, f"{cs:.2f}", va="center", ha="left" if v >= 0 else "right", fontsize=8)

        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel("mean Δp  (Specific − Generic)")
        ax.set_title(f"Framing directionality (top |Δp| strategies) — {_short_model_name(model)} @ T={float(temp):g}\nText labels show sign-consistency across conditions")
        ax.grid(True, axis="x", linestyle="--", alpha=0.3)
        plt.tight_layout()

        out = os.path.join(save_dir, f"eval_fr_directionality_bars__{_safe_filename(_short_model_name(model))}__T{float(temp):g}.png")
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)


def plot_scenario_hotspots_bars(hotspots_df: pd.DataFrame, save_dir: str, top_k: int = 10):
    """
    Experiment (4): Bar plot of Top-k scenario deviation magnitude per (Model, Temperature).
    """
    os.makedirs(save_dir, exist_ok=True)
    if hotspots_df is None or len(hotspots_df) == 0:
        return

    for (model, temp), g in hotspots_df.groupby(["Model", "Temperature"], dropna=False):
        gg = g.sort_values("rank_within_model_temp", ascending=True).head(int(top_k)).copy()
        # reverse for barh (rank 1 at top)
        gg = gg.iloc[::-1]
        fig, ax = plt.subplots(figsize=(8.4, 4.6))
        y = np.arange(len(gg))
        ax.barh(y, gg["dev_from_avg_profile_L2"].astype(float), color="#6a51a3", alpha=0.85)
        ax.set_yticks(y)
        ax.set_yticklabels([str(x) for x in gg["scenario"].tolist()], fontsize=8)
        ax.set_xlabel("Deviation from Average Profile (L2, min–max scaled axes)")
        ax.set_title(f"Scenario hotspots vs Average Profile (Top {min(top_k, len(gg))}) — {_short_model_name(model)} @ T={float(temp):g}")
        ax.grid(True, axis="x", linestyle="--", alpha=0.3)
        plt.tight_layout()

        out = os.path.join(save_dir, f"eval_scenario_hotspots_top{min(top_k, len(gg))}__{_safe_filename(_short_model_name(model))}__T{float(temp):g}.png")
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)


def build_scenario_axis_table(
    df: pd.DataFrame,
    ds_cond_table: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Scenario-level axis values to support average-profile deviation hotspots.

    Computes, for each (Model, Temperature, scenario), while keeping fixed-condition
    consistency with (Num Context):
      - FR_scenario: 1 - mean_v JSD(Generic, Specific) for available context variants v
      - CR_scenario: mean_{problem_type, v in SEMANTIC_VARIANTS} JSD(Base, v)
      - DS_scenario: mean_{context_variant, problem_type} DS_condition (if provided; else computed)

    Note: NS is intentionally omitted here (user requested to de-emphasize numeric axis).
    """
    rows: List[Dict[str, object]] = []
    mt_s_keys = ["Model", "Temperature", "scenario"]

    # If DS condition table not provided, build it once.
    ds_tbl = ds_cond_table
    if ds_tbl is None:
        ds_tbl = build_condition_level_ds_diagnostics(df)

    for (model, temp, scenario), g in df.groupby(mt_s_keys, dropna=False):
        # FR_scenario: average over context_variant of JSD(Generic, Specific), then complement.
        jsds = []
        for _, gv in g.groupby(["Num Context", "context_variant"], dropna=False):
            gg = gv[gv["problem_type"] == "generic"]["Standard Mapping"]
            ss = gv[gv["problem_type"] == "specific"]["Standard Mapping"]
            if len(gg) == 0 or len(ss) == 0:
                continue
            jsds.append(_safe_jsd(_strategy_distribution(gg), _strategy_distribution(ss)))
        FR_s = float(1.0 - np.nanmean(jsds)) if len(jsds) else np.nan

        # CR_scenario: within scenario, average over problem_type and semantic variants.
        cr_vals = []
        for _, gp in g.groupby(["Num Context", "problem_type"], dropna=False):
            base = gp[gp["context_variant"] == "base"]["Standard Mapping"]
            if len(base) == 0:
                continue
            p_base = _strategy_distribution(base)
            for v in SEMANTIC_VARIANTS:
                dv = gp[gp["context_variant"] == v]["Standard Mapping"]
                if len(dv) == 0:
                    continue
                cr_vals.append(_safe_jsd(p_base, _strategy_distribution(dv)))
        CR_s = float(np.nanmean(cr_vals)) if len(cr_vals) else np.nan

        # DS_scenario: mean of DS_condition across (context_variant, problem_type) within scenario.
        if ds_tbl is not None and len(ds_tbl) > 0:
            sub = ds_tbl[
                (ds_tbl["Model"] == model)
                & (ds_tbl["Temperature"] == float(temp))
                & (ds_tbl["scenario"] == scenario)
            ]
            DS_s = float(np.nanmean(sub["DS_condition"].to_numpy())) if len(sub) else np.nan
        else:
            DS_s = np.nan

        rows.append(
            {
                "Model": model,
                "Temperature": float(temp),
                "scenario": scenario,
                "FR_scenario": FR_s,
                "CR_scenario": CR_s,
                "DS_scenario": DS_s,
            }
        )

    out = pd.DataFrame(rows)
    if len(out) == 0:
        return out
    return out.sort_values(["Model", "Temperature", "scenario"]).reset_index(drop=True)


def build_average_profile_deviation_hotspots(
    scenario_axis_df: pd.DataFrame,
    top_k: int = 10,
) -> pd.DataFrame:
    """
    Average-profile deviation hotspots.

    For each (Temperature, scenario), compute the mean axis vector across models
    and measure each model's L2 distance from that mean after min–max scaling
    each axis over the whole scenario table.

    Returns Top-k deviating scenarios per (Model, Temperature).
    """
    d = scenario_axis_df.copy()
    axes = ["FR_scenario", "CR_scenario", "DS_scenario"]
    for a in axes:
        d[a + "_scaled"] = _minmax_scale(d[a])

    # mean baseline per (Temperature, scenario)
    base = (
        d.groupby(["Temperature", "scenario"], dropna=False)[[a + "_scaled" for a in axes]]
        .mean()
        .reset_index()
        .rename(columns={a + "_scaled": f"{a}_baseline_scaled" for a in axes})
    )

    d2 = d.merge(base, on=["Temperature", "scenario"], how="left")
    vec = d2[[a + "_scaled" for a in axes]].to_numpy(dtype=float)
    bvec = d2[[f"{a}_baseline_scaled" for a in axes]].to_numpy(dtype=float)
    d2["dev_from_avg_profile_L2"] = np.sqrt(np.nansum((vec - bvec) ** 2, axis=1))

    # rank within (Model, Temperature)
    d2["rank_within_model_temp"] = (
        d2.groupby(["Model", "Temperature"], dropna=False)["dev_from_avg_profile_L2"]
        .rank(ascending=False, method="first")
        .astype(int)
    )
    out = d2[d2["rank_within_model_temp"] <= int(top_k)].copy()
    keep = [
        "Model",
        "Temperature",
        "scenario",
        "dev_from_avg_profile_L2",
        "rank_within_model_temp",
        "FR_scenario",
        "CR_scenario",
        "DS_scenario",
        "FR_scenario_scaled",
        "CR_scenario_scaled",
        "DS_scenario_scaled",
        "FR_scenario_baseline_scaled",
        "CR_scenario_baseline_scaled",
        "DS_scenario_baseline_scaled",
    ]
    return out[keep].sort_values(["Model", "Temperature", "rank_within_model_temp"]).reset_index(drop=True)


def run_model_profile_analysis(
    input_dir: str = "infer_results",
    summary_dir: str = "./final_results/summary",
    plots_dir: str = "./final_results/plots",
    n_perm_efd: int = 200,
    top_k_efd: int = 20,
    random_state: int = 42,
    radar_facet_by: str = "model",
):
    os.makedirs(summary_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    print("Loading profile data...")
    df = load_profile_data(input_dir=input_dir)
    print(f"Rows loaded: {len(df):,}")

    print("Computing five-axis model profiles...")
    profile_df = build_model_profiles(
        df,
        n_perm_efd=n_perm_efd,
        top_k_efd=top_k_efd,
        random_state=random_state,
    )

    profile_path = os.path.join(summary_dir, "model_profile_by_model_temp.csv")
    profile_df.to_csv(profile_path, index=False)
    print(f"Saved: {profile_path}")

    # Raw vs scaled tables for clear interpretation
    raw_table, scaled_table = build_profile_key_tables(profile_df)
    raw_table_path = os.path.join(summary_dir, "model_profile_raw_metrics_table.csv")
    scaled_table_path = os.path.join(summary_dir, "model_profile_scaled_for_radar.csv")
    raw_table.to_csv(raw_table_path, index=False)
    scaled_table.to_csv(scaled_table_path, index=False)
    print(f"Saved: {raw_table_path}")
    print(f"Saved: {scaled_table_path}")

    print("Building temperature comparison table...")
    delta_df = build_temperature_delta_table(profile_df)
    delta_path = os.path.join(summary_dir, "model_profile_temperature_delta.csv")
    delta_df.to_csv(delta_path, index=False)
    print(f"Saved: {delta_path}")

    print("Plotting radar chart...")
    radar_path = os.path.join(plots_dir, "eval_model_profile_radar.png")
    plot_radar_by_temperature(profile_df, radar_path, facet_by=radar_facet_by)
    print(f"Saved: {radar_path}")

    # --- Localization on the experimental grid (DS, FR directionality, CR-by-scenario):
    #     run `python -m result_analysis.fr_cr_ds_localization_analysis` (or import
    #     `run_fr_cr_ds_localization`). Previously this block also lived here.
    #
    # print("Building condition-level DS diagnostics...")
    # ds_cond_df = build_condition_level_ds_diagnostics(df)
    # ds_cond_path = os.path.join(summary_dir, "model_profile_ds_condition_diagnostics.csv")
    # ds_cond_df.to_csv(ds_cond_path, index=False)
    # print(f"Saved: {ds_cond_path}")
    #
    # print("Plotting DS diagnostics (heatmaps + failure-mode scatters)...")
    # plot_ds_condition_heatmaps(ds_cond_df, save_dir=plots_dir)
    # plot_ds_failure_mode_scatter(ds_cond_df, save_dir=plots_dir)
    #
    # print("Building FR directionality summary (Δp Specific − Generic)...")
    # fr_dir_df = build_fr_directionality_summary(df)
    # fr_dir_path = os.path.join(summary_dir, "model_profile_fr_directionality_by_strategy.csv")
    # fr_dir_df.to_csv(fr_dir_path, index=False)
    # print(f"Saved: {fr_dir_path}")
    #
    # print("Plotting FR directionality bars...")
    # plot_fr_directionality_bars(fr_dir_df, save_dir=plots_dir, top_k=8)
    #
    # print("Building temperature asymmetry table (ΔFR, ΔEFI)...")
    # fr_efi_delta_df = build_temperature_fr_efi_delta_table(profile_df)
    # fr_efi_delta_path = os.path.join(summary_dir, "model_profile_temp_delta_fr_efi.csv")
    # fr_efi_delta_df.to_csv(fr_efi_delta_path, index=False)
    # print(f"Saved: {fr_efi_delta_path}")
    #
    # print("Plotting temperature asymmetry scatter (ΔFR vs ΔEFI)...")
    # fr_efi_scatter_path = os.path.join(plots_dir, "eval_temp_delta_fr_vs_efi.png")
    # plot_temp_delta_fr_efi(fr_efi_delta_df, fr_efi_scatter_path)
    # print(f"Saved: {fr_efi_scatter_path}")

    # --- Scenario-level FR/CR/DS table + "average profile deviation" hotspot ranking
    #     (시나리오 전개 / cohort-deviation narrative). Kept as library functions;
    #     not executed in this pipeline. Use fr_cr_ds_localization_analysis for grid
    #     localization, or call build_scenario_axis_table + build_average_profile_deviation_hotspots manually.
    #
    # print("Building scenario-level axis table (FR/CR/DS) and deviation hotspots...")
    # scenario_axis_df = build_scenario_axis_table(df, ds_cond_table=ds_cond_df)
    # scenario_axis_path = os.path.join(summary_dir, "model_profile_scenario_axes_fr_cr_ds.csv")
    # scenario_axis_df.to_csv(scenario_axis_path, index=False)
    # print(f"Saved: {scenario_axis_path}")
    #
    # hotspots_df = build_average_profile_deviation_hotspots(scenario_axis_df, top_k=10)
    # hotspots_path = os.path.join(summary_dir, "model_profile_scenario_hotspots_top10.csv")
    # hotspots_df.to_csv(hotspots_path, index=False)
    # print(f"Saved: {hotspots_path}")
    #
    # print("Plotting scenario hotspot bars...")
    # plot_scenario_hotspots_bars(hotspots_df, save_dir=plots_dir, top_k=10)

    print("\nDone.")
    print("\nPreview: model_profile_by_model_temp")
    print(profile_df.round(4).to_string(index=False))
    print("\nPreview: model_profile_raw_metrics_table")
    print(raw_table.round(4).to_string(index=False))
    print("\nPreview: model_profile_scaled_for_radar")
    print(scaled_table.round(4).to_string(index=False))
    print("\nPreview: model_profile_temperature_delta")
    print(delta_df.round(4).to_string(index=False))


if __name__ == "__main__":
    run_model_profile_analysis(
        input_dir="infer_results",
        summary_dir="./final_results/summary",
        plots_dir="./final_results/plots",
        n_perm_efd=200,  # speed/rigor trade-off
        top_k_efd=20,
        random_state=42,
        radar_facet_by="model",
    )

