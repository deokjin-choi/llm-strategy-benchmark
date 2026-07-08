"""
result_analysis/fr_directionality_overall.py

Whole-dataset version of the FR directionality bar plot.

The original `model_behavioral_profile.build_fr_directionality_summary` /
`plot_fr_directionality_bars` split the Δp = p(Specific) - p(Generic) effect by
(Model, Temperature). This script instead **pools the entire dataset** (all models,
all temperatures) and reports the directional framing effect per strategy.

Two aggregation methods are provided, and the script visualizes BOTH so their
difference is explicit:

  - Method A — condition-weighted (MACRO-average):  [same methodology as the original plot]
      A "condition" is (Model, Temperature, scenario, Num Context, context_variant).
      For each condition we pool repeats into Specific / Generic distributions and
      compute Δp = p(Specific) - p(Generic). We then average Δp across ALL conditions
      with EQUAL WEIGHT per condition. Small and large cells count the same, so this
      down-weights data-rich cells but is robust to imbalance across conditions.
      It also yields a sign-consistency score (fraction of conditions agreeing with
      the mean sign).

  - Method B — pooled (MICRO-average):
      Ignore condition structure entirely. Pool ALL Specific responses into one
      distribution and ALL Generic responses into another, then Δp = p(Specific) -
      p(Generic). Every individual response counts equally, so conditions with more
      responses dominate. There is no per-condition sign-consistency (no conditions).

  When conditions are balanced the two methods agree; they diverge when response
  counts are uneven across conditions (e.g., unequal repeats / variants / models).

Outputs:
  - final_results/summary/fr_directionality_overall_compare_by_strategy.csv  (A + B merged)
  - final_results/plots/eval_fr_directionality_bars__ALL_methodA.png          (Method A only; NOT the paper Fig. 4, which carries CI/stars from specific_generic_delta_ci.py)
  - final_results/plots/eval_fr_directionality_bars__ALL_compare.png         (A vs B grouped)
  - final_results/summary/fr_directionality_by_context_variant_long.csv      (interaction long)
  - final_results/summary/fr_directionality_by_context_variant_matrix.csv   (heatmap matrix)
  - final_results/plots/eval_fr_directionality_heatmap_by_context_variant__ALL.png

Run:
  python -m result_analysis.fr_directionality_overall
  (or, from inside result_analysis/:  python fr_directionality_overall.py)
"""

from __future__ import annotations

import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from result_analysis.model_behavioral_profile import (
        CONTEXT_VARIANTS,
        FIXED_CONDITION_ID_COLS,
        load_profile_data,
        valid_strategies,
        _strategy_distribution,
    )
except ImportError:
    from model_behavioral_profile import (
        CONTEXT_VARIANTS,
        FIXED_CONDITION_ID_COLS,
        load_profile_data,
        valid_strategies,
        _strategy_distribution,
    )


def build_fr_directionality_summary_overall(df: pd.DataFrame) -> pd.DataFrame:
    """
    Method A — condition-weighted (MACRO-average).

    Whole-dataset directional framing effect: Δp = p(Specific) - p(Generic) per strategy,
    averaged across ALL conditions with EQUAL WEIGHT per condition (every condition
    counts the same regardless of how many responses it contains).

    A condition is (Model, Temperature, scenario, Num Context, context_variant).

    Returns one row per Strategy:
      - mean_delta_specific_minus_generic
      - consistency_sign : fraction of conditions where sign(Δp) == sign(mean_delta)
      - n_conditions     : number of contributing conditions
    """
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

    if len(per_cond) == 0:
        return pd.DataFrame(
            columns=[
                "Strategy",
                "mean_delta_specific_minus_generic",
                "consistency_sign",
                "n_conditions",
            ]
        )

    D = np.vstack(per_cond)  # shape: (n_conditions, n_strategies)
    mean_delta = np.nanmean(D, axis=0)
    sign_mean = np.sign(mean_delta)
    with np.errstate(invalid="ignore"):
        consistency = np.nanmean(np.sign(D) == sign_mean, axis=0)

    rows: List[Dict[str, object]] = []
    for i, strat in enumerate(valid_strategies):
        rows.append(
            {
                "Strategy": strat,
                "mean_delta_specific_minus_generic": float(mean_delta[i]),
                "consistency_sign": float(consistency[i]),
                "n_conditions": int(D.shape[0]),
            }
        )

    out = pd.DataFrame(rows)
    return out.sort_values(
        "mean_delta_specific_minus_generic", ascending=False
    ).reset_index(drop=True)


def build_fr_directionality_summary_overall_pooled(df: pd.DataFrame) -> pd.DataFrame:
    """
    Method B — pooled (MICRO-average).

    Ignore condition structure: pool ALL Specific responses into one distribution and
    ALL Generic responses into another, then Δp = p(Specific) - p(Generic) per strategy.
    Every individual response counts equally, so high-volume conditions dominate.

    Returns one row per Strategy:
      - mean_delta_specific_minus_generic_pooled
      - p_specific_pooled, p_generic_pooled
      - n_specific, n_generic : total pooled response counts
    """
    gen = df[df["problem_type"] == "generic"]["Standard Mapping"]
    spec = df[df["problem_type"] == "specific"]["Standard Mapping"]

    if len(gen) == 0 or len(spec) == 0:
        return pd.DataFrame(
            columns=[
                "Strategy",
                "mean_delta_specific_minus_generic_pooled",
                "p_specific_pooled",
                "p_generic_pooled",
                "n_specific",
                "n_generic",
            ]
        )

    pg = _strategy_distribution(gen)
    ps = _strategy_distribution(spec)
    delta = ps - pg

    rows: List[Dict[str, object]] = []
    for i, strat in enumerate(valid_strategies):
        rows.append(
            {
                "Strategy": strat,
                "mean_delta_specific_minus_generic_pooled": float(delta[i]),
                "p_specific_pooled": float(ps[i]),
                "p_generic_pooled": float(pg[i]),
                "n_specific": int(len(spec)),
                "n_generic": int(len(gen)),
            }
        )

    out = pd.DataFrame(rows)
    return out.sort_values(
        "mean_delta_specific_minus_generic_pooled", ascending=False
    ).reset_index(drop=True)


def build_fr_directionality_by_variant(df: pd.DataFrame) -> pd.DataFrame:
    """
    Context × brand-framing interaction: Δp = p(Specific) - p(Generic) per strategy,
    stratified by context_variant.

    Uses the same condition-weighted (macro) logic as
    `build_fr_directionality_summary_overall`, but averages Δp only within each
    context_variant instead of pooling all variants together.

    A condition is (Model, Temperature, scenario, Num Context, context_variant).
    For each variant we compute one Δp vector per condition, then macro-average
    across conditions in that variant.

    Returns one row per (context_variant, Strategy):
      - mean_delta_specific_minus_generic
      - consistency_sign : fraction of conditions in that variant agreeing with mean sign
      - n_conditions     : contributing conditions in that variant
    """
    cond_cols = ["Model", "Temperature", *FIXED_CONDITION_ID_COLS, "context_variant"]
    rows: List[Dict[str, object]] = []

    for variant, dv in df.groupby("context_variant", dropna=False):
        per_cond: List[np.ndarray] = []
        for _, g in dv.groupby(cond_cols, dropna=False):
            gg = g[g["problem_type"] == "generic"]["Standard Mapping"]
            ss = g[g["problem_type"] == "specific"]["Standard Mapping"]
            if len(gg) == 0 or len(ss) == 0:
                continue
            pg = _strategy_distribution(gg)
            ps = _strategy_distribution(ss)
            per_cond.append(ps - pg)

        if len(per_cond) == 0:
            continue

        D = np.vstack(per_cond)
        mean_delta = np.nanmean(D, axis=0)
        sign_mean = np.sign(mean_delta)
        with np.errstate(invalid="ignore"):
            consistency = np.nanmean(np.sign(D) == sign_mean, axis=0)

        for i, strat in enumerate(valid_strategies):
            rows.append(
                {
                    "context_variant": variant,
                    "Strategy": strat,
                    "mean_delta_specific_minus_generic": float(mean_delta[i]),
                    "consistency_sign": float(consistency[i]),
                    "n_conditions": int(D.shape[0]),
                }
            )

    out = pd.DataFrame(rows)
    if len(out) == 0:
        return out

    variant_order = {v: i for i, v in enumerate(CONTEXT_VARIANTS)}
    out["__variant_order"] = out["context_variant"].map(variant_order)
    strat_order = {s: i for i, s in enumerate(valid_strategies)}
    out["__strat_order"] = out["Strategy"].map(strat_order)
    return out.sort_values(["__variant_order", "__strat_order"]).drop(
        columns=["__variant_order", "__strat_order"]
    ).reset_index(drop=True)


def build_fr_directionality_variant_matrix(
    fr_by_variant_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Pivot long `build_fr_directionality_by_variant` output to a heatmap matrix:
    rows = context_variant, columns = Strategy.
    """
    if fr_by_variant_df is None or len(fr_by_variant_df) == 0:
        return pd.DataFrame()

    mat = fr_by_variant_df.pivot(
        index="context_variant",
        columns="Strategy",
        values="mean_delta_specific_minus_generic",
    )
    row_order = [v for v in CONTEXT_VARIANTS if v in mat.index]
    col_order = [s for s in valid_strategies if s in mat.columns]
    return mat.reindex(index=row_order, columns=col_order)


def plot_fr_directionality_heatmap_by_variant(
    fr_by_variant_df: pd.DataFrame,
    save_dir: str,
    filename: str = "eval_fr_directionality_heatmap_by_context_variant__ALL.png",
):
    """
    Heatmap of Δp (Specific − Generic) by context_variant × strategy.

    Rows are context variants (base included); columns are strategies in canonical
    order. Diverging colormap centered at zero highlights interaction patterns:
    e.g. whether Technology Leadership gains under opp_focus are amplified by
    brand framing relative to base.
    """
    os.makedirs(save_dir, exist_ok=True)
    mat = build_fr_directionality_variant_matrix(fr_by_variant_df)
    if mat.empty:
        print("No per-variant directionality data to plot.")
        return None

    fig, ax = plt.subplots(figsize=(10, 6))
    vals = mat.to_numpy(dtype=float)
    vmax = float(np.nanmax(np.abs(vals))) if np.isfinite(vals).any() else 1.0
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0

    im = ax.imshow(vals, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(np.arange(mat.shape[1]))
    ax.set_xticklabels(mat.columns.tolist(), rotation=25, ha="right", fontsize=9)
    ax.set_yticks(np.arange(mat.shape[0]))
    ax.set_yticklabels(mat.index.tolist(), fontsize=9)
    ax.set_xlabel("Strategy", fontsize=9)
    ax.set_ylabel("Context Variant", fontsize=9)
    ax.set_title(
        "Brand framing effect by context variant × strategy\n"
        "Δp = p(Specific) − p(Generic), macro-averaged over conditions within each variant",
        fontsize=10,
        pad=10,
    )

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = vals[i, j]
            if np.isfinite(v):
                norm = abs(v) / vmax if vmax > 0 else 0.0
                txt_color = "white" if norm > 0.55 else "#222222"
                ax.text(
                    j,
                    i,
                    f"{v:+.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=txt_color,
                )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Δp (percentage points)", fontsize=9)
    plt.tight_layout()

    out = os.path.join(save_dir, filename)
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def build_fr_directionality_compare(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge Method A (condition-weighted) and Method B (pooled) into one table,
    including their difference so the methodological gap is explicit.
    """
    a = build_fr_directionality_summary_overall(df)
    b = build_fr_directionality_summary_overall_pooled(df)
    if len(a) == 0 or len(b) == 0:
        return pd.DataFrame()

    merged = a.merge(b, on="Strategy", how="outer")
    merged["delta_A_condition_weighted"] = merged["mean_delta_specific_minus_generic"]
    merged["delta_B_pooled"] = merged["mean_delta_specific_minus_generic_pooled"]
    merged["A_minus_B"] = merged["delta_A_condition_weighted"] - merged["delta_B_pooled"]

    keep = [
        "Strategy",
        "delta_A_condition_weighted",
        "delta_B_pooled",
        "A_minus_B",
        "consistency_sign",
        "n_conditions",
        "p_specific_pooled",
        "p_generic_pooled",
        "n_specific",
        "n_generic",
    ]
    keep = [c for c in keep if c in merged.columns]
    return merged[keep].sort_values(
        "delta_A_condition_weighted", ascending=False
    ).reset_index(drop=True)


def plot_fr_directionality_bars_overall(
    fr_dir_df: pd.DataFrame,
    save_dir: str,
    top_k: int = 8,
    filename: str = "eval_fr_directionality_bars__ALL_methodA.png",
):
    """
    Single, paper-style bar plot for Method A (condition-weighted) Δp over the whole
    dataset. Bars are colored by sign; value labels show Δp itself (no dual encoding).
    Sign-consistency is kept in the summary CSV (report it in text/caption instead).
    """
    os.makedirs(save_dir, exist_ok=True)
    if fr_dir_df is None or len(fr_dir_df) == 0:
        print("No directionality data to plot.")
        return None

    # Fixed canonical strategy order (same as the rest of the paper). valid_strategies
    # is defined bottom->top, so plotting it as-is puts Technology Leadership at the
    # top and Maintain at the bottom in the barh. `top_k` is kept for API compat but
    # the figure always shows all archetypes in this fixed order for cross-figure consistency.
    order_idx = {s: i for i, s in enumerate(valid_strategies)}
    gg = fr_dir_df.copy()
    gg = gg[gg["Strategy"].isin(order_idx)].copy()
    gg["__order"] = gg["Strategy"].map(order_idx)
    gg = gg.sort_values("__order").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    vals = gg["mean_delta_specific_minus_generic"].astype(float).to_numpy()
    labels = gg["Strategy"].astype(str).to_numpy()
    y = np.arange(len(labels))
    colors = np.where(vals >= 0, "#2c7fb8", "#d95f0e")
    ax.barh(y, vals, color=colors, alpha=0.85)
    ax.axvline(0, color="#444", linewidth=1.0)

    # Reserve x headroom proportional to the data range so the outside value labels
    # always stay inside the axes (they used to spill past the left spine and collide
    # with the y-axis strategy names for long negative bars).
    span = float(np.nanmax(np.abs(vals))) if len(vals) else 1.0
    if not np.isfinite(span) or span <= 0:
        span = 1.0
    left = min(0.0, float(np.nanmin(vals))) - 0.40 * span
    right = max(0.0, float(np.nanmax(vals))) + 0.32 * span
    ax.set_xlim(left, right)

    # Value labels = Δp itself (number == bar length), so there is no dual encoding.
    # Sign-consistency is intentionally NOT drawn here; it lives in the summary CSV
    # and is better stated in the text/caption (see notes in run docstring).
    gap = 0.015 * span
    for i, v in enumerate(vals):
        ax.text(
            v + (gap if v >= 0 else -gap),
            i,
            f"{v:+.2f}",
            va="center",
            ha="left" if v >= 0 else "right",
            fontsize=8,
            color="#333",
        )

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Δp  (Specific − Generic)")
    ax.set_title("Strategy shift under brand exposure")
    ax.grid(True, axis="x", linestyle="--", alpha=0.3)
    plt.tight_layout()

    out = os.path.join(save_dir, filename)
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_fr_directionality_compare(
    compare_df: pd.DataFrame,
    save_dir: str,
    top_k: int = 8,
    filename: str = "eval_fr_directionality_bars__ALL_compare.png",
):
    """
    Grouped horizontal bars comparing Method A (condition-weighted / macro) vs
    Method B (pooled / micro) Δp per strategy, so the difference between the two
    aggregation schemes is visually explicit.
    """
    os.makedirs(save_dir, exist_ok=True)
    if compare_df is None or len(compare_df) == 0:
        print("No comparison data to plot.")
        return None

    # Fixed canonical strategy order (see plot_fr_directionality_bars_overall).
    order_idx = {s: i for i, s in enumerate(valid_strategies)}
    gg = compare_df.copy()
    gg = gg[gg["Strategy"].isin(order_idx)].copy()
    gg["__order"] = gg["Strategy"].map(order_idx)
    gg = gg.sort_values("__order").reset_index(drop=True)

    labels = gg["Strategy"].astype(str).to_numpy()
    a_vals = gg["delta_A_condition_weighted"].astype(float).to_numpy()
    b_vals = gg["delta_B_pooled"].astype(float).to_numpy()

    y = np.arange(len(labels))
    h = 0.38

    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    ax.barh(y + h / 2, a_vals, height=h, color="#2c7fb8", alpha=0.9,
            label="A: condition-weighted (macro)")
    ax.barh(y - h / 2, b_vals, height=h, color="#7fbf7b", alpha=0.9,
            label="B: pooled (micro)")
    ax.axvline(0, color="#444", linewidth=1.0)

    for i, v in enumerate(a_vals):
        ax.text(v + (0.0015 if v >= 0 else -0.0015), i + h / 2, f"{v:+.2f}",
                va="center", ha="left" if v >= 0 else "right", fontsize=7.5)
    for i, v in enumerate(b_vals):
        ax.text(v + (0.0015 if v >= 0 else -0.0015), i - h / 2, f"{v:+.2f}",
                va="center", ha="left" if v >= 0 else "right", fontsize=7.5)

    n_cond = int(gg["n_conditions"].iloc[0]) if "n_conditions" in gg.columns and len(gg) else 0
    n_spec = int(gg["n_specific"].iloc[0]) if "n_specific" in gg.columns and len(gg) else 0
    n_gen = int(gg["n_generic"].iloc[0]) if "n_generic" in gg.columns and len(gg) else 0

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("mean Δp  (Specific − Generic)")
    ax.set_title(
        "Framing directionality (whole dataset): aggregation method comparison\n"
        f"A = equal weight over {n_cond} conditions  |  "
        f"B = pooled over {n_spec:,} Specific / {n_gen:,} Generic responses"
    )
    ax.grid(True, axis="x", linestyle="--", alpha=0.3)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    plt.tight_layout()

    out = os.path.join(save_dir, filename)
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def run_fr_directionality_overall(
    input_dir: str = "infer_results",
    summary_dir: str = "./final_results/summary",
    plots_dir: str = "./final_results/plots",
    top_k: int = 8,
):
    os.makedirs(summary_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    print("Loading profile data...")
    df = load_profile_data(input_dir=input_dir)
    print(f"Rows loaded: {len(df):,}")

    print("Building whole-dataset FR directionality (Method A: condition-weighted)...")
    fr_dir_df = build_fr_directionality_summary_overall(df)

    print("Building whole-dataset FR directionality (Method B: pooled)...")
    compare_df = build_fr_directionality_compare(df)

    print("Building FR directionality by context_variant (interaction slice)...")
    fr_by_variant_df = build_fr_directionality_by_variant(df)

    csv_path = os.path.join(summary_dir, "fr_directionality_overall_compare_by_strategy.csv")
    compare_df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    csv_variant_path = os.path.join(
        summary_dir, "fr_directionality_by_context_variant_long.csv"
    )
    fr_by_variant_df.to_csv(csv_variant_path, index=False)
    print(f"Saved: {csv_variant_path}")

    csv_variant_matrix_path = os.path.join(
        summary_dir, "fr_directionality_by_context_variant_matrix.csv"
    )
    build_fr_directionality_variant_matrix(fr_by_variant_df).to_csv(csv_variant_matrix_path)
    print(f"Saved: {csv_variant_matrix_path}")

    print("Plotting Method A bars...")
    out_a = plot_fr_directionality_bars_overall(fr_dir_df, save_dir=plots_dir, top_k=top_k)
    if out_a:
        print(f"Saved: {out_a}")

    print("Plotting A vs B comparison bars...")
    out_cmp = plot_fr_directionality_compare(compare_df, save_dir=plots_dir, top_k=top_k)
    if out_cmp:
        print(f"Saved: {out_cmp}")

    print("Plotting context_variant × strategy interaction heatmap...")
    out_hm = plot_fr_directionality_heatmap_by_variant(
        fr_by_variant_df, save_dir=plots_dir
    )
    if out_hm:
        print(f"Saved: {out_hm}")

    print("\nPreview: fr_directionality_overall_compare_by_strategy")
    print(compare_df.round(4).to_string(index=False))
    print("\nDone.")


if __name__ == "__main__":
    run_fr_directionality_overall(
        input_dir="infer_results",
        summary_dir="./final_results/summary",
        plots_dir="./final_results/plots",
        top_k=8,
    )
