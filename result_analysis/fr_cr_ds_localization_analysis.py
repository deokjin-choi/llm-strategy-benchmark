"""
result_analysis/fr_cr_ds_localization_analysis.py

Experimental-grid localization for three axes only (no cohort "scenario hotspot" narrative):

  (1) Decision stability (DS): condition-level DS_condition, toggle/entropy diagnostics,
      heatmaps (variant × framing) and failure-mode scatters.

  (2) Framing directionality (FR): per-strategy Δp = p(Specific) − p(Generic), with
      sign-consistency across conditions; bar plots.

  (3) Context responsiveness by scenario (CR): mean JSD(Base, semantic variants)
      within each historical scenario; CSV + bar charts + cross-scenario spread summary.

Outputs (default paths under final_results/):
  - summary/model_profile_ds_condition_diagnostics.csv
  - summary/model_profile_fr_directionality_by_strategy.csv
  - summary/model_profile_cr_by_scenario.csv
  - summary/model_profile_cr_scenario_heterogeneity.csv
  - plots/eval_ds_hotspots_heatmap__*.png, eval_ds_failure_modes_scatter__*.png
  - plots/eval_fr_directionality_bars__*.png
  - plots/eval_cr_by_scenario_bars__*.png
"""

from __future__ import annotations

import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from result_analysis.model_behavioral_profile import (
        SEMANTIC_VARIANTS,
        build_condition_level_ds_diagnostics,
        build_fr_directionality_summary,
        plot_ds_condition_heatmaps,
        plot_ds_failure_mode_scatter,
        plot_fr_directionality_bars,
        _safe_filename,
        _safe_jsd,
        _short_model_name,
        _strategy_distribution,
        load_profile_data,
    )
except ImportError:
    from model_behavioral_profile import (
        SEMANTIC_VARIANTS,
        build_condition_level_ds_diagnostics,
        build_fr_directionality_summary,
        plot_ds_condition_heatmaps,
        plot_ds_failure_mode_scatter,
        plot_fr_directionality_bars,
        _safe_filename,
        _safe_jsd,
        _short_model_name,
        _strategy_distribution,
        load_profile_data,
    )


def build_cr_scenario_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per (Model, Temperature, scenario): CR_scenario = mean JSD(Base, v) over
    semantic variants v and over (Num Context, problem_type), matching the CR
    block in model_behavioral_profile.build_scenario_axis_table (CR column only).
    """
    rows: List[Dict[str, object]] = []
    mt_s_keys = ["Model", "Temperature", "scenario"]

    for (model, temp, scenario), g in df.groupby(mt_s_keys, dropna=False):
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
        rows.append(
            {
                "Model": model,
                "Temperature": float(temp),
                "scenario": scenario,
                "CR_scenario": CR_s,
            }
        )

    out = pd.DataFrame(rows)
    if len(out) == 0:
        return out
    return out.sort_values(["Model", "Temperature", "scenario"]).reset_index(drop=True)


def build_cr_scenario_heterogeneity(cr_df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per (Model, Temperature): spread of CR_scenario across historical scenarios.
    """
    if cr_df is None or len(cr_df) == 0:
        return pd.DataFrame()

    def _agg(s: pd.Series) -> pd.Series:
        x = s.astype(float)
        return pd.Series(
            {
                "CR_mean_across_scenarios": float(np.nanmean(x)),
                "CR_std_across_scenarios": float(np.nanstd(x, ddof=0)),
                "CR_min_scenario": float(np.nanmin(x)),
                "CR_max_scenario": float(np.nanmax(x)),
                "n_scenarios": int(s.notna().sum()),
            }
        )

    return (
        cr_df.groupby(["Model", "Temperature"], dropna=False)["CR_scenario"]
        .apply(_agg)
        .reset_index()
    )


def plot_cr_by_scenario_bars(cr_df: pd.DataFrame, save_dir: str) -> None:
    """Horizontal bar chart of CR_scenario per scenario for each (Model, Temperature)."""
    os.makedirs(save_dir, exist_ok=True)
    if cr_df is None or len(cr_df) == 0:
        return

    for (model, temp), g in cr_df.groupby(["Model", "Temperature"], dropna=False):
        gg = g.sort_values("CR_scenario", ascending=True).copy()
        fig, ax = plt.subplots(figsize=(8.0, 4.2))
        y = np.arange(len(gg))
        ax.barh(y, gg["CR_scenario"].astype(float), color="#238b45", alpha=0.88)
        ax.set_yticks(y)
        ax.set_yticklabels([str(x) for x in gg["scenario"].tolist()], fontsize=8)
        ax.set_xlabel("CR_scenario  (mean JSD vs semantic variants)")
        ax.set_title(
            f"Context responsiveness by scenario — {_short_model_name(model)} @ T={float(temp):g}"
        )
        ax.set_xlim(0.0, max(0.05, float(gg["CR_scenario"].max()) * 1.08))
        ax.grid(True, axis="x", linestyle="--", alpha=0.3)
        plt.tight_layout()
        out = os.path.join(
            save_dir,
            f"eval_cr_by_scenario_bars__{_safe_filename(_short_model_name(model))}__T{float(temp):g}.png",
        )
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)


def run_fr_cr_ds_localization(
    input_dir: str = "infer_results",
    summary_dir: str = "./final_results/summary",
    plots_dir: str = "./final_results/plots",
    fr_top_k: int = 8,
) -> None:
    os.makedirs(summary_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    print("Loading profile data...")
    df = load_profile_data(input_dir=input_dir)
    print(f"Rows loaded: {len(df):,}")

    # --- (1) DS: condition-level stability and hotspots
    print("Building condition-level DS diagnostics...")
    ds_cond_df = build_condition_level_ds_diagnostics(df)
    ds_path = os.path.join(summary_dir, "model_profile_ds_condition_diagnostics.csv")
    ds_cond_df.to_csv(ds_path, index=False)
    print(f"Saved: {ds_path}")

    print("Plotting DS heatmaps and failure-mode scatters...")
    plot_ds_condition_heatmaps(ds_cond_df, save_dir=plots_dir)
    plot_ds_failure_mode_scatter(ds_cond_df, save_dir=plots_dir)

    # --- (2) FR: Δp and sign-consistency
    print("Building FR directionality (Δp Specific − Generic)...")
    fr_dir_df = build_fr_directionality_summary(df)
    fr_path = os.path.join(summary_dir, "model_profile_fr_directionality_by_strategy.csv")
    fr_dir_df.to_csv(fr_path, index=False)
    print(f"Saved: {fr_path}")

    print("Plotting FR directionality bars...")
    plot_fr_directionality_bars(fr_dir_df, save_dir=plots_dir, top_k=fr_top_k)

    # --- (3) CR: by historical scenario only
    print("Building CR by scenario...")
    cr_df = build_cr_scenario_table(df)
    cr_path = os.path.join(summary_dir, "model_profile_cr_by_scenario.csv")
    cr_df.to_csv(cr_path, index=False)
    print(f"Saved: {cr_path}")

    cr_het = build_cr_scenario_heterogeneity(cr_df)
    cr_het_path = os.path.join(summary_dir, "model_profile_cr_scenario_heterogeneity.csv")
    cr_het.to_csv(cr_het_path, index=False)
    print(f"Saved: {cr_het_path}")

    print("Plotting CR-by-scenario bars...")
    plot_cr_by_scenario_bars(cr_df, save_dir=plots_dir)

    print("\nDone.")
    print("\nPreview: DS diagnostics (head)")
    print(ds_cond_df.head(8).round(4).to_string(index=False))
    print("\nPreview: FR directionality (head)")
    print(fr_dir_df.head(8).round(4).to_string(index=False))
    print("\nPreview: CR by scenario (head)")
    print(cr_df.head(12).round(4).to_string(index=False))
    print("\nPreview: CR heterogeneity across scenarios")
    print(cr_het.round(4).to_string(index=False))


if __name__ == "__main__":
    run_fr_cr_ds_localization(
        input_dir="infer_results",
        summary_dir="./final_results/summary",
        plots_dir="./final_results/plots",
        fr_top_k=8,
    )
