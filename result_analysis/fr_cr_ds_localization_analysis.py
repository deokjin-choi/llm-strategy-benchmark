"""
result_analysis/fr_cr_ds_localization_analysis.py

Primary output (current default run):

  **Scenario × Model overview** — one figure per temperature with **three panels**
  (FR_scenario, CR_scenario, DS_scenario). Each panel uses the same definitions as
  `model_behavioral_profile.build_scenario_axis_table`. Values are **min–max scaled
  to [0, 1] within that panel** so the colormap spans the full range per metric.

  Plots: `final_results/plots/eval_scenario_model_overview_FR_CR_DS__T{t}.png`
  Table: `final_results/summary/model_profile_scenario_axes_fr_cr_ds.csv`

Optional: pass `deep_dive=True` to `run_fr_cr_ds_localization` to also run
`deep_dive_fr_cr_ds_plots.run_deep_dive_fr_cr_ds` (Table-3 default cells).

(Legacy pipelines — DS/FR-bars/CR-bars etc. — are kept in the file but commented out
in `run_fr_cr_ds_localization`; uncomment blocks to restore prior outputs.)
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
        build_scenario_axis_table,
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
        build_scenario_axis_table,
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


def _minmax01_2d(a: np.ndarray) -> np.ndarray:
    """Per-panel min–max to [0, 1]; NaN preserved; constant matrix → 0.5 where finite."""
    a = np.asarray(a, dtype=float)
    out = np.full_like(a, np.nan, dtype=float)
    mask = np.isfinite(a)
    if not mask.any():
        return out
    lo = float(np.nanmin(a))
    hi = float(np.nanmax(a))
    if hi <= lo:
        out[mask] = 0.5
        return out
    out[mask] = (a[mask] - lo) / (hi - lo)
    return out


def plot_scenario_model_overview_fr_cr_ds(
    scenario_axis_df: pd.DataFrame,
    save_dir: str,
    cmap: str = "viridis",
) -> None:
    """
    One PNG per distinct Temperature: 3 subplots (FR_scenario, CR_scenario, DS_scenario).
    Rows = historical scenarios, columns = models. Each subplot min–max normalizes its own matrix
    so color scale uses the full [0, 1] range for that metric (interpretation: relative within panel).
    """
    os.makedirs(save_dir, exist_ok=True)
    if scenario_axis_df is None or len(scenario_axis_df) == 0:
        return

    d = scenario_axis_df.copy()
    metrics = [
        ("FR_scenario", "FR (scenario-mean)\n1 − mean_v JSD(Generic, Specific)"),
        ("CR_scenario", "CR (scenario-mean)\nmean JSD(base, semantic variants)"),
        ("DS_scenario", "DS (scenario-mean)\nmean DS_condition over cells"),
    ]

    for temp, dt in d.groupby("Temperature", dropna=False):
        scenarios = sorted(dt["scenario"].astype(str).unique())
        models = sorted(dt["Model"].astype(str).unique())
        col_labels = [_short_model_name(m) for m in models]

        fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.2), constrained_layout=True)
        tfloat = float(temp)

        for ax, (col, title) in zip(axes, metrics):
            mat = np.full((len(scenarios), len(models)), np.nan, dtype=float)
            for i, scen in enumerate(scenarios):
                for j, mod in enumerate(models):
                    sub = dt[(dt["scenario"] == scen) & (dt["Model"] == mod)]
                    if len(sub) == 0:
                        continue
                    v = sub[col].astype(float)
                    mat[i, j] = float(v.iloc[0]) if np.isfinite(v.iloc[0]) else np.nan

            scaled = _minmax01_2d(mat)
            masked = np.ma.masked_invalid(scaled)

            im = ax.imshow(masked, vmin=0.0, vmax=1.0, cmap=cmap, aspect="auto")
            ax.set_xticks(np.arange(len(models)))
            ax.set_xticklabels(col_labels, rotation=35, ha="right", fontsize=8)
            ax.set_yticks(np.arange(len(scenarios)))
            ax.set_yticklabels(scenarios, fontsize=8)
            ax.set_title(title, fontsize=10)
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            cbar.set_label("min–max scaled", fontsize=8)

        fig.suptitle(
            f"Scenario × model — FR, CR, DS (each panel scaled separately) @ T={tfloat:g}",
            fontsize=12,
        )
        out = os.path.join(
            save_dir,
            f"eval_scenario_model_overview_FR_CR_DS__T{tfloat:g}.png",
        )
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out}")


def run_fr_cr_ds_localization(
    input_dir: str = "infer_results",
    summary_dir: str = "./final_results/summary",
    plots_dir: str = "./final_results/plots",
    fr_top_k: int = 8,
    deep_dive: bool = False,
) -> None:
    os.makedirs(summary_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    print("Loading profile data...")
    df = load_profile_data(input_dir=input_dir)
    print(f"Rows loaded: {len(df):,}")

    # --- Scenario × model overview (FR / CR / DS): aligns with main-paper axis definitions
    print("Building condition-level DS (for DS_scenario aggregation)...")
    ds_cond_df = build_condition_level_ds_diagnostics(df)
    print("Building scenario-level FR / CR / DS table...")
    scenario_axis_df = build_scenario_axis_table(df, ds_cond_table=ds_cond_df)
    axis_path = os.path.join(summary_dir, "model_profile_scenario_axes_fr_cr_ds.csv")
    scenario_axis_df.to_csv(axis_path, index=False)
    print(f"Saved: {axis_path}")

    print("Plotting scenario × model overview (3 panels per temperature)...")
    plot_scenario_model_overview_fr_cr_ds(scenario_axis_df, save_dir=plots_dir)

    print("\nDone (overview only).")

    if deep_dive:
        try:
            from result_analysis.deep_dive_fr_cr_ds_plots import run_deep_dive_fr_cr_ds
        except ImportError:
            from deep_dive_fr_cr_ds_plots import run_deep_dive_fr_cr_ds
        print("Running FR/CR/DS deep-dive plots (Table-3 default cells)...")
        run_deep_dive_fr_cr_ds(
            input_dir=input_dir,
            summary_dir=summary_dir,
            plots_dir=plots_dir,
        )

    # --- Legacy outputs (uncomment to restore)
    # print("Building condition-level DS diagnostics...")
    # ds_path = os.path.join(summary_dir, "model_profile_ds_condition_diagnostics.csv")
    # ds_cond_df.to_csv(ds_path, index=False)
    # print(f"Saved: {ds_path}")
    # plot_ds_condition_heatmaps(ds_cond_df, save_dir=plots_dir)
    # plot_ds_failure_mode_scatter(ds_cond_df, save_dir=plots_dir)
    #
    # fr_dir_df = build_fr_directionality_summary(df)
    # fr_path = os.path.join(summary_dir, "model_profile_fr_directionality_by_strategy.csv")
    # fr_dir_df.to_csv(fr_path, index=False)
    # plot_fr_directionality_bars(fr_dir_df, save_dir=plots_dir, top_k=fr_top_k)
    #
    # cr_df = build_cr_scenario_table(df)
    # cr_path = os.path.join(summary_dir, "model_profile_cr_by_scenario.csv")
    # cr_df.to_csv(cr_path, index=False)
    # cr_het = build_cr_scenario_heterogeneity(cr_df)
    # cr_het.to_csv(os.path.join(summary_dir, "model_profile_cr_scenario_heterogeneity.csv"), index=False)
    # plot_cr_by_scenario_bars(cr_df, save_dir=plots_dir)
    #
    # print(ds_cond_df.head(8).round(4).to_string(index=False))
    # print(fr_dir_df.head(8).round(4).to_string(index=False))
    # print(cr_df.head(12).round(4).to_string(index=False))
    # print(cr_het.round(4).to_string(index=False))

    print("\nPreview: scenario-axis table (head)")
    print(scenario_axis_df.head(12).round(4).to_string(index=False))


if __name__ == "__main__":
    run_fr_cr_ds_localization(
        input_dir="infer_results",
        summary_dir="./final_results/summary",
        plots_dir="./final_results/plots",
        fr_top_k=8,
    )
