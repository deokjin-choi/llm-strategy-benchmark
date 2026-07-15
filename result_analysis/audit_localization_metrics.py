"""
Audit localization metrics for §5.6 (Table 6, scenario × model landscape).

Context sensitivity (per model × temperature × scenario cell):
  For each fixed (Num Context, problem_type), compute JSD(base, v) for each
  perturbation variant v ∈ {competitive_dynamics, count_fact, opp_focus,
  randomized_numbers}, take the maximum over v, then macro-average those
  condition-level maxima within the cell.

Firm-identity sensitivity (per cell):
  For each fixed (Num Context, context_variant), compute |Δp| per strategy,
  take max over strategies, then macro-average within the cell.

Rationale sensitivity (per cell):
  Mean RDS over matched Generic–Specific pairs in the cell.

Table 6 macro-averages context and firm-identity metrics over scenarios;
firm-identity uses mean max |Δp| per scenario cell (§5.3); RDS follows Section 5.5.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from result_analysis.model_behavioral_profile import (
        PERTURBATION_VARIANTS,
        load_profile_data,
        valid_strategies,
        _safe_filename,
        _safe_jsd,
        _short_model_name,
        _strategy_distribution,
    )
    from result_analysis.rds_by_strategy_ci import build_cell_level_rds, load_rds_pairs_with_distances
except ImportError:
    from model_behavioral_profile import (
        PERTURBATION_VARIANTS,
        load_profile_data,
        valid_strategies,
        _safe_filename,
        _safe_jsd,
        _short_model_name,
        _strategy_distribution,
    )
    from rds_by_strategy_ci import build_cell_level_rds, load_rds_pairs_with_distances

SUMMARY_DIR = "./final_results/summary"
PLOTS_DIR = "./final_results/plots"


def compute_context_sensitivity_cell(g: pd.DataFrame) -> float:
    """Max JSD(base, v) over perturbation variants, averaged over conditions."""
    vals: List[float] = []
    for _, gp in g.groupby(["Num Context", "problem_type"], dropna=False):
        base = gp[gp["context_variant"] == "base"]["Standard Mapping"]
        if len(base) == 0:
            continue
        p_base = _strategy_distribution(base)
        jsd_by_v: List[float] = []
        for v in PERTURBATION_VARIANTS:
            dv = gp[gp["context_variant"] == v]["Standard Mapping"]
            if len(dv) == 0:
                continue
            jsd_by_v.append(_safe_jsd(p_base, _strategy_distribution(dv)))
        if jsd_by_v:
            vals.append(float(max(jsd_by_v)))
    return float(np.nanmean(vals)) if vals else np.nan


def compute_firm_identity_max_delta_p_cell(g: pd.DataFrame) -> float:
    vals: List[float] = []
    for _, gv in g.groupby(["Num Context", "context_variant"], dropna=False):
        gg = gv[gv["problem_type"] == "generic"]["Standard Mapping"]
        ss = gv[gv["problem_type"] == "specific"]["Standard Mapping"]
        if len(gg) == 0 or len(ss) == 0:
            continue
        pg = _strategy_distribution(gg)
        ps = _strategy_distribution(ss)
        max_dp = float(np.max(np.abs(ps - pg)))
        vals.append(max_dp)
    return float(np.nanmean(vals)) if vals else np.nan


def compute_mean_firm_identity_jsd_cell(g: pd.DataFrame) -> float:
    """Mean JSD(Generic, Specific) over conditions in the cell."""
    vals: List[float] = []
    for _, gv in g.groupby(["Num Context", "context_variant"], dropna=False):
        gg = gv[gv["problem_type"] == "generic"]["Standard Mapping"]
        ss = gv[gv["problem_type"] == "specific"]["Standard Mapping"]
        if len(gg) == 0 or len(ss) == 0:
            continue
        vals.append(_safe_jsd(_strategy_distribution(gg), _strategy_distribution(ss)))
    return float(np.nanmean(vals)) if vals else np.nan


def build_scenario_audit_table(
    df: pd.DataFrame,
    rds_cells: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    mt_s_keys = ["Model", "Temperature", "scenario"]

    rds_lookup: Dict[tuple, float] = {}
    if rds_cells is not None and len(rds_cells) > 0:
        for _, r in rds_cells.iterrows():
            rds_lookup[
                (
                    r["Model"],
                    float(r["Temperature"]),
                    str(r["scenario"]),
                )
            ] = float(r["mean_rds"])

    for (model, temp, scenario), g in df.groupby(mt_s_keys, dropna=False):
        key = (model, float(temp), str(scenario))
        rows.append(
            {
                "Model": model,
                "Temperature": float(temp),
                "scenario": scenario,
                "context_sensitivity": compute_context_sensitivity_cell(g),
                "firm_identity_max_delta_p": compute_firm_identity_max_delta_p_cell(g),
                "mean_firm_identity_jsd": compute_mean_firm_identity_jsd_cell(g),
                "mean_rds": rds_lookup.get(key, np.nan),
            }
        )

    out = pd.DataFrame(rows)
    if len(out) == 0:
        return out
    return out.sort_values(["Model", "Temperature", "scenario"]).reset_index(drop=True)


def build_rds_scenario_means(rds_cells: pd.DataFrame) -> pd.DataFrame:
    return (
        rds_cells.groupby(["Model", "Temperature", "scenario"], dropna=False)["mean_rds"]
        .mean()
        .reset_index()
    )


def build_model_temp_audit_table(scenario_audit: pd.DataFrame, rds_cells: pd.DataFrame) -> pd.DataFrame:
    rds_by_mt = (
        rds_cells.groupby(["Model", "Temperature"], dropna=False)["mean_rds"]
        .mean()
        .reset_index(name="mean_rds_macro")
    )
    ctx_fi = (
        scenario_audit.groupby(["Model", "Temperature"], dropna=False)
        .agg(
            mean_context_jsd=("context_sensitivity", "mean"),
            mean_firm_identity_max_delta_p=("firm_identity_max_delta_p", "mean"),
        )
        .reset_index()
    )
    merged = ctx_fi.merge(rds_by_mt, on=["Model", "Temperature"], how="left")
    merged["model_short"] = merged["Model"].str.split("/").str[-1]
    return merged.sort_values(["model_short", "Temperature"]).reset_index(drop=True)


def _minmax01_2d(a: np.ndarray) -> np.ndarray:
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


def plot_scenario_model_audit_landscape(
    scenario_audit: pd.DataFrame,
    save_dir: str,
    cmap: str = "viridis",
) -> None:
    """
    Scenario × model heatmaps for audit axes (no FR/CR/DS legacy panels).
    One PNG per temperature; three panels: context JSD, firm-identity |Δp|, mean RDS.
    """
    os.makedirs(save_dir, exist_ok=True)
    if scenario_audit is None or len(scenario_audit) == 0:
        return

    d = scenario_audit.copy()
    metrics = [
        (
            "context_sensitivity",
            "Context sensitivity\nmax JSD(base, perturbation variants)",
        ),
        (
            "firm_identity_max_delta_p",
            "Firm-identity sensitivity\nmax |Δp| across strategies",
        ),
        (
            "mean_rds",
            "Rationale sensitivity\nmean RDS (matched pairs)",
        ),
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
            f"Scenario × model audit landscape (each panel scaled separately) @ T={tfloat:g}",
            fontsize=12,
        )
        out = os.path.join(
            save_dir,
            f"eval_audit_scenario_model_landscape__T{tfloat:g}.png",
        )
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out}")


def plot_scenario_model_audit_landscape_paper(
    scenario_audit: pd.DataFrame,
    save_path: str,
    cmap: str = "viridis",
) -> None:
    """Two temperatures × three audit panels for §5.6.2 Fig. 8."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if scenario_audit is None or len(scenario_audit) == 0:
        return

    d = scenario_audit.copy()
    temps = sorted(d["Temperature"].astype(float).unique())
    metrics = [
        ("context_sensitivity", "Context sensitivity"),
        ("firm_identity_max_delta_p", "Firm-identity sensitivity"),
        ("mean_rds", "Rationale sensitivity"),
    ]
    scenarios = sorted(d["scenario"].astype(str).unique())
    models = sorted(d["Model"].astype(str).unique())
    col_labels = [_short_model_name(m) for m in models]

    fig, axes = plt.subplots(
        len(temps), len(metrics), figsize=(14.5, 4.6 * len(temps)), constrained_layout=True
    )
    if len(temps) == 1:
        axes = np.array([axes])

    for i, temp in enumerate(temps):
        dt = d[d["Temperature"].astype(float) == float(temp)]
        for j, (col, title) in enumerate(metrics):
            ax = axes[i, j]
            mat = np.full((len(scenarios), len(models)), np.nan, dtype=float)
            for si, scen in enumerate(scenarios):
                for mj, mod in enumerate(models):
                    sub = dt[(dt["scenario"] == scen) & (dt["Model"] == mod)]
                    if len(sub) == 0:
                        continue
                    v = sub[col].astype(float)
                    mat[si, mj] = float(v.iloc[0]) if np.isfinite(v.iloc[0]) else np.nan

            scaled = _minmax01_2d(mat)
            masked = np.ma.masked_invalid(scaled)
            im = ax.imshow(masked, vmin=0.0, vmax=1.0, cmap=cmap, aspect="auto")
            ax.set_xticks(np.arange(len(models)))
            ax.set_xticklabels(col_labels, rotation=35, ha="right", fontsize=7)
            ax.set_yticks(np.arange(len(scenarios)))
            ax.set_yticklabels(scenarios, fontsize=7)
            row_title = f"T={float(temp):g} — {title}" if j == 0 else title
            ax.set_title(row_title if j == 0 else title, fontsize=9)
            if j == len(metrics) - 1:
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    fig.suptitle(
        "Scenario × model audit landscape (each panel min–max scaled separately)",
        fontsize=12,
    )
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def run_audit_localization(
    input_dir: str = "infer_results",
    summary_dir: str = SUMMARY_DIR,
    plots_dir: str = PLOTS_DIR,
) -> pd.DataFrame:
    os.makedirs(summary_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    print("Loading profile data...")
    df = load_profile_data(input_dir=input_dir)
    print(f"Rows loaded: {len(df):,}")

    print("Loading RDS cell means...")
    pairs = load_rds_pairs_with_distances()
    rds_cells = build_cell_level_rds(pairs)
    rds_scenario = build_rds_scenario_means(rds_cells)

    print("Building scenario-level audit table...")
    scenario_audit = build_scenario_audit_table(df, rds_cells=rds_scenario)
    scenario_path = os.path.join(summary_dir, "audit_scenario_model_cells.csv")
    scenario_audit.to_csv(scenario_path, index=False)
    print(f"Saved: {scenario_path}")

    model_temp = build_model_temp_audit_table(scenario_audit, rds_cells)
    mt_path = os.path.join(summary_dir, "audit_metrics_by_model_temp.csv")
    model_temp[
        [
            "model_short",
            "Temperature",
            "mean_context_jsd",
            "mean_firm_identity_max_delta_p",
            "mean_rds_macro",
        ]
    ].to_csv(mt_path, index=False)
    print(f"Saved: {mt_path}")

    print("Plotting scenario × model audit landscape...")
    plot_scenario_model_audit_landscape(scenario_audit, save_dir=plots_dir)
    paper_fig = os.path.join(plots_dir, "eval_audit_scenario_model_landscape__paper.png")
    plot_scenario_model_audit_landscape_paper(scenario_audit, save_path=paper_fig)

    return model_temp


def main() -> None:
    out = run_audit_localization()
    print(out.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
