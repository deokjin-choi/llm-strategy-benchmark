"""
Model × temperature audit-metric summaries for Table 6 (§5.6.1).

Mean RDS uses the same cell macro-average as Section 5.5:
  1. Matched Generic–Specific rationale pairs per
     (Model, Temperature, scenario, Num Context, context_variant, strategy)
  2. Mean RDS within each cell
  3. Equal-weight macro-average across cells per (Model, Temperature)

Mean semantic JSD and mean firm-identity JSD come from model_profile_by_model_temp.csv
(CR and 1 − FR, macro-averaged over scenarios).

Output: final_results/summary/audit_metrics_by_model_temp.csv
"""

from __future__ import annotations

import os

import pandas as pd

from result_analysis.rds_by_strategy_ci import build_cell_level_rds, load_rds_pairs_with_distances

SUMMARY_DIR = "./final_results/summary"
OUT_PATH = os.path.join(SUMMARY_DIR, "audit_metrics_by_model_temp.csv")
PROFILE_PATH = os.path.join(SUMMARY_DIR, "model_profile_by_model_temp.csv")


def compute_audit_metrics_by_model_temp() -> pd.DataFrame:
    pairs = load_rds_pairs_with_distances()
    cells = build_cell_level_rds(pairs)
    rds_by_mt = (
        cells.groupby(["Model", "Temperature"], dropna=False)["mean_rds"]
        .mean()
        .reset_index(name="mean_rds_macro")
    )

    prof = pd.read_csv(PROFILE_PATH)
    prof["mean_semantic_jsd"] = prof["CR_context_responsiveness"]
    prof["mean_firm_identity_jsd"] = 1.0 - prof["FR_framing_robustness"]

    merged = prof.merge(rds_by_mt, on=["Model", "Temperature"], how="left")
    merged["model_short"] = merged["Model"].str.split("/").str[-1]

    out = merged[
        [
            "model_short",
            "Temperature",
            "mean_semantic_jsd",
            "mean_firm_identity_jsd",
            "mean_rds_macro",
        ]
    ].sort_values(["model_short", "Temperature"])
    return out


def main() -> None:
    os.makedirs(SUMMARY_DIR, exist_ok=True)
    out = compute_audit_metrics_by_model_temp()
    out.to_csv(OUT_PATH, index=False)
    print(f"Wrote {OUT_PATH}")
    print(out.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
