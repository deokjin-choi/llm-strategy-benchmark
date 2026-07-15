"""
Model × temperature audit-metric summaries for Table 6 (§5.6.1).

Mean RDS uses the same cell macro-average as Section 5.5:
  1. Matched Generic–Specific rationale pairs per
     (Model, Temperature, scenario, Num Context, context_variant, strategy)
  2. Mean RDS within each cell
  3. Equal-weight macro-average across cells per (Model, Temperature)

Mean context JSD and mean firm-identity max |Δp| use audit_localization_metrics
(max JSD over perturbation variants including randomized_numbers; max |Δp| over
archetypes within each condition, then scenario-macro-average).

Output: final_results/summary/audit_metrics_by_model_temp.csv
"""

from __future__ import annotations

import os

from result_analysis.audit_localization_metrics import run_audit_localization

SUMMARY_DIR = "./final_results/summary"
OUT_PATH = os.path.join(SUMMARY_DIR, "audit_metrics_by_model_temp.csv")


def compute_audit_metrics_by_model_temp():
    return run_audit_localization()


def main() -> None:
    os.makedirs(SUMMARY_DIR, exist_ok=True)
    out = compute_audit_metrics_by_model_temp()
    print(out.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
