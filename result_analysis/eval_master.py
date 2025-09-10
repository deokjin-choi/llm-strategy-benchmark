# result_analysis/eval_master.py
"""
Master evaluation script:
- Loads merged results (from infer_results/*.csv via scenario_eval)
- Runs ratio analysis, delta analysis, PCA, signal usage
- Generates all summary CSVs into final_results/summary/
- Generates all plots into final_results/plots/
"""

import os
from eval_overall_strategy_ratio import plot_strategy_ratio_heatmap
from eval_total_eda import (
    load_overall_ratio, plot_ratio_heatmap, plot_ratio_delta_from_base,
    compute_ratio_metrics, plot_ratio_metrics, plot_ratio_pca,
    load_delta_all, compute_delta_metrics,
    plot_delta_sensitivity, plot_delta_sign_consistency,
    plot_anchoring_index, plot_shock_size
)
from eval_type_delta import load_delta_data, plot_delta_by_case
from scenario_eval import analyze_all_scenarios

def run_master_eval():
    # Step 1: Run scenario-level aggregation
    df_combined = analyze_all_scenarios()

    # Define dirs
    summary_dir = "./final_results/summary"

    # Step 2: Overall ratios
    ratio_df = load_overall_ratio(os.path.join(summary_dir, "analysis_overall_ratio.csv"))
    plot_ratio_heatmap(ratio_df)
    plot_ratio_delta_from_base(ratio_df)
    metrics_ratio = compute_ratio_metrics(ratio_df)
    plot_ratio_metrics(metrics_ratio)
    plot_ratio_pca(ratio_df)

    # Step 3: Delta metrics
    delta_df = load_delta_all(summary_dir)
    metrics_delta, anch_df, shock_df = compute_delta_metrics(delta_df)
    plot_delta_sensitivity(metrics_delta)
    plot_delta_sign_consistency(metrics_delta)
    plot_anchoring_index(anch_df)
    plot_shock_size(shock_df)

    # Step 4: Case-wise deltas
    df_summary = load_delta_data(summary_dir)
    plot_delta_by_case(df_summary)

if __name__ == "__main__":
    run_master_eval()
