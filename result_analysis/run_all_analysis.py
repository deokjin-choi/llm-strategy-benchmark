# result_analysis/eval_master.py
"""
Master evaluation script:
- Loads merged results (from infer_results/*.csv via scenario_eval)
- Runs ratio analysis, delta analysis, PCA, signal usage
- Generates all summary CSVs into final_results/summary/
- Generates all plots into final_results/plots/
"""

import os

from result_analysis.overall_results import (
    load_overall_ratio, plot_ratio_heatmap, plot_ratio_heatamp_delta_from_base,
    compute_ratio_metrics, plot_ratio_metrics, plot_ratio_pca
)

from result_analysis.delta_analysis import (
    load_delta_all, compute_delta_metrics,
    plot_delta_sensitivity, plot_delta_sign_consistency,
    plot_anchoring_index, plot_shock_size
)   

from result_analysis.brand_bias_analysis import load_delta, plot_generic_vs_specific_delta, plot_correlation_generic_vs_specific
from result_analysis.make_summary import build_summary_from_infer

def run_master_eval():
    # Step 1: build_summary for analaysis -> summary 밑에 필요한 분석파일(ratio, delta 등) 생성
    build_summary_from_infer()

    # Define dirs
    summary_dir = "./final_results/summary"

    # Step 2: Overall ratios
    ratio_df = load_overall_ratio(os.path.join(summary_dir, "analysis_overall_ratio.csv"))
    plot_ratio_heatmap(ratio_df)
    plot_ratio_heatamp_delta_from_base(ratio_df)
    metrics_ratio = compute_ratio_metrics(ratio_df)
    plot_ratio_metrics(metrics_ratio) # 시나리오 별 엔트로피, base 대비 JSD, base와 rank correlation
    plot_ratio_pca(ratio_df)

    # Step 3: Delta metrics
    delta_df = load_delta_all(summary_dir)
    metrics_delta, anch_df, shock_df = compute_delta_metrics(delta_df)
    plot_delta_sensitivity(metrics_delta)
    plot_delta_sign_consistency(metrics_delta)
    plot_anchoring_index(anch_df)
    plot_shock_size(shock_df)

    # Step 4: generic vs specific 차이 분석
    df_summary = load_delta(summary_dir)
    plot_generic_vs_specific_delta(df_summary)
    plot_correlation_generic_vs_specific(df_summary)

if __name__ == "__main__":
    run_master_eval()


