# scenario_eval.py

import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

# -----------------------------
# 1) Helper functions for analysis
# -----------------------------
def ratio_table(df, keys):
    """Return normalized ratio table of Standard Mapping over the given keys."""
    counts = (
        df.groupby(keys + ["Standard Mapping"])
        .size()
        .rename("Count")
        .reset_index()
    )
    totals = counts.groupby(keys)["Count"].transform("sum")
    counts["Ratio"] = counts["Count"] / totals
    return counts.pivot(index=keys, columns="Standard Mapping", values="Ratio").fillna(0)

def diff_table(df, keys):
    """Calculate the delta (specific - generic) for a given set of keys."""
    specific_df = ratio_table(df[df['problem_type'] == 'specific'], keys)
    generic_df = ratio_table(df[df['problem_type'] == 'generic'], keys)
    specific_df, generic_df = specific_df.align(generic_df, join='outer', fill_value=0)
    diff_df = specific_df - generic_df
    diff_df.columns = [f"Δ {c} (specific-generic)" for c in diff_df.columns]
    return diff_df


# -----------------------------
# 2) Main Evaluation Logic
# -----------------------------
def analyze_all_scenarios():
    """
    infer_results 폴더의 *scenarios*.csv 파일을 불러와 분석하고,
    결과는 final_results/summary, final_results/plots에 저장합니다.
    """
    print("--- Starting analysis from 'infer_results' ---")

    input_dir = "infer_results"        # 입력: merge된 csv
    output_dir = "final_results"       # 출력: 최종 결과
    summary_dir = os.path.join(output_dir, "summary")
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(summary_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # infer_results 폴더의 모든 *scenarios*.csv 파일 수집
    all_combined_files = glob.glob(os.path.join(input_dir, "*scenarios*.csv"))
    if not all_combined_files:
        print(f"No CSV files found in '{input_dir}'. Exiting.")
        return

    standardization_map = {
        'C → Open Innovation': 'Open Innovation',
        'A → Technology Leadership': 'Technology Leadership',
        'C -> Open Innovation': 'Open Innovation',
        'A -> Technology Leadership': 'Technology Leadership'
    }

    all_data_frames = []
    for file_path in all_combined_files:
        df = pd.read_csv(file_path)

        # 표준화
        df['Standard Mapping'].replace(standardization_map, inplace=True)
        df['Standard Mapping'].fillna('N/A', inplace=True)

        # 파일명에서 시나리오 타입 추출
        file_name = os.path.basename(file_path).replace(".csv", "")
        if file_name.endswith("_scenarios"):
            scenario_type = "base"
        else:
            scenario_type = file_name.split("_", 1)[-1]   # 접두사 제거
            if scenario_type.startswith("scenarios_"):
                scenario_type = scenario_type.replace("scenarios_", "")

        df['scenario_type'] = scenario_type
        all_data_frames.append(df)

    if not all_data_frames:
        print("No data to analyze. Exiting.")
        return

    df_combined = pd.concat(all_data_frames, ignore_index=True)
    df_combined.fillna({
        'Standard Mapping': 'N/A',
        'Chosen Option': 'N/A',
        'Key Signals Used': ''
    }, inplace=True)

    print(f"\n--- Combined analysis running. Results will be saved in '{output_dir}' ---")

    # Overall ratio
    print("\n=== Overall Standard Mapping Ratio by Scenario Type ===")
    df_overall = ratio_table(df_combined, ["scenario_type"])
    print(df_overall.round(3).to_string())
    df_overall.to_csv(os.path.join(summary_dir, "analysis_overall_ratio.csv"))

    # Delta vs base
    if "base" in df_combined['scenario_type'].unique():
        print("\n=== Delta of Scenarios from Base ===")
        base_ratios = ratio_table(df_combined[df_combined['scenario_type'] == 'base'],
                                  ["scenario", "problem_type"])
        for st_to_compare in [st for st in df_combined['scenario_type'].unique() if st != 'base']:
            compare_ratios = ratio_table(df_combined[df_combined['scenario_type'] == st_to_compare],
                                         ["scenario", "problem_type"])
            compare_ratios, base_ratios_aligned = compare_ratios.align(base_ratios,
                                                                       join='outer', fill_value=0)
            delta_df = compare_ratios - base_ratios_aligned
            delta_df.columns = [f"Δ {c} ({st_to_compare}-base)" for c in delta_df.columns]

            print(f"\n--- Delta: {st_to_compare} vs Base ---")
            print(delta_df.round(3).to_string())
            delta_df.to_csv(os.path.join(summary_dir,
                                         f"analysis_delta_{st_to_compare}-base.csv"))

    # Signal usage
    print("\n=== Key Signal Usage by Scenario Type ===")
    df_combined['used_any_signal'] = df_combined['Key Signals Used'].apply(
        lambda x: 1 if pd.notna(x) and x != '' else 0
    )
    signal_usage = df_combined.groupby(['scenario_type'])['used_any_signal'].mean()
    print(signal_usage.round(3).to_string())
    signal_usage.to_csv(os.path.join(summary_dir, "analysis_signal_usage_by_type.csv"))

    print(f"\n--- Analysis finished. Results saved in '{output_dir}' ---")
    print("-" * 50)

    return df_combined


if __name__ == "__main__":
    analyze_all_scenarios()
