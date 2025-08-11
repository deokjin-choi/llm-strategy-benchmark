# scenario_eval.py

import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import shutil # shutil 모듈 추가

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

def plot_bar_chart(df, title, filename, output_dir):
    """Generate and save a bar chart for generic vs specific ratios."""
    plt.figure(figsize=(10, 6))
    
    cols = sorted(df.columns)
    x = range(len(cols))
    w = 0.35
    
    vals_g = [df.loc['generic'].get(c, 0) for c in cols]
    vals_s = [df.loc['specific'].get(c, 0) for c in cols]

    plt.bar([i - w/2 for i in x], vals_g, width=w, label="generic")
    plt.bar([i + w/2 for i in x], vals_s, width=w, label="specific")
    
    plt.xticks(list(x), cols, rotation=20, ha="right")
    plt.ylabel("Ratio")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, f"{filename}.png")
    plt.savefig(plot_path, dpi=150)
    plt.show()

def plot_scenario_comparison_bar_chart(df, scenario, filename, output_dir):
    """Generate and save a bar chart comparing different scenario types (base, count_fact, reframe)."""
    
    df_filtered = df[df['scenario'] == scenario]
    if df_filtered.empty:
        print(f"No data for scenario: {scenario}")
        return

    df_plot_data = df_filtered.groupby(['scenario_type', 'Standard Mapping']).size().unstack(fill_value=0)
    
    df_plot_data = df_plot_data.div(df_plot_data.sum(axis=1), axis=0)
    
    plt.figure(figsize=(12, 7))
    df_plot_data.plot(kind='bar', figsize=(12, 7), rot=0)
    plt.title(f"Standard Mapping Ratios Comparison for Scenario: {scenario}")
    plt.xlabel("Scenario Type")
    plt.ylabel("Ratio")
    plt.xticks(rotation=0)
    plt.legend(title="Standard Mapping", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, f"{filename}.png")
    plt.savefig(plot_path, dpi=150)
    plt.show()

def plot_delta_comparison(base_ratios, compare_ratios, title, filename, output_dir):
    """Generate and save a stacked bar chart showing the delta (compare - base)."""
    
    delta_df = compare_ratios - base_ratios
    delta_df = delta_df.fillna(0)

    positive_delta = delta_df.clip(lower=0)
    negative_delta = delta_df.clip(upper=0)
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    positive_delta.plot(kind='bar', stacked=True, ax=ax, width=0.4, color='tab:blue', position=0)
    negative_delta.plot(kind='bar', stacked=True, ax=ax, width=0.4, color='tab:red', position=1)
    
    ax.set_title(title)
    ax.set_xlabel("Scenario & Problem Type")
    ax.set_ylabel("Delta (Change in Ratio)")
    ax.legend(title="Delta", loc='upper left', bbox_to_anchor=(1, 1))
    ax.axhline(0, color='black', linewidth=0.8)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"{filename}.png")
    plt.savefig(plot_path, dpi=150)
    plt.show()


# -----------------------------
# 2) Main Evaluation Logic
# -----------------------------
def analyze_all_scenarios():
    print("--- Starting combined analysis for all 'results' directories ---")
    
    results_dirs = [d for d in os.listdir(os.getcwd()) if os.path.isdir(d) and "results" in d]
    if not results_dirs:
        print("No directories with 'results' in their name found. Exiting.")
        return
        
    all_data_frames = []
    output_dir = "combined_analysis"
    plots_dir = os.path.join(output_dir, "plots")
    
    # combined_analysis 폴더가 없으면 생성
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    for results_dir in results_dirs:
        scenario_type = results_dir.replace('results_', '')
        if scenario_type == 'results':
            scenario_type = 'base'
            
        all_files = glob.glob(os.path.join(results_dir, "*.csv"))
        if not all_files:
            print(f"No CSV files found in {results_dir}. Skipping.")
            continue
            
        df = pd.concat([pd.read_csv(f) for f in all_files], ignore_index=True)
        df['scenario_type'] = scenario_type
        
        combined_csv_path = os.path.join(results_dir, f"{results_dir}_all.csv")
        df.drop(columns=['scenario_type'], inplace=True)
        df.to_csv(combined_csv_path, index=False)
        print(f"All CSVs in {results_dir} merged into {combined_csv_path}")

        # 병합된 파일을 combined_analysis 폴더로 복사
        destination_path = os.path.join(output_dir, f"{results_dir}_all.csv")
        shutil.copyfile(combined_csv_path, destination_path)
        print(f"Copied '{os.path.basename(combined_csv_path)}' to '{output_dir}'.")

        # 원본 파일 삭제 로직은 주석 처리 또는 삭제 가능
        # for file in all_files:
        #     os.remove(file)
        # print(f"Original CSV files in {results_dir} have been removed.")

        df['scenario_type'] = scenario_type
        all_data_frames.append(df)

    if not all_data_frames:
        print("No data to analyze. Exiting.")
        return

    df_combined = pd.concat(all_data_frames, ignore_index=True)
    df_combined.fillna({'Standard Mapping': 'N/A', 'Chosen Option': 'N/A', 'Key Signals Used': ''}, inplace=True)
    
    print(f"\n--- Combined analysis for all scenarios is running. Results will be saved in '{output_dir}' ---")

    print("\n=== Overall Standard Mapping Ratio by Scenario Type ===")
    df_overall = ratio_table(df_combined, ["scenario_type"])
    print(df_overall.round(3).to_string())
    df_overall.to_csv(os.path.join(output_dir, "analysis_overall_ratio.csv"))
    
    print("\n=== Delta of Scenarios from Base (e.g., count_fact - base) ===")
    base_ratios = ratio_table(df_combined[df_combined['scenario_type'] == 'base'], ["scenario", "problem_type"])
    
    scenario_types_to_compare = [st for st in df_combined['scenario_type'].unique() if st != 'base']
    for st_to_compare in scenario_types_to_compare:
        compare_ratios = ratio_table(df_combined[df_combined['scenario_type'] == st_to_compare], ["scenario", "problem_type"])
        
        compare_ratios, base_ratios_aligned = compare_ratios.align(base_ratios, join='outer', fill_value=0)
        
        delta_df = compare_ratios - base_ratios_aligned
        delta_df.columns = [f"Δ {c} ({st_to_compare}-base)" for c in delta_df.columns]
        
        print(f"\n--- Delta: {st_to_compare} vs Base ---")
        print(delta_df.round(3).to_string())
        delta_df.to_csv(os.path.join(output_dir, f"analysis_delta_{st_to_compare}-base.csv"))
        
        plot_delta_comparison(base_ratios_aligned, compare_ratios,
                              title=f"Delta Comparison ({st_to_compare} vs Base)",
                              filename=f"delta_comparison_{st_to_compare}-base",
                              output_dir=plots_dir)
        print(f"Delta comparison plot for '{st_to_compare}' generated.")


    print("\n=== Key Signal Usage by Scenario Type ===")
    df_combined['used_any_signal'] = df_combined['Key Signals Used'].apply(lambda x: 1 if pd.notna(x) and x != '' else 0)
    signal_usage = df_combined.groupby(['scenario_type'])['used_any_signal'].mean()
    print(signal_usage.round(3).to_string())
    signal_usage.to_csv(os.path.join(output_dir, "analysis_signal_usage_by_type.csv"))
    
    print("\n=== Generating comparison plots for each scenario ===")
    unique_scenarios = df_combined['scenario'].unique()
    for scenario_for_plot in unique_scenarios:
        try:
            plot_scenario_comparison_bar_chart(df_combined, scenario_for_plot, f"comparison_{scenario_for_plot}", plots_dir)
            print(f"Comparison plot for scenario '{scenario_for_plot}' generated successfully.")
        except Exception as e:
            print(f"Could not generate plot for scenario '{scenario_for_plot}': {e}")
        
    print(f"\n--- Combined analysis finished. Results saved in '{output_dir}'. ---")
    print("-" * 50)


if __name__ == "__main__":
    analyze_all_scenarios()