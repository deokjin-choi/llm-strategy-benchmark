# scenario_eval.py

import pandas as pd
import matplotlib.pyplot as plt
import os

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

def plot_bar_chart(df, title, filename):
    """Generate and save a bar chart for generic vs specific ratios."""
    plt.figure(figsize=(10, 6))
    
    # Sort columns alphabetically for consistent plotting order
    cols = sorted(df.columns)
    x = range(len(cols))
    w = 0.35
    
    df_generic = df.xs('generic', level='problem_type', drop_level=False)
    df_specific = df.xs('specific', level='problem_type', drop_level=False)

    vals_g = [df_generic.iloc[0].get(c, 0) for c in cols]
    vals_s = [df_specific.iloc[0].get(c, 0) for c in cols]

    plt.bar([i - w/2 for i in x], vals_g, width=w, label="generic")
    plt.bar([i + w/2 for i in x], vals_s, width=w, label="specific")
    
    plt.xticks(list(x), cols, rotation=20, ha="right")
    plt.ylabel("Ratio")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"plots/{filename}.png", dpi=150)
    plt.show()

# -----------------------------
# 2) Main Evaluation Logic
# -----------------------------
def analyze_results(results_dir: str):
    # 폴더가 없으면 생성
    if not os.path.exists("plots"):
        os.makedirs("plots")
    
    # 모든 CSV 파일을 하나의 DataFrame으로 통합
    all_files = [os.path.join(results_dir, f) for f in os.listdir(results_dir) if f.endswith('.csv')]
    if not all_files:
        print("No CSV files found in the results directory.")
        return
        
    df = pd.concat([pd.read_csv(f) for f in all_files], ignore_index=True)
    
    # NaN 값 처리: 파싱 에러나 결측치가 있는 경우 'N/A'로 채움
    df.fillna({'Standard Mapping': 'N/A', 'Chosen Option': 'N/A', 'Key Signals Used': ''}, inplace=True)
    
    # --- 분석 및 시각화 시작 ---
    print("\n--- Analysis of all scenarios combined ---")

    # A) Overall Ratio
    print("\n=== Overall Standard Mapping Ratio by Problem Type ===")
    df_overall = ratio_table(df, ["problem_type"])
    print(df_overall.round(3).to_string())
    df_overall.to_csv("analysis_overall_ratio.csv")
    
    # B) Ratio by Scenario
    print("\n=== Standard Mapping Ratio by Scenario & Problem Type ===")
    df_scenario_ratio = ratio_table(df, ["scenario", "problem_type"])
    print(df_scenario_ratio.round(3).to_string())
    df_scenario_ratio.to_csv("analysis_scenario_ratio.csv")
    
    # C) Delta by Scenario
    print("\n=== Delta (Specific - Generic) by Scenario ===")
    df_delta_scenario = diff_table(df, ["scenario", "problem_type"])
    print(df_delta_scenario.round(3).to_string())
    df_delta_scenario.to_csv("analysis_delta_scenario.csv")

    # D) Key Signal Usage Analysis
    print("\n=== Key Signal Usage by Scenario & Problem Type ===")
    df['used_any_signal'] = df['Key Signals Used'].apply(lambda x: 1 if pd.notna(x) and x != '' else 0)
    signal_usage = df.groupby(['scenario', 'problem_type'])['used_any_signal'].mean()
    print(signal_usage.round(3).to_string())
    signal_usage.to_csv("analysis_signal_usage.csv")
    
    # E) Plotting (Example for one scenario)
    try:
        scenario_for_plot = df['scenario'].iloc[0]
        df_plot = df[df['scenario'] == scenario_for_plot]
        df_plot_ratio = ratio_table(df_plot, ['problem_type'])
        plot_bar_chart(df_plot_ratio, f"Standard Mapping Ratio for Scenario: {scenario_for_plot}", f"{scenario_for_plot}_ratio")
    except Exception as e:
        print(f"Could not generate plot for a scenario: {e}")

if __name__ == "__main__":
    analyze_results(results_dir="results")