# scenario_eval.py

import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import shutil # shutil 모듈 추가
import pandas as pd
import json
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
    """
    이미 통합된 combined_analysis 폴더의 모든 *_all.csv 파일을 사용하여 분석을 수행합니다.
    """
    print("--- Starting combined analysis from pre-existing 'combined_analysis' directory ---")

    output_dir = "combined_analysis"
    plots_dir = os.path.join(output_dir, "plots")

    # combined_analysis 폴더가 없거나, 분석에 필요한 파일이 없으면 종료
    if not os.path.exists(output_dir):
        print(f"Directory '{output_dir}' not found. Please ensure the data consolidation step has been completed.")
        return

    # combined_analysis 폴더의 모든 *_all.csv 파일을 읽어와 하나의 데이터프레임으로 결합
    all_combined_files = glob.glob(os.path.join(output_dir, "*_all.csv"))
    if not all_combined_files:
        print(f"No '_all.csv' files found in '{output_dir}'. Exiting.")
        return

    key_columns = ['scenario', 'problem_type', 'repeat', 'Model', 'Temperature',
       'Max Tokens', 'Context Tags', 'Num Context', 'Prompt', 'Raw Output',
       'Parse Error', 'Standard Mapping', 'Rationale','Key Signals Used']
    
    standardization_map = {
    'C → Open Innovation': 'Open Innovation',
    'A → Technology Leadership': 'Technology Leadership',
    'C -> Open Innovation': 'Open Innovation',
    'A -> Technology Leadership': 'Technology Leadership'
}
        
    all_data_frames = []
    for file_path in all_combined_files:
        df = pd.read_csv(file_path)
        df = df[key_columns]

        # Standardize the 'Standard Mapping' column
        df['Standard Mapping'].replace(standardization_map, inplace=True)
        df['Standard Mapping'].fillna('N/A', inplace=True)

        # 파일 이름에서 시나리오 유형을 추출 (e.g., results_count_fact_all.csv -> count_fact)
        file_name = os.path.basename(file_path)
        if file_name == "results_all.csv":
             scenario_type = 'base'
        else:
             scenario_type = file_name.replace('results_', '').replace('_all.csv', '')

        df['scenario_type'] = scenario_type

        all_data_frames.append(df)

    if not all_data_frames:
        print("No data to analyze. Exiting.")
        return

    df_combined = pd.concat(all_data_frames, ignore_index=True)
    df_combined.fillna({'Standard Mapping': 'N/A', 'Chosen Option': 'N/A', 'Key Signals Used': ''}, inplace=True)
    
    print(f"\n--- Combined analysis for all scenarios is running. Results will be saved in '{output_dir}' ---")

    # --- 기존 분석 로직 그대로 유지 ---
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

    return df_combined

#%%
tmp = analyze_all_scenarios()
#%% : 컨텍스트 없을 때, 기본 확률 분포 계산

# 1) 시나리오 정의 JSON 불러오기
with open("scenarios.json", "r", encoding="utf-8") as f:
    scenarios = json.load(f)

# 2) 시나리오별 valid 전략 집합 만들기
valid_strategies = {
    s: set(v["mapping"] for v in data["execution_options"].values())
    for s, data in scenarios.items()
}

# 3) 함수 정의: 보정된 확률 계산
def corrected_distribution(df, scenario_name):
    allowed = valid_strategies[scenario_name]
    subset = df[(df["Num Context"] == 0) & (df["scenario"] == scenario_name)]
    subset = subset[subset["Standard Mapping"].isin(allowed)]  # 불필요한 전략 제거
    return subset["Standard Mapping"].value_counts(normalize=True)

# 4) 모든 시나리오에 대해 계산
corrected_results = {
    s: corrected_distribution(tmp, s)
    for s in tmp["scenario"].unique()
}

corrected_df = pd.DataFrame(corrected_results).fillna(0).T
print(corrected_df)

corrected_df.mean()

#%% 모델 별 확률 분포 계산
# 1) 각 시나리오별 허용된 전략 정의 (이미 있으시겠죠)
valid_strategies = {
    "1_founder_period": ["Fast Follower", "Technology Leadership", "Open Innovation"],
    "2_roadster_launch": ["Technology Leadership", "Fast Follower", "Open Innovation"],
    "3_model_s_launch": ["Technology Leadership", "Fast Follower", "Open Innovation"],
    "4_model_x_launch": ["Technology Leadership", "Niche Focus", "Maintain"],
    "5_model_3_mass_market": ["Technology Leadership", "Maintain", "Open Innovation"],
    "6_energy_infra": ["Technology Leadership", "Technology Leadership", "Diversification", "Retrenchment"],
}

# 2) 보정된 분포 계산: 모델별 + 시나리오별
counts = []
for (model, scenario), subset in tmp[tmp["Num Context"] == 0].groupby(["Model", "scenario"]):
    allowed = valid_strategies[scenario]
    # 허용된 전략만 남김
    filtered = subset[subset["Standard Mapping"].isin(allowed)]
    # 분포 계산
    dist = filtered["Standard Mapping"].value_counts(normalize=True)
    dist.name = (model, scenario)
    counts.append(dist)

# 3) DataFrame으로 합치기
corrected_df_model = pd.DataFrame(counts).fillna(0)

# 4) 모델 단위로 평균 (시나리오별 균등 가중치)
model_level_distribution = corrected_df_model.groupby(level=0).mean()

print(model_level_distribution)

#%% laama가 tl을 압도적으로 선택하는 이유 파악
# 모델 이름 (데이터셋에 맞게 확인 필요)
LLAMA   = "meta-llama/Meta-Llama-3.1-8B-Instruct"
MISTRAL = "mistralai/Mistral-7B-Instruct-v0.3"
QWEN    = "Qwen/Qwen2.5-14B-Instruct"

# pick_mode: 같은 조건에서 여러 row 있으면 최빈값, 없으면 첫값
def pick_mode(x):
    m = x.mode()
    return m.iloc[0] if not m.empty else x.iloc[0]

# tmp: 기존 분석 결과 DataFrame이 있다고 가정
# tmp = pd.read_csv("your_results.csv")


noctx = tmp[tmp["Num Context"] == 0].copy()
idx_cols = ["scenario", "problem_type", "scenario_type"]

# 전략 선택 피벗
choice_pivot = noctx.pivot_table(
    index=idx_cols, columns="Model", values="Standard Mapping", aggfunc=pick_mode
)
# 라쇼날 피벗
rat_pivot = noctx.pivot_table(
    index=idx_cols, columns="Model", values="Rationale", aggfunc=pick_mode
)
# 키 시그널 피벗
keysig_pivot = noctx.pivot_table(
    index=idx_cols, columns="Model", values="Key Signals Used", aggfunc=pick_mode
)

is_llama_tl = (choice_pivot.get(LLAMA) == "Technology Leadership")
is_mistral_not_tl = choice_pivot.get(MISTRAL).notna() & (choice_pivot.get(MISTRAL) != "Technology Leadership")
is_qwen_not_tl    = choice_pivot.get(QWEN).notna()    & (choice_pivot.get(QWEN)    != "Technology Leadership")
others_not_tl = is_mistral_not_tl | is_qwen_not_tl

mask = is_llama_tl & others_not_tl
cases_idx = choice_pivot.index[mask]

view_choice = choice_pivot.loc[cases_idx, [LLAMA, MISTRAL, QWEN]].rename(
    columns={LLAMA: "Choice (LLaMA)", MISTRAL: "Choice (Mistral)", QWEN: "Choice (Qwen)"}
)
view_rat = rat_pivot.loc[cases_idx, [LLAMA, MISTRAL, QWEN]].rename(
    columns={LLAMA: "Rationale (LLaMA)", MISTRAL: "Rationale (Mistral)", QWEN: "Rationale (Qwen)"}
)
view_keys = keysig_pivot.loc[cases_idx, [LLAMA, MISTRAL, QWEN]].rename(
    columns={LLAMA: "KeySignals (LLaMA)", MISTRAL: "KeySignals (Mistral)", QWEN: "KeySignals (Qwen)"}
)

llama_vs_others = (
    view_choice
    .join(view_rat, how="left")
    .join(view_keys, how="left")
    .sort_index()
)

# 확인
print("🔎 LLaMA=TL, Others≠TL 케이스 샘플")
print(llama_vs_others.head(10))

print("📊 Scenario_type별 분포")
print(llama_vs_others.reset_index()["scenario_type"].value_counts())

# 다른 모델들이 TL이 아닌 경우 뭘 선택했는지
others_stack = []
if MISTRAL in choice_pivot.columns:
    m_ = choice_pivot.loc[cases_idx, MISTRAL]
    others_stack.append(m_[m_.notna() & (m_ != "Technology Leadership")])
if QWEN in choice_pivot.columns:
    q_ = choice_pivot.loc[cases_idx, QWEN]
    others_stack.append(q_[q_.notna() & (q_ != "Technology Leadership")])

others_series = pd.concat(others_stack)
print("📊 다른 모델들이 선택한 전략 분포")
print(others_series.value_counts(normalize=True).round(3))

def nonempty(x):
    if pd.isna(x): return False
    s = str(x).strip()
    return len(s) > 0 and s != "[]"

print("📋 Non-context인데 key_signals가 비어있지 않은 비율")
for tag_col in ["KeySignals (LLaMA)", "KeySignals (Mistral)", "KeySignals (Qwen)"]:
    viol = llama_vs_others[tag_col].dropna().apply(nonempty).mean()
    print(f"{tag_col}: {viol:.3f}")


#%%

if __name__ == "__main__":
    tmp = analyze_all_scenarios()



