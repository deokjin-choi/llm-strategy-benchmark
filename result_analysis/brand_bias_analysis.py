# generic vs specific 케이스별 차이 분석
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from scipy.stats import pearsonr

# 허용된 전략 목록 정의
valid_strategies = [
    "Maintain", "Retrenchment", "Niche Focus",
    "Diversification", "Open Innovation",
    "Fast Follower", "Technology Leadership"
]

def load_delta(data_dir="./final_results/summary"):
    all_data = []
    for filepath in glob.glob(os.path.join(data_dir, "analysis_delta_*base.csv")):
        case = os.path.basename(filepath).replace("analysis_delta_", "").replace("-base.csv", "")
        df = pd.read_csv(filepath)

        # Wide → Long 변환
        df_melted = df.melt(
            id_vars=["scenario", "problem_type"],
            var_name="strategy",
            value_name="delta"
        )

        # "Δ StrategyName (xxx)" → StrategyName만 추출
        df_melted["strategy"] = df_melted["strategy"].str.extract(r"Δ (.*?) \(")

        # 유효한 전략만 남기기
        df_melted = df_melted[df_melted["strategy"].isin(valid_strategies)]

        df_melted["case"] = case
        all_data.append(df_melted)

    if not all_data:
        print("No delta CSV files found.")
        return pd.DataFrame()

    df_all = pd.concat(all_data, ignore_index=True)

    df_summary = df_all.pivot_table(
        index=["case", "strategy"],
        columns="problem_type",
        values="delta",
        aggfunc="mean"
    ).reset_index()

    df_summary = df_summary.rename_axis(None, axis=1)
    return df_summary

def plot_generic_vs_specific_delta(df_summary):
    # Strategy order
    strategy_order = valid_strategies
    df_summary["strategy"] = pd.Categorical(df_summary["strategy"], categories=strategy_order, ordered=True)
    df_summary = df_summary.sort_values(["case", "strategy"])

    # Global y-limits
    ymin = float(min(df_summary["generic"].min(), df_summary["specific"].min()))
    ymax = float(max(df_summary["generic"].max(), df_summary["specific"].max()))
    pad = (ymax - ymin) * 0.1 if ymax > ymin else 0.01
    ylims = (ymin - pad, ymax + pad)

    cases = ["competitive_dynamics", "count_fact", "opp_focus", "randomized_numbers"]

    # 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
    axes = axes.flatten()

    for i, case in enumerate(cases):
        sub = df_summary[df_summary["case"] == case].copy()
        x = np.arange(len(strategy_order))
        width = 0.35

        # case+strategy 조합을 유일하게 만들기
        sub = (
            sub.groupby("strategy")[["generic", "specific"]]
            .mean()
            .reindex(strategy_order)
        )

        gen = sub["generic"].values
        spec = sub["specific"].values

        ax = axes[i]
        ax.bar(x - width/2, gen, width, label="generic")
        ax.bar(x + width/2, spec, width, label="specific")
        ax.axhline(0, linewidth=1, color="black")
        ax.set_xticks(x)
        ax.set_xticklabels(strategy_order, rotation=20, ha="right")
        ax.set_ylim(*ylims)
        ax.set_title(case)

    # Add legend to last subplot
    axes[-1].legend()
    suptitle = "Generic and Specific Δ by Scenario"
    fig.suptitle(suptitle, fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    # 저장 경로
    save_dir = "./final_results/plots"
    os.makedirs(save_dir, exist_ok=True)
    safe_title = suptitle.replace(" ", "_")
    save_path = os.path.join(save_dir, f"eval_{safe_title}.png")
    fig.savefig(save_path, dpi=300)

    # 출력
    plt.show()

def plot_correlation_generic_vs_specific(df_summary):
    """
    Plots the correlation between generic and specific deltas for each case.
    
    Args:
        df_summary (pd.DataFrame): A DataFrame containing 'case', 'strategy',
                                   'generic', and 'specific' delta values.
    """
    print("\n--- Plotting Correlation between Generic and Specific Deltas ---")
    
    # 필수 열이 있는지 확인
    required_cols = ['case', 'strategy', 'generic', 'specific']
    if not all(col in df_summary.columns for col in required_cols):
        print(f"Error: Required columns {required_cols} not found in DataFrame.")
        return

    cases = df_summary['case'].unique().tolist()
    if not cases:
        print("No cases found for correlation analysis.")
        return

    # 그래프 레이아웃 설정
    fig_cols = 2
    fig_rows = (len(cases) + fig_cols - 1) // fig_cols
    fig, axes = plt.subplots(fig_rows, fig_cols, figsize=(5 * fig_cols, 5 * fig_rows), squeeze=False)
    axes = axes.flatten()

    for i, case in enumerate(cases):
        ax = axes[i]
        
        case_data = df_summary[df_summary['case'] == case].copy()
        
        # 데이터가 충분하지 않으면 건너뛰기
        if len(case_data) < 2:
            ax.set_title(f"Scenario: {case} (Not enough data)")
            ax.set_xlabel("Generic Δ"); ax.set_ylabel("Specific Δ")
            continue
            
        x = case_data['generic']
        y = case_data['specific']
        
        # 상관계수 계산
        corr_coef, _ = pearsonr(x, y)
        
        # 산점도 그리기
        ax.scatter(x, y, alpha=0.6, s=50)
        
        # 상관계수 텍스트 추가
        ax.text(0.05, 0.95, f'r = {corr_coef:.2f}', 
                transform=ax.transAxes, fontsize=12, verticalalignment='top')
        
        # 제목 및 축 레이블 설정
        ax.set_title(f"Scenario: {case}", fontsize=14)
        ax.set_xlabel("Generic Δ")
        ax.set_ylabel("Specific Δ")
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
        ax.axvline(0, color='gray', linestyle='--', linewidth=0.5)
        ax.grid(True, linestyle='--', alpha=0.5)
        
    # 남는 서브플롯 숨기기
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
        
    suptitle = "Correlation between Generic and Specific Δ by Scenario"
    fig.suptitle(suptitle, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # 그래프 저장
    save_dir = "./final_results/plots"
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(os.path.join(save_dir, "eval_case_correlation_deltas.png"), dpi=300)
    plt.show()


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

def compute_brand_bias_by_scenario(input_dir="infer_results"):
    """
    infer_results/*.csv → scenario_type × problem_type 비율 요약 반환
    """
    all_files = glob.glob(os.path.join(input_dir, "*scenarios*.csv"))
    if not all_files:
        print(f"No scenario files found in {input_dir}")
        return pd.DataFrame()

    all_dfs = []
    for file in all_files:
        df = pd.read_csv(file)

        # 파일명에서 시나리오 타입 추출
        file_name = os.path.basename(file).replace(".csv", "")
        if file_name.endswith("_scenarios"):
            scenario_type = "base"
        else:
            scenario_type = file_name.split("_", 1)[-1]
            if scenario_type.startswith("scenarios_"):
                scenario_type = scenario_type.replace("scenarios_", "")
        df["scenario_type"] = scenario_type

        all_dfs.append(df)

    df_combined = pd.concat(all_dfs, ignore_index=True)

    # 전략 비율 요약 (scenario_type × problem_type 기준)
    df_brand = ratio_table(df_combined, ["scenario_type", "problem_type"])
    df_brand = df_brand.loc[:, df_brand.columns.isin(valid_strategies)]
    df_brand = df_brand[valid_strategies]

    print("\n=== Brand bias summary by scenario (generic vs specific) ===")
    print(df_brand.round(3).to_string())
    return df_brand


def plot_brand_bias_heatmap(df_brand):
    """
    Brand bias 차이를 시나리오별로 히트맵으로 시각화 (specific - generic)
    """
    save_dir = "./final_results/plots"
    os.makedirs(save_dir, exist_ok=True)

    # generic/specific pivot → 차이
    df_generic = df_brand.loc[(slice(None), "generic"), :].droplevel(1)
    df_specific = df_brand.loc[(slice(None), "specific"), :].droplevel(1)
    df_diff = df_specific - df_generic  # specific - generic

    fig, ax = plt.subplots(figsize=(10,6))
    vmax = np.abs(df_diff.values).max()
    im = ax.imshow(df_diff.values, aspect="auto", cmap="bwr", vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(df_diff.shape[1]))
    ax.set_xticklabels(df_diff.columns, rotation=25, ha="right")
    ax.set_yticks(range(df_diff.shape[0]))
    ax.set_yticklabels(df_diff.index)

    for i in range(df_diff.shape[0]):
        for j in range(df_diff.shape[1]):
            ax.text(j, i, f"{df_diff.values[i,j]:.2f}", ha="center", va="center", fontsize=9)

    title = "Brand Bias by Scenario (Specific - Generic)"
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Δ Ratio (specific - generic)")

    fig.tight_layout()

    # 저장
    safe_title = title.replace(" ", "_")
    save_path = os.path.join(save_dir, f"eval_{safe_title}.png")
    fig.savefig(save_path, dpi=300)

    plt.show()
    return df_diff


if __name__ == "__main__":

    # 브랜드 편향 분석
    bias_df = compute_brand_bias_by_scenario("infer_results")
    plot_brand_bias_heatmap(bias_df)

    # generic vs specific 델타 분석
    df_summary = load_delta("./final_results/summary")
    if not df_summary.empty:
        plot_generic_vs_specific_delta(df_summary)
        plot_correlation_generic_vs_specific(df_summary)



