# context_dependency_analysis.py

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

valid_strategies = ["Maintain", "Retrenchment", "Niche Focus",
                    "Diversification", "Open Innovation",
                    "Fast Follower", "Technology Leadership"]

# -----------------------------
# 1) Compute Context Dependency Index
# -----------------------------
def compute_context_dependency(input_dir="infer_results"):
    """
    infer_results/*.csv → num_context 기준 전략 비율 요약 및 CDI 반환
    """
    all_files = glob.glob(os.path.join(input_dir, "*scenarios*.csv"))
    if not all_files:
        print(f"No scenario files found in {input_dir}")
        return pd.DataFrame(), pd.DataFrame()

    all_dfs = []
    for file in all_files:
        df = pd.read_csv(file)
        all_dfs.append(df)

    df_combined = pd.concat(all_dfs, ignore_index=True)

    # 전략 비율 (num_context 기준)
    df_ratio = ratio_table(df_combined, ["Num Context"])
    df_ratio = df_ratio.loc[:, df_ratio.columns.isin(valid_strategies)]
    df_ratio = df_ratio[valid_strategies]

    return df_ratio


# -----------------------------
# 2) Plot Context Dependency
# -----------------------------
def plot_context_dependency(df_ratio):
    """
    전략별 num_context 의존성 비율과 CDI 시각화
    """
    save_dir = "./final_results/plots"
    os.makedirs(save_dir, exist_ok=True)

    # Heatmap: num_context vs 전략 비율
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(df_ratio.values, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)

    ax.set_xticks(range(df_ratio.shape[1]))
    ax.set_xticklabels(df_ratio.columns, rotation=25, ha="right")
    ax.set_yticks(range(df_ratio.shape[0]))
    ax.set_yticklabels(df_ratio.index)

    for i in range(df_ratio.shape[0]):
        for j in range(df_ratio.shape[1]):
            ax.text(j, i, f"{df_ratio.values[i, j]:.2f}",
                    va="center", ha="center", fontsize=8)

    title = "Strategy Ratio by Num Context - all Scenarios"
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Ratio")

    fig.tight_layout()
    safe_title = title.replace(" ", "_")
    save_path = os.path.join(save_dir, f"eval_{safe_title}.png")
    fig.savefig(save_path, dpi=300)
    plt.show()


# -----------------------------
# 1) 엔트로피 계산 함수
# -----------------------------
def entropy(p, eps=1e-12):
    """Shannon entropy for probability vector p"""
    p = np.asarray(p, dtype=float) + eps
    p /= p.sum()
    return -np.sum(p * np.log(p))

# -----------------------------
# 2) 전략 다양성 계산
# -----------------------------
def compute_context_entropy(input_dir="infer_results"):
    """
    infer_results/*.csv → num_context별 전략 분포 엔트로피 계산
    """
    all_files = glob.glob(os.path.join(input_dir, "*scenarios*.csv"))
    if not all_files:
        print(f"No scenario files found in {input_dir}")
        return pd.DataFrame()

    all_dfs = []
    for file in all_files:
        df = pd.read_csv(file)
        all_dfs.append(df)

    df_combined = pd.concat(all_dfs, ignore_index=True)

    # 전략 비율 테이블
    df_ratio = ratio_table(df_combined, ["Num Context"])
    df_ratio = df_ratio.loc[:, df_ratio.columns.isin(valid_strategies)]
    df_ratio = df_ratio[valid_strategies]

    # 각 Num Context에서 엔트로피 계산
    entropy_scores = df_ratio.apply(lambda row: entropy(row.values), axis=1)

    df_entropy = pd.DataFrame({
        "Num Context": df_ratio.index,
        "Entropy": entropy_scores
    }).set_index("Num Context")

    print("\n=== Entropy by Num Context ===")
    print(df_entropy.round(3).to_string())

    return df_entropy


# -----------------------------
# 3) 시각화
# -----------------------------
def plot_context_entropy(df_entropy):
    """
    컨텍스트 수 vs 전략 다양성 (Entropy)
    """
    save_dir = "./final_results/plots"
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7,5))
    ax.plot(df_entropy.index, df_entropy["Entropy"], marker="o")
    ax.set_xlabel("Num Context")
    ax.set_ylabel("Entropy (Strategy Diversity)")
    ax.set_title("Strategy Diversity by Context Count - all Scenarios")

    # 저장
    safe_title = "Context_Strategy_Entropy"
    save_path = os.path.join(save_dir, f"eval_{safe_title}.png")
    fig.savefig(save_path, dpi=300)

    plt.show()


def compute_dynamic_context_all(input_dir="infer_results"):
    """
    base 시나리오 내 모든 전략에 대해 Num Context별 비율 변화와 Instability 지표 계산
    """
    all_files = glob.glob(os.path.join(input_dir, "*scenarios*.csv"))
    if not all_files:
        print(f"No scenario files found in {input_dir}")
        return pd.DataFrame(), {}

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

    # base 시나리오만 필터링
    df_combined = pd.concat(all_dfs, ignore_index=True)
    df_combined = df_combined[df_combined["scenario_type"] == "base"]

    # 전략 비율 (Num Context 기준)
    df_ratio = ratio_table(df_combined, ["Num Context"])
    df_ratio = df_ratio.loc[:, df_ratio.columns.isin(valid_strategies)]
    df_ratio = df_ratio[valid_strategies].reset_index()

    # 전략별 Instability 계산
    instability = {}
    for strat in valid_strategies:
        vals = df_ratio[strat].values
        instability[strat] = {
            "std": np.std(vals),
            "range": np.max(vals) - np.min(vals)
        }

    # 종합 지표 (평균)
    overall_instability = {
        "avg_std": np.mean([instability[s]["std"] for s in valid_strategies]),
        "avg_range": np.mean([instability[s]["range"] for s in valid_strategies])
    }

    print("\n=== Dynamic Context Effect (All Strategies) ===")
    print(df_ratio.round(3).to_string(index=False))
    print("\nInstability by Strategy:")
    for strat, vals in instability.items():
        print(f"{strat:22s} std={vals['std']:.3f}, range={vals['range']:.3f}")

    print("\nOverall Instability Score:")
    print(f"Average std   = {overall_instability['avg_std']:.3f}")
    print(f"Average range = {overall_instability['avg_range']:.3f}")

    return df_ratio, instability, overall_instability


def plot_dynamic_context_heatmap(df_ratio_base):
    """
    base 시나리오 내 모든 전략에 대해모든 전략에 대해 Num Context × 전략 비율을 히트맵으로 시각화
    """
    save_dir = "./final_results/plots"
    os.makedirs(save_dir, exist_ok=True)

    # Num Context를 인덱스로 설정
    if "Num Context" in df_ratio_base.columns:
        df_ratio_base = df_ratio_base.set_index("Num Context")

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(df_ratio_base.values, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)

    # 축 라벨
    ax.set_xticks(range(len(df_ratio_base.columns)))
    ax.set_xticklabels(df_ratio_base.columns, rotation=25, ha="right")
    ax.set_yticks(range(len(df_ratio_base.index)))
    ax.set_yticklabels(df_ratio_base.index)

    # 셀 값 표시
    for i in range(df_ratio_base.shape[0]):
        for j in range(df_ratio_base.shape[1]):
            ax.text(j, i, f"{df_ratio_base.values[i, j]:.2f}",
                    ha="center", va="center", fontsize=8)

    ax.set_xlabel("Strategy")
    ax.set_ylabel("Num Context")
    ax.set_title("Strategy Ratio by Num Context - base Scenario")

    # 컬러바
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Ratio")

    # 저장
    save_path = os.path.join(save_dir, "eval_Dynamic_Context_All_Strategies_Heatmap.png")
    fig.savefig(save_path, dpi=300)

    plt.show()


def plot_strategy_boxplot(df_ratio_base):
    """
    Boxplot: 전략별 비율 분포 (Num Context 변화 반영)
    """
    save_dir = "./final_results/plots"
    os.makedirs(save_dir, exist_ok=True)

    df_melted = df_ratio_base.reset_index(drop = True).melt(
        id_vars="Num Context", var_name="Strategy", value_name="Ratio"
    )

    fig, ax = plt.subplots(figsize=(10,6))
    df_melted.boxplot(
        column="Ratio", by="Strategy", ax=ax, grid=False, showmeans=True
    )
    ax.set_title("Strategy Ratio Distribution by Context for base Scenario")
    ax.set_xlabel("Strategy")   # ✅ x축 라벨 명시
    ax.set_ylabel("Ratio")
    plt.suptitle("")  # 상단 자동 제목 제거

    save_path = os.path.join(save_dir, "eval_strategy_boxplot.png")
    fig.savefig(save_path, dpi=300)
    plt.show()

def plot_trend_with_instability(df_ratio_base, instability):
    """
    누적 Δ (Trend) + Instability (std 범위 음영) 시각화
    """
    save_dir = "./final_results/plots"
    os.makedirs(save_dir, exist_ok=True)

    # Δ 누적합 계산
    df_delta = df_ratio_base.set_index("Num Context").diff().fillna(0)
    df_cumdelta = df_delta.cumsum().reset_index()

    fig, ax = plt.subplots(figsize=(12, 6))

    # Line plot + Std shading
    for strat in valid_strategies:
        y = df_cumdelta[strat]
        std_val = instability[strat]["std"]  # 전략별 std 값

        # 누적 Δ 라인
        ax.plot(
            df_cumdelta["Num Context"],
            y,
            marker="o",
            label=strat
        )

        # std 기반 음영 (± std)
        ax.fill_between(
            df_cumdelta["Num Context"],
            y - std_val,
            y + std_val,
            alpha=0.2
        )

    ax.set_xlabel("Num Context")
    ax.set_ylabel("Cumulative Δ (Trend)")
    ax.set_title("Base Scenario: Strategy Trends with Instability (±Std)")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    fig.tight_layout()
    save_path = os.path.join(save_dir, "eval_trend_with_instability_shaded.png")
    fig.savefig(save_path, dpi=300)
    plt.show()





# -----------------------------
# 3) Main
# -----------------------------
if __name__ == "__main__":
    
    # 컨텍스트 개수에 따른 전략 선택 비율
    df_ratio = compute_context_dependency("infer_results")
    if not df_ratio.empty:
        plot_context_dependency(df_ratio)

    # 컨텍스트 개수에 따른 전략 다양성 (엔트로피)
    df_entropy = compute_context_entropy("infer_results")
    if not df_entropy.empty:
        plot_context_entropy(df_entropy)

    # base 시나리오의 모든 전략에 대해 컨텍스트 개수 변화에 따른 비율 변화 및 불안정성 지표
    # df_ratio_base, instability, overall_instability = compute_dynamic_context_all("infer_results")
    # if not df_ratio_base.empty:
    #     plot_dynamic_context_heatmap(df_ratio_base)
    #     plot_strategy_boxplot(df_ratio_base)      # Boxplot
    #     plot_trend_with_instability(df_ratio_base, instability)

