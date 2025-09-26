import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os


# 허용된 전략 목록 정의
valid_strategies = [
    "Maintain", "Retrenchment", "Niche Focus",
    "Diversification", "Open Innovation",
    "Fast Follower", "Technology Leadership"
]


# -------------------------------------
# Helper: 전략 비율 테이블
# -------------------------------------
def ratio_table(df, keys):
    counts = (
        df.groupby(keys + ["Standard Mapping"])
        .size()
        .rename("Count")
        .reset_index()
    )
    totals = counts.groupby(keys)["Count"].transform("sum")
    counts["Ratio"] = counts["Count"] / totals
    return counts.pivot(index=keys, columns="Standard Mapping", values="Ratio").fillna(0)


# -------------------------------------
# 1) Temperature별 전략 비율 요약
# -------------------------------------
def compute_temp_effects(input_dir="infer_results"):
    """
    infer_results/*.csv → temperature별 전략 비율 요약 반환
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

    # 전략 비율 요약 (Temperature 기준)
    df_temp = ratio_table(df_combined, ["scenario_type","Temperature"])

    # 유효한 전략만 남기고 순서 맞추기
    df_temp = df_temp.loc[:, df_temp.columns.isin(valid_strategies)]
    df_temp = df_temp[valid_strategies]

    print("\n=== Strategy ratio by Temperature ===")
    print(df_temp.round(3).to_string())
    return df_temp



# -------------------------------------
# 2) Temperature 효과 히트맵
# -------------------------------------
def plot_temp_heatmap(df_temp):
    """
    Temperature별 전략 비율 차이를 히트맵으로 시각화 (0.7 - 0, 시나리오별)
    """
    save_dir = "./final_results/plots"
    os.makedirs(save_dir, exist_ok=True)

    # temp=0, 0.7로 나눠서 정리
    df_0   = df_temp.xs(0.0, level="Temperature")
    df_07  = df_temp.xs(0.7, level="Temperature")

    # 차이 (0.7 - 0)
    df_diff = df_07 - df_0

    print("\n=== Δ Ratio (Temp 0.7 - 0) by Scenario ===")
    print(df_diff.round(3).to_string())

    # 히트맵 그리기
    fig, ax = plt.subplots(figsize=(10,6))
    vmax = np.abs(df_diff.values).max()
    im = ax.imshow(df_diff.values, aspect="auto", cmap="bwr", vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(df_diff.shape[1]))
    ax.set_xticklabels(df_diff.columns, rotation=25, ha="right")
    ax.set_yticks(range(df_diff.shape[0]))
    ax.set_yticklabels(df_diff.index)

    # 값 직접 표시
    for i in range(df_diff.shape[0]):
        for j in range(df_diff.shape[1]):
            ax.text(j, i, f"{df_diff.values[i,j]:.2f}", ha="center", va="center", fontsize=9)

    title = "Temperature Effect by Scenario (0.7 - 0)"
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Δ Ratio (0.7 - 0)")

    fig.tight_layout()

    # 저장
    safe_title = title.replace(" ", "_")
    save_path = os.path.join(save_dir, f"eval_{safe_title}.png")
    fig.savefig(save_path, dpi=300)

    plt.show()
    return df_diff

import plotly.graph_objects as go

def plot_temp_transition_sankey(df_temp, scenario="base"):
    """
    특정 시나리오에서 Temperature 0 → 0.7 전략 분포 이동을 Sankey 다이어그램으로 시각화
    """
    # temp=0, 0.7 분리
    df0 = df_temp.xs(0.0, level="Temperature").loc[scenario]
    df07 = df_temp.xs(0.7, level="Temperature").loc[scenario]

    # 전략 목록
    strategies = df0.index.tolist()

    # Δ 비율 계산
    diff = df07 - df0

    # 감소 전략 → 증가 전략 연결
    flows = []
    for s_from in strategies:
        if diff[s_from] < 0:  # 감소
            for s_to in strategies:
                if diff[s_to] > 0:  # 증가
                    flow_val = min(-diff[s_from], diff[s_to])  # 분배
                    if flow_val > 0:
                        flows.append((s_from, s_to, flow_val))
                        diff[s_from] += flow_val
                        diff[s_to] -= flow_val

    # Sankey 구성
    labels = [f"{s} (T=0)" for s in strategies] + [f"{s} (T=0.7)" for s in strategies]
    source, target, value = [], [], []
    for f, t, v in flows:
        source.append(strategies.index(f))               # T=0 위치
        target.append(len(strategies) + strategies.index(t))  # T=0.7 위치
        value.append(v)

    # 다이어그램
    fig = go.Figure(go.Sankey(
        node=dict(
            pad=20, thickness=20,
            line=dict(color="black", width=0.5),
            label=labels
        ),
        link=dict(
            source=source,
            target=target,
            value=value
        )
    ))

    fig.update_layout(title_text=f"Strategy Transition (Temp 0 → 0.7) | {scenario}", font_size=12)
    fig.show()

def run_all_temp_transitions(df_temp):
    """
    df_temp 내 모든 scenario_type에 대해 Sankey 다이어그램 생성
    """
    scenarios = df_temp.index.get_level_values("scenario_type").unique()

    for scen in scenarios:
        print(f"\n--- Plotting Sankey for scenario: {scen} ---")
        try:
            plot_temp_transition_sankey(df_temp, scenario=scen)
        except Exception as e:
            print(f"Failed for {scen}: {e}")


def entropy(p, eps=1e-12):
    """Shannon entropy for a probability distribution p"""
    p = np.asarray(p, dtype=float) + eps
    p = p / p.sum()
    return -np.sum(p * np.log(p))

def compute_entropy_by_temp(df_temp):
    """
    df_temp (scenario_type × Temperature × 전략 비율) → 
    시나리오별 temp=0 vs 0.7 엔트로피와 차이 반환
    """
    results = []
    scenarios = df_temp.index.get_level_values("scenario_type").unique()

    for scen in scenarios:
        df0 = df_temp.loc[(scen, 0.0)]
        df07 = df_temp.loc[(scen, 0.7)]
        H0 = entropy(df0.values)
        H07 = entropy(df07.values)
        results.append({
            "scenario_type": scen,
            "entropy_temp0": H0,
            "entropy_temp07": H07,
            "delta_entropy": H07 - H0
        })

    return pd.DataFrame(results)


# -------------------------------------
# 실행 (메인)
# -------------------------------------
if __name__ == "__main__":

    # Temperature 분석
    temp_df = compute_temp_effects("infer_results")
    if not temp_df.empty:
        plot_temp_heatmap(temp_df)
        #run_all_temp_transitions(temp_df)
        df_entropy = compute_entropy_by_temp(temp_df)
        print("\n=== Entropy by Temperature ===")
        print(df_entropy.round(3).to_string(index=False))
