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

def load_delta_data(data_dir="./final_results/summary"):
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


def plot_delta_by_case(df_summary):
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
    suptitle = "Generic vs Specific by Case (identical axes)"
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

if __name__ == "__main__":
    df_summary = load_delta_data("./final_results/summary")
    if not df_summary.empty:
        plot_delta_by_case(df_summary)
