# delta_analysis.py
"""
Delta-specific evaluation script:
- Loads delta results from final_results/summary/
- Runs detailed delta analysis (sensitivity, sign consistency, anchoring, shock size)
- Generates all related plots into final_results/plots/
"""
import os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------
# Save helper
# ---------------------------
def save_fig(fig, title):
    """Saves a matplotlib figure to the plots directory with a safe filename."""
    save_dir = "./final_results/plots"
    os.makedirs(save_dir, exist_ok=True)
    # 파일명 안전하게 치환
    safe_title = (
        title.replace(" ", "_")
             .replace("Δ", "Delta")
             .replace("|", "_")
             .replace("−", "-")
             .replace("(", "")
             .replace(")", "")
    )
    path = os.path.join(save_dir, f"eval_delta_{safe_title}.png")
    fig.savefig(path, dpi=300)

# ---------------------------
# Delta Analysis
# ---------------------------
def load_delta_all(data_dir="./final_results/summary"):
    """
    Loads all delta CSV files and combines them into a single long-format DataFrame.
    """
    all_data = []
    # analysis_delta_*base.csv 파일을 모두 찾아서 로드
    for filepath in glob.glob(os.path.join(data_dir, "analysis_delta_*base.csv")):
        case = os.path.basename(filepath).replace("analysis_delta_", "").replace("-base.csv", "")
        df = pd.read_csv(filepath)
        # Wide-format을 Long-format으로 변환
        long = df.melt(id_vars=["scenario","problem_type"], var_name="strategy", value_name="delta")
        # "Δ StrategyName (...)"에서 전략 이름만 추출
        long["strategy"] = long["strategy"].str.extract(r"Δ (.*?) \(")
        long["case"] = case
        all_data.append(long)
    
    if not all_data:
        print(f"No delta CSV files found in '{data_dir}'.")
        return pd.DataFrame()
        
    return pd.concat(all_data, ignore_index=True)

def compute_delta_metrics(df_delta):
    """
    Computes detailed delta metrics including sensitivity, sign consistency,
    anchoring index, and L2 shock size.
    """
    strategies = ["Maintain", "Retrenchment", "Niche Focus", "Diversification",
                  "Open Innovation", "Fast Follower", "Technology Leadership"]

    # Sensitivity: Mean absolute delta and variance
    sv = (df_delta
          .groupby(["strategy", "problem_type"])["delta"]
          .agg(mean_abs=lambda x: np.mean(np.abs(x)), var=np.var, mean=np.mean))
    sv = sv.reset_index()

    # Sign Consistency: Proportion of positive deltas
    signc = (df_delta
             .assign(pos=lambda d: (d["delta"] > 0).astype(int))
             .groupby(["strategy", "problem_type"])["pos"]
             .mean()
             .reset_index()
             .rename(columns={"pos": "prop_pos"}))

    # Anchoring Index: Difference in sensitivity (generic vs specific)
    g = sv[sv["problem_type"] == "generic"][["strategy", "mean_abs"]].set_index("strategy")
    s = sv[sv["problem_type"] == "specific"][["strategy", "mean_abs"]].set_index("strategy")
    anch = (g["mean_abs"] - s["mean_abs"]).rename("anchoring_index").reset_index()

    # L2 Shock Size: Overall magnitude of change per case/problem_type
    def l2(v):
        return float(np.sqrt(np.sum(v**2)))
    shock = (df_delta
             .pivot_table(index=["case", "problem_type", "scenario"], columns="strategy",
                          values="delta", aggfunc="mean")
             .fillna(0)
             .apply(lambda row: l2(row.values), axis=1)
             .reset_index()
             .groupby(["case", "problem_type"])[0]
             .mean()
             .rename("l2_shock")
             .reset_index())

    metrics = sv.merge(signc, on=["strategy", "problem_type"], how="left")
    return metrics, anch, shock

def plot_delta_sensitivity(metrics_df):
    """Plots the mean absolute delta for generic vs specific problems."""
    order = ["Maintain", "Retrenchment", "Niche Focus", "Diversification",
             "Open Innovation", "Fast Follower", "Technology Leadership"]
    fig, ax = plt.subplots(figsize=(10, 5))
    subg = metrics_df[metrics_df["problem_type"] == "generic"].set_index("strategy").reindex(order)
    subs = metrics_df[metrics_df["problem_type"] == "specific"].set_index("strategy").reindex(order)
    x = np.arange(len(order)); w = 0.35
    ax.bar(x - w/2, subg["mean_abs"].values, w, label="generic")
    ax.bar(x + w/2, subs["mean_abs"].values, w, label="specific")
    ax.set_xticks(x); ax.set_xticklabels(order, rotation=20, ha="right")
    ax.set_ylabel("Mean |Δ|")
    title = "Sensitivity by Strategy"
    ax.set_title(title)
    ax.legend(); plt.tight_layout()
    save_fig(fig, title)
    plt.show()

def plot_delta_sign_consistency(metrics_df):
    """Plots the proportion of positive deltas for each strategy."""
    order = ["Maintain", "Retrenchment", "Niche Focus", "Diversification",
             "Open Innovation", "Fast Follower", "Technology Leadership"]
    fig, ax = plt.subplots(figsize=(10, 5))
    subg = metrics_df[metrics_df["problem_type"] == "generic"].set_index("strategy").reindex(order)
    subs = metrics_df[metrics_df["problem_type"] == "specific"].set_index("strategy").reindex(order)
    x = np.arange(len(order)); w = 0.35
    ax.bar(x - w/2, subg["prop_pos"].values, w, label="generic")
    ax.bar(x + w/2, subs["prop_pos"].values, w, label="specific")
    ax.set_xticks(x); ax.set_xticklabels(order, rotation=20, ha="right")
    ax.set_ylim(0, 1); ax.axhline(0.5, color="k", lw=1)
    ax.set_ylabel("Proportion of Δ > 0")
    title = "Sign Consistency"
    ax.set_title(title)
    ax.legend(); plt.tight_layout()
    save_fig(fig, title)
    plt.show()

def plot_anchoring_index(anch_df):
    """Plots the anchoring index for each strategy."""
    order = ["Maintain", "Retrenchment", "Niche Focus", "Diversification",
             "Open Innovation", "Fast Follower", "Technology Leadership"]
    anch_df = anch_df.set_index("strategy").reindex(order)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(order)), anch_df["anchoring_index"].values)
    ax.axhline(0, color="k", lw=1)
    ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=20, ha="right")
    title = "Anchoring Index (Generic vs Specific)"
    ax.set_title(title)
    plt.tight_layout()
    save_fig(fig, title)
    plt.show()

def plot_shock_size(shock_df):
    """Plots the overall L2 shock size for each case."""
    cases = shock_df["case"].unique().tolist()
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(cases)); w = 0.35
    g = shock_df[shock_df["problem_type"] == "generic"].set_index("case").reindex(cases)["l2_shock"].values
    s = shock_df[shock_df["problem_type"] == "specific"].set_index("case").reindex(cases)["l2_shock"].values
    ax.bar(x - w/2, g, w, label="generic")
    ax.bar(x + w/2, s, w, label="specific")
    ax.set_xticks(x); ax.set_xticklabels(cases, rotation=10)
    title = "Overall Shock Size (L2 Norm)"
    ax.set_title(title); ax.legend()
    plt.tight_layout()
    save_fig(fig, title)
    plt.show()

# ---------------------------
# Main Execution Block
# ---------------------------
if __name__ == "__main__":
    print("--- Starting detailed delta analysis ---")
    
    # 델타 데이터 로드
    summary_dir = "./final_results/summary"
    delta_df = load_delta_all(summary_dir)
    
    if delta_df.empty:
        print("No delta data to analyze. Exiting.")
    else:
        # 지표 계산
        metrics_delta, anch_df, shock_df = compute_delta_metrics(delta_df)
        
        # 결과 출력 (간단히 확인)
        print("\n[Delta metrics: first rows]\n", metrics_delta.head().round(4))
        print("\n[Anchoring index]\n", anch_df.round(4))
        print("\n[Shock size]\n", shock_df.round(4))
        
        # 그래프 생성
        plot_delta_sensitivity(metrics_delta)
        plot_delta_sign_consistency(metrics_delta)
        plot_anchoring_index(anch_df)
        plot_shock_size(shock_df)
    
    print("\n--- Detailed delta analysis finished ---")