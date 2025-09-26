# eval_overall_eda.py
import os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------
# Utilities
# ---------------------------
def js_divergence(p, q, eps=1e-12):
    """Jensen-Shannon divergence for two probability vectors (safe for zeros)."""
    p = np.asarray(p, dtype=float) + eps
    q = np.asarray(q, dtype=float) + eps
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    def kl(a, b):
        return np.sum(a * np.log(a / b))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)

def entropy(p, eps=1e-12):
    p = np.asarray(p, dtype=float) + eps
    p = p / p.sum()
    return -np.sum(p * np.log(p))

# ---------------------------
# Save helper
# ---------------------------
def save_fig(fig, title):
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
    path = os.path.join(save_dir, f"eval_eda_{safe_title}.png")
    fig.savefig(path, dpi=300)

# ---------------------------
# A) Overall ratio EDA
# ---------------------------
def load_overall_ratio(csv_path="./final_results/summary/analysis_overall_ratio.csv"):
    df = pd.read_csv(csv_path)
    df = df.set_index("scenario_type") # 유용한 전략 옵션만 가지고 옴!
    cols = ["Maintain", "Retrenchment", "Niche Focus","Diversification","Open Innovation","Fast Follower","Technology Leadership"]
    df = df[cols]
    return df

def plot_ratio_heatmap(df):
    fig, ax = plt.subplots(figsize=(10,6))
    im = ax.imshow(df.values, aspect="auto", cmap="YlGnBu", vmin=0, vmax=df.values.max())
    ax.set_xticks(range(df.shape[1])); ax.set_xticklabels(df.columns, rotation=25, ha="right")
    ax.set_yticks(range(df.shape[0])); ax.set_yticklabels(df.index)
    title = "Strategy Ratio by Scenario"
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax); cbar.set_label("Strategy Ratio")
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            ax.text(j, i, f"{df.values[i,j]:.2f}", va="center", ha="center", fontsize=9)
    fig.tight_layout()
    save_fig(fig, title)
    plt.show()

def plot_ratio_heatamp_delta_from_base(df):
    base = df.loc["base"].values
    delta = df.copy()
    for idx in df.index:
        delta.loc[idx] = df.loc[idx].values - base
    fig, ax = plt.subplots(figsize=(10,6))
    vmax = np.abs(delta.values).max()
    im = ax.imshow(delta.values, aspect="auto", cmap="bwr", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(delta.shape[1])); ax.set_xticklabels(delta.columns, rotation=25, ha="right")
    ax.set_yticks(range(delta.shape[0])); ax.set_yticklabels(delta.index)
    title = "Delta Strategy Ratio vs Base"
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax); cbar.set_label("Δ ratio")
    for i in range(delta.shape[0]):
        for j in range(delta.shape[1]):
            ax.text(j, i, f"{delta.values[i,j]:.02f}", va="center", ha="center", fontsize=9)
    fig.tight_layout()
    save_fig(fig, title)
    plt.show()

def compute_ratio_metrics(df):
    base = df.loc["base"]
    metrics = []
    for scen, row in df.iterrows():
        H = entropy(row.values)
        jsd = js_divergence(row.values, base.values)
        rho = pd.Series(row.values).rank().corr(pd.Series(base.values).rank(), method="spearman")
        metrics.append({"scenario": scen, "entropy": H, "jsd_from_base": jsd, "spearman_vs_base": rho})
    return pd.DataFrame(metrics).set_index("scenario")

def plot_ratio_metrics(metrics_df):
    fig, axes = plt.subplots(1, 3, figsize=(14,8))
    axes[0].bar(metrics_df.index, metrics_df["entropy"])
    axes[0].set_title("Entropy (concentration↓)"); axes[0].tick_params(axis='x', rotation=25)
    axes[1].bar(metrics_df.index, metrics_df["jsd_from_base"])
    axes[1].set_title("JSD from Base (divergence)"); axes[1].tick_params(axis='x', rotation=25)
    axes[2].bar(metrics_df.index, metrics_df["spearman_vs_base"])
    axes[2].set_ylim(-1,1); axes[2].axhline(0, color="k", lw=1)
    axes[2].set_title("Rank corr vs Base"); axes[2].tick_params(axis='x', rotation=25)
    title = "Overall Ratio Metrics"
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0,0,1,0.93])
    save_fig(fig, title)
    plt.show()

def plot_ratio_pca(df):
    X = df.values
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    Z = U[:, :2] * S[:2]
    fig, ax = plt.subplots(figsize=(6,5))
    ax.axhline(0, color="k", lw=1); ax.axvline(0, color="k", lw=1)
    for (x,y), label in zip(Z, df.index):
        ax.scatter(x, y)
        ax.text(x, y, f" {label}", va="center", ha="left")
    title = "PCA of Strategy Ratios 2D"
    ax.set_title(title)
    plt.tight_layout()
    save_fig(fig, title)
    plt.show()

# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    # A) Overall ratios
    ratio_df = load_overall_ratio("./final_results/summary/analysis_overall_ratio.csv")
    plot_ratio_heatmap(ratio_df)
    plot_ratio_heatamp_delta_from_base(ratio_df)
    metrics_ratio = compute_ratio_metrics(ratio_df)
    print("\n[Overall ratio metrics]\n", metrics_ratio.round(4))
    plot_ratio_metrics(metrics_ratio)
    plot_ratio_pca(ratio_df)