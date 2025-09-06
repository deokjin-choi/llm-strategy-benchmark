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
    save_dir = "./combined_analysis/plots"
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
def load_overall_ratio(csv_path="./combined_analysis/analysis_overall_ratio.csv"):
    df = pd.read_csv(csv_path)
    df = df.set_index("scenario_type")
    cols = ["Diversification","Fast Follower","Maintain","Niche Focus","Open Innovation","Technology Leadership"]
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

def plot_ratio_delta_from_base(df):
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
    fig, axes = plt.subplots(1, 3, figsize=(14,4))
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
# B) Delta EDA
# ---------------------------
def load_delta_all(data_dir="./combined_analysis"):
    all_data = []
    for filepath in glob.glob(os.path.join(data_dir, "analysis_delta_*base.csv")):
        case = os.path.basename(filepath).replace("analysis_delta_", "").replace("-base.csv", "")
        df = pd.read_csv(filepath)
        long = df.melt(id_vars=["scenario","problem_type"], var_name="strategy", value_name="delta")
        long["strategy"] = long["strategy"].str.extract(r"Δ (.*?) \(")
        long["case"] = case
        all_data.append(long)
    return pd.concat(all_data, ignore_index=True)

def compute_delta_metrics(df_delta):
    strategies = ["Diversification","Fast Follower","Maintain","Niche Focus","Open Innovation","Technology Leadership"]
    sv = (df_delta
          .groupby(["strategy","problem_type"])["delta"]
          .agg(mean_abs=lambda x: np.mean(np.abs(x)),
               var=np.var,
               mean=np.mean))
    sv = sv.reset_index()

    signc = (df_delta
             .assign(pos=lambda d: (d["delta"] > 0).astype(int))
             .groupby(["strategy","problem_type"])["pos"]
             .mean()
             .reset_index()
             .rename(columns={"pos":"prop_pos"}))

    g = sv[sv["problem_type"]=="generic"][["strategy","mean_abs"]].set_index("strategy")
    s = sv[sv["problem_type"]=="specific"][["strategy","mean_abs"]].set_index("strategy")
    anch = (g["mean_abs"] - s["mean_abs"]).rename("anchoring_index").reset_index()

    def l2(v): return float(np.sqrt(np.sum(v**2)))
    shock = (df_delta
             .pivot_table(index=["case","problem_type","scenario"], columns="strategy", values="delta", aggfunc="mean")
             .fillna(0)
             .apply(lambda row: l2(row.values), axis=1)
             .reset_index()
             .groupby(["case","problem_type"])[0]
             .mean()
             .rename("l2_shock")
             .reset_index())

    metrics = sv.merge(signc, on=["strategy","problem_type"], how="left")
    return metrics, anch, shock

def plot_delta_sensitivity(metrics_df):
    order = ["Diversification","Fast Follower","Maintain","Niche Focus","Open Innovation","Technology Leadership"]
    fig, ax = plt.subplots(figsize=(10,5))
    subg = metrics_df[metrics_df["problem_type"]=="generic"].set_index("strategy").reindex(order)
    subs = metrics_df[metrics_df["problem_type"]=="specific"].set_index("strategy").reindex(order)
    x = np.arange(len(order)); w=0.35
    ax.bar(x-w/2, subg["mean_abs"].values, w, label="generic")
    ax.bar(x+w/2, subs["mean_abs"].values, w, label="specific")
    ax.set_xticks(x); ax.set_xticklabels(order, rotation=20, ha="right")
    ax.set_ylabel("Mean |Δ|")
    title = "Sensitivity by Strategy"
    ax.set_title(title)
    ax.legend(); plt.tight_layout()
    save_fig(fig, title)
    plt.show()

def plot_delta_sign_consistency(metrics_df):
    order = ["Diversification","Fast Follower","Maintain","Niche Focus","Open Innovation","Technology Leadership"]
    fig, ax = plt.subplots(figsize=(10,5))
    subg = metrics_df[metrics_df["problem_type"]=="generic"].set_index("strategy").reindex(order)
    subs = metrics_df[metrics_df["problem_type"]=="specific"].set_index("strategy").reindex(order)
    x = np.arange(len(order)); w=0.35
    ax.bar(x-w/2, subg["prop_pos"].values, w, label="generic")
    ax.bar(x+w/2, subs["prop_pos"].values, w, label="specific")
    ax.set_xticks(x); ax.set_xticklabels(order, rotation=20, ha="right")
    ax.set_ylim(0,1); ax.axhline(0.5, color="k", lw=1)
    ax.set_ylabel("Proportion of Δ > 0")
    title = "Sign Consistency"
    ax.set_title(title)
    ax.legend(); plt.tight_layout()
    save_fig(fig, title)
    plt.show()

def plot_anchoring_index(anch_df):
    order = ["Diversification","Fast Follower","Maintain","Niche Focus","Open Innovation","Technology Leadership"]
    anch_df = anch_df.set_index("strategy").reindex(order)
    fig, ax = plt.subplots(figsize=(8,4))
    ax.bar(range(len(order)), anch_df["anchoring_index"].values)
    ax.axhline(0, color="k", lw=1)
    ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=20, ha="right")
    title = "Anchoring Index generic_vs_specific"
    ax.set_title(title)
    plt.tight_layout()
    save_fig(fig, title)
    plt.show()

def plot_shock_size(shock_df):
    cases = shock_df["case"].unique().tolist()
    fig, ax = plt.subplots(figsize=(10,5))
    x = np.arange(len(cases)); w=0.35
    g = shock_df[shock_df["problem_type"]=="generic"].set_index("case").reindex(cases)["l2_shock"].values
    s = shock_df[shock_df["problem_type"]=="specific"].set_index("case").reindex(cases)["l2_shock"].values
    ax.bar(x-w/2, g, w, label="generic")
    ax.bar(x+w/2, s, w, label="specific")
    ax.set_xticks(x); ax.set_xticklabels(cases, rotation=10)
    title = "Overall Shock Size L2"
    ax.set_title(title); ax.legend()
    plt.tight_layout()
    save_fig(fig, title)
    plt.show()

# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    # A) Overall ratios
    ratio_df = load_overall_ratio("./combined_analysis/analysis_overall_ratio.csv")
    plot_ratio_heatmap(ratio_df)
    plot_ratio_delta_from_base(ratio_df)
    metrics_ratio = compute_ratio_metrics(ratio_df)
    print("\n[Overall ratio metrics]\n", metrics_ratio.round(4))
    plot_ratio_metrics(metrics_ratio)
    plot_ratio_pca(ratio_df)

    # B) Deltas
    delta_df = load_delta_all("./combined_analysis")
    metrics_delta, anch_df, shock_df = compute_delta_metrics(delta_df)
    print("\n[Delta metrics: first rows]\n", metrics_delta.head().round(4))
    print("\n[Anchoring index]\n", anch_df.round(4))
    print("\n[Shock size]\n", shock_df.round(4))

    plot_delta_sensitivity(metrics_delta)
    plot_delta_sign_consistency(metrics_delta)
    plot_anchoring_index(anch_df)
    plot_shock_size(shock_df)
