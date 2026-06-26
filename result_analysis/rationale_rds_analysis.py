"""
result_analysis/rationale_rds_analysis.py

Rationale Divergence Score (RDS) Analysis
==========================================
For each matched pair (same scenario / repeat / model / temperature /
num_context / context_variant / chosen_strategy), compute the cosine
distance between the Generic and Specific rationale embeddings.

RDS ∈ [0, 1]:  0 = identical narrative,  1 = maximally different narrative.

Key insight: strategy choice may be the same (Chosen Option identical) yet
the justification framing can differ substantially — a form of "stealth bias"
that simple output comparison misses.

Outputs
-------
  final_results/summary/rationale_rds_by_context_variant.csv
  final_results/summary/rationale_rds_by_strategy.csv
  final_results/summary/rationale_rds_heatmap.csv      (context × strategy)
  final_results/plots/eval_rationale_rds_heatmap.png

Usage
-----
  python -m result_analysis.rationale_rds_analysis
"""

from __future__ import annotations

import os
import re
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

try:
    from result_analysis.rationale_analysis import load_rationale_data, mask_brand_terms
except ImportError:
    from rationale_analysis import load_rationale_data, mask_brand_terms

# ── constants ──────────────────────────────────────────────────────────────────
PAIR_GROUP_COLS = [
    "scenario", "repeat", "Model", "Temperature",
    "Num Context", "context_variant", "Chosen Option",
]

CONTEXT_ORDER = [
    "base", "opp_focus", "count_fact",
    "competitive_dynamics", "randomized_numbers",
]
CONTEXT_LABELS = {
    "base":                  "Base",
    "opp_focus":             "Opp. Focus",
    "count_fact":            "Count. Fact",
    "competitive_dynamics":  "Comp. Dynamics",
    "randomized_numbers":    "Rand. Numbers",
}

SUMMARY_DIR = "./final_results/summary"
PLOTS_DIR   = "./final_results/plots"

BATCH_SIZE  = 128   # SentenceTransformer batch size
EMB_MODEL   = "paraphrase-MiniLM-L6-v2"


# ── helpers ────────────────────────────────────────────────────────────────────
def _clean(text: str) -> str:
    t = mask_brand_terms(text)
    t = re.sub(r"<BRAND>", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _build_pair_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    For every group in PAIR_GROUP_COLS that has at least one Generic AND
    one Specific rationale, emit one row per (generic, specific) pair.
    Retains context_variant, Chosen Option, and Standard Mapping for grouping.
    """
    # Build a (scenario, Chosen Option) → Standard Mapping lookup
    strat_map = (
        df[["scenario", "Chosen Option", "Standard Mapping"]]
        .dropna(subset=["Standard Mapping"])
        .drop_duplicates(subset=["scenario", "Chosen Option"])
        .set_index(["scenario", "Chosen Option"])["Standard Mapping"]
        .to_dict()
    )

    rows = []
    for key, g in df.groupby(PAIR_GROUP_COLS, dropna=False):
        gen_texts  = g.loc[g["problem_type"] == "generic",  "Rationale"].fillna("").tolist()
        spec_texts = g.loc[g["problem_type"] == "specific", "Rationale"].fillna("").tolist()
        m = min(len(gen_texts), len(spec_texts))
        if m == 0:
            continue
        key_dict = dict(zip(PAIR_GROUP_COLS, key))
        scenario      = key_dict["scenario"]
        chosen_option = key_dict["Chosen Option"]
        strategy_name = strat_map.get((scenario, chosen_option), chosen_option)
        for i in range(m):
            rows.append({
                **key_dict,
                "strategy":  strategy_name,
                "gen_text":  _clean(gen_texts[i]),
                "spec_text": _clean(spec_texts[i]),
            })
    return pd.DataFrame(rows)


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine distance between two (N, D) arrays."""
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return 1.0 - np.einsum("ij,ij->i", a, b)


# ── main ───────────────────────────────────────────────────────────────────────
def run(input_dir: str = "infer_results") -> None:
    from sentence_transformers import SentenceTransformer

    os.makedirs(SUMMARY_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR,   exist_ok=True)

    # ── 1. load & pair ────────────────────────────────────────────────────────
    print("Loading rationale data …")
    df = load_rationale_data(input_dir)
    print(f"  Total rows: {len(df):,}")

    print("Building matched pairs …")
    pair_df = _build_pair_df(df)
    print(f"  Matched pairs: {len(pair_df):,}")

    # ── 2. embed ──────────────────────────────────────────────────────────────
    print(f"\nEmbedding rationales with {EMB_MODEL} …")
    model = SentenceTransformer(EMB_MODEL)

    gen_embs  = model.encode(pair_df["gen_text"].tolist(),
                             batch_size=BATCH_SIZE, show_progress_bar=True)
    spec_embs = model.encode(pair_df["spec_text"].tolist(),
                             batch_size=BATCH_SIZE, show_progress_bar=True)

    pair_df["rds"] = _cosine_distance(gen_embs, spec_embs)
    print(f"\n  Overall RDS  mean={pair_df['rds'].mean():.4f}  "
          f"median={pair_df['rds'].median():.4f}  "
          f"std={pair_df['rds'].std():.4f}")

    # ── 3. aggregate ──────────────────────────────────────────────────────────
    # By context variant
    rds_cv = (pair_df.groupby("context_variant")["rds"]
              .agg(mean_rds="mean", median_rds="median", n_pairs="count")
              .reset_index())
    rds_cv.to_csv(os.path.join(SUMMARY_DIR, "rationale_rds_by_context_variant.csv"),
                  index=False)
    print("\n── RDS by context variant ────────────────────────────────────────")
    print(rds_cv.to_string(index=False))

    # By strategy (Standard Mapping)
    rds_strat = (pair_df.groupby("strategy")["rds"]
                 .agg(mean_rds="mean", median_rds="median", n_pairs="count")
                 .sort_values("mean_rds", ascending=False)
                 .reset_index())
    rds_strat.to_csv(os.path.join(SUMMARY_DIR, "rationale_rds_by_strategy.csv"),
                     index=False)
    print("\n── RDS by strategy ───────────────────────────────────────────────")
    print(rds_strat.to_string(index=False))

    # Heatmap: context × strategy
    heatmap_df = (pair_df.groupby(["context_variant", "strategy"])["rds"]
                  .mean()
                  .reset_index()
                  .pivot(index="strategy", columns="context_variant", values="rds"))

    # reorder columns by canonical context order
    col_order = [c for c in CONTEXT_ORDER if c in heatmap_df.columns]
    heatmap_df = heatmap_df[col_order]
    heatmap_df.columns = [CONTEXT_LABELS.get(c, c) for c in heatmap_df.columns]

    heatmap_df.to_csv(os.path.join(SUMMARY_DIR, "rationale_rds_heatmap.csv"))
    print("\nHeatmap shape:", heatmap_df.shape)

    # ── 4. plot ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.heatmap(
        heatmap_df,
        ax=ax,
        annot=True,
        fmt=".3f",
        cmap="YlOrRd",
        linewidths=0.4,
        linecolor="#dddddd",
        cbar_kws={"label": "Mean RDS (cosine distance)", "shrink": 0.8},
        annot_kws={"size": 8},
    )
    ax.set_title(
        "Rationale Divergence Score (RDS) by Context Variant × Strategy\n"
        "Matched pairs: same strategy chosen, only brand framing varies",
        fontsize=10, pad=10,
    )
    ax.set_xlabel("Context Variant", fontsize=9)
    ax.set_ylabel("Strategy (Chosen Option)", fontsize=9)
    ax.tick_params(axis="x", labelsize=8, rotation=20)
    ax.tick_params(axis="y", labelsize=8, rotation=0)
    plt.tight_layout()

    out_path = os.path.join(PLOTS_DIR, "eval_rationale_rds_heatmap.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved heatmap → {out_path}")

    # ── 5. high-RDS examples ──────────────────────────────────────────────────
    print("\n── Top-5 highest-RDS pairs (same strategy, divergent rationale) ──")
    top5 = pair_df.nlargest(5, "rds")[
        ["context_variant", "strategy", "Model", "rds", "gen_text", "spec_text"]
    ]
    for _, row in top5.iterrows():
        print(f"\n  RDS={row['rds']:.4f}  strategy={row['strategy']}"
              f"  ctx={row['context_variant']}  model={row['Model']}")
        print(f"  [Generic]  {row['gen_text'][:200]}")
        print(f"  [Specific] {row['spec_text'][:200]}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Rationale Divergence Score analysis")
    parser.add_argument("--input_dir", default="infer_results")
    args = parser.parse_args()
    run(args.input_dir)
