"""
result_analysis/bertopic_rationale_analysis.py

Word-level semantic dendrogram for rationale framing analysis.

Takes the top discriminative bigrams/unigrams from the log-odds analysis
(Generic>Specific and Specific>Generic), embeds each token with
SentenceTransformer, then builds a hierarchical clustering dendrogram.

Goal: show that Specific-dominant words cluster together on one side
(vision / leadership language) and Generic-dominant words cluster on
the other (constraint / execution language) — a structural confirmation
of the Table 3 keyword evidence.

Stage 1  Compute log-odds, select top-N discriminative words, save CSV.
         Output: final_results/summary/framing_discriminative_words.csv
           word | delta_log_odds | abs_delta | direction

Stage 2  Embed each word with SentenceTransformer, hierarchical-cluster,
         draw dendrogram coloured by framing direction.
         Leaf colour: red = Specific-dominant, blue = Generic-dominant
         Output: final_results/plots/eval_framing_word_dendrogram.png

Usage:
  python -m result_analysis.bertopic_rationale_analysis --stage 1
  python -m result_analysis.bertopic_rationale_analysis --stage 2
"""

from __future__ import annotations

import argparse
import os
import re
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.cluster.hierarchy import linkage, dendrogram as scipy_dendrogram
from scipy.spatial.distance import pdist

try:
    from result_analysis.rationale_analysis import (
        load_rationale_data,
        mask_brand_terms,
        NUMERIC_RELATED_STOPWORDS,
    )
except ImportError:
    from rationale_analysis import (
        load_rationale_data,
        mask_brand_terms,
        NUMERIC_RELATED_STOPWORDS,
    )

# ── constants ──────────────────────────────────────────────────────────────────
PAIR_GROUP_COLS = [
    "scenario", "repeat", "Model", "Temperature", "Num Context", "Chosen Option"
]
BRAND_STOP = {
    "tesla", "model", "elon", "musk", "supercharger",
    "company", "brand", "firm", "organization",
}
TOP_N_WORDS = 15    # top-N words from each direction → 30 leaves total
RANDOM_SEED = 42

SUMMARY_DIR    = "./final_results/summary"
PLOTS_DIR      = "./final_results/plots"
WORDS_CSV_PATH = os.path.join(SUMMARY_DIR, "framing_discriminative_words.csv")


# ── data helpers ───────────────────────────────────────────────────────────────
def _build_matched_pairs(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Return (generic_texts, specific_texts) for exactly-matched pairs."""
    gen, spec = [], []
    for _, g in df.groupby(PAIR_GROUP_COLS, dropna=False):
        gs = g.loc[g["problem_type"] == "generic",  "Rationale"].fillna("").tolist()
        ss = g.loc[g["problem_type"] == "specific", "Rationale"].fillna("").tolist()
        m = min(len(gs), len(ss))
        gen.extend(gs[:m])
        spec.extend(ss[:m])
    return gen, spec


def _mask_and_clean(texts: List[str]) -> List[str]:
    out = []
    for t in texts:
        t = mask_brand_terms(t)
        t = re.sub(r"<BRAND>", " ", t)
        t = re.sub(r"\s+", " ", t).strip()
        out.append(t)
    return out


# ── log-odds ───────────────────────────────────────────────────────────────────
def _compute_logodds(
    gen_texts: List[str],
    spec_texts: List[str],
    top_n_each: int = TOP_N_WORDS,
) -> pd.DataFrame:
    """
    Informative Dirichlet log-odds between Specific and Generic rationales.
    Returns top_n_each rows from each side as a DataFrame.
    """
    from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS

    stop_words = sorted(
        ENGLISH_STOP_WORDS.union(BRAND_STOP).union(NUMERIC_RELATED_STOPWORDS)
    )
    vec = CountVectorizer(
        stop_words=stop_words,
        ngram_range=(1, 2),
        min_df=3,
        token_pattern=r"(?u)\b[a-zA-Z]{2,}\b",
    )
    vec.fit(gen_texts + spec_texts)
    vocab = np.array(vec.get_feature_names_out())

    cg    = np.asarray(vec.transform(gen_texts).sum(axis=0)).flatten()
    cs    = np.asarray(vec.transform(spec_texts).sum(axis=0)).flatten()
    alpha = cg + cs
    alpha0 = alpha.sum()
    eps   = 1e-12

    def _log_odds(c: np.ndarray) -> np.ndarray:
        c0    = c.sum()
        denom = (c0 + alpha0) - (c + alpha)
        return np.log((c + alpha + eps) / (denom + eps))

    delta = _log_odds(cs) - _log_odds(cg)   # positive → Specific > Generic

    pos_idx = np.argsort(delta)[-top_n_each:][::-1]   # Specific > Generic
    neg_idx = np.argsort(delta)[:top_n_each]           # Generic  > Specific

    rows = []
    for i in pos_idx:
        rows.append({"word": vocab[i], "delta_log_odds": float(delta[i]),
                     "abs_delta": float(abs(delta[i])), "direction": "Specific>Generic"})
    for i in neg_idx:
        rows.append({"word": vocab[i], "delta_log_odds": float(delta[i]),
                     "abs_delta": float(abs(delta[i])), "direction": "Generic>Specific"})

    return pd.DataFrame(rows)


# ── Stage 1 ────────────────────────────────────────────────────────────────────
def run_stage1(input_dir: str = "infer_results") -> None:
    os.makedirs(SUMMARY_DIR, exist_ok=True)

    print("Loading rationale data …")
    df = load_rationale_data(input_dir)
    print(f"  Total rows: {len(df):,}")

    print("Building matched pairs …")
    gen_raw, spec_raw = _build_matched_pairs(df)
    print(f"  Matched pairs: {len(gen_raw):,}")

    gen_clean  = _mask_and_clean(gen_raw)
    spec_clean = _mask_and_clean(spec_raw)

    print(f"\nComputing log-odds (top {TOP_N_WORDS} per side) …")
    words_df = _compute_logodds(gen_clean, spec_clean, top_n_each=TOP_N_WORDS)

    words_df.to_csv(WORDS_CSV_PATH, index=False)
    print(f"Saved → {WORDS_CSV_PATH}")

    print("\n── Top Specific>Generic ──────────────────────────────────────────")
    print(words_df[words_df.direction == "Specific>Generic"]
          .head(15)[["word", "delta_log_odds"]].to_string(index=False))
    print("\n── Top Generic>Specific ──────────────────────────────────────────")
    print(words_df[words_df.direction == "Generic>Specific"]
          .head(15)[["word", "delta_log_odds"]].to_string(index=False))
    print(f"\nTotal discriminative words: {len(words_df)}")
    print("Run stage 2 to embed and cluster these words into a dendrogram.")


# ── Stage 2 ────────────────────────────────────────────────────────────────────
def run_stage2() -> None:
    from sentence_transformers import SentenceTransformer

    if not os.path.exists(WORDS_CSV_PATH):
        raise FileNotFoundError(f"Run stage 1 first: {WORDS_CSV_PATH} not found.")

    os.makedirs(PLOTS_DIR, exist_ok=True)

    words_df = pd.read_csv(WORDS_CSV_PATH)
    words    = words_df["word"].tolist()
    dirs     = words_df["direction"].tolist()
    n_words  = len(words)

    print(f"Embedding {n_words} discriminative words …")
    model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
    embs  = model.encode(words, show_progress_bar=True, batch_size=64)
    embs  = embs / np.linalg.norm(embs, axis=1, keepdims=True)   # L2-normalise

    # ── hierarchical clustering (average-linkage, cosine distance) ─────────────
    dist_vec = pdist(embs, metric="cosine")
    Z        = linkage(dist_vec, method="average")

    # ── colors ────────────────────────────────────────────────────────────────
    RED  = "#c0392b"   # Specific > Generic
    BLUE = "#2980b9"   # Generic  > Specific
    color_map   = {"Specific>Generic": RED, "Generic>Specific": BLUE}
    leaf_colors = [color_map[d] for d in dirs]

    # ── plot ──────────────────────────────────────────────────────────────────
    fig_h = max(12, n_words * 0.26)
    fig, ax = plt.subplots(figsize=(12, fig_h))

    scipy_dendrogram(
        Z,
        labels=words,
        orientation="left",
        ax=ax,
        color_threshold=0,
        above_threshold_color="#aaaaaa",
        leaf_font_size=7.5,
    )

    # recolour leaf tick-labels
    label_to_color = dict(zip(words, leaf_colors))
    for tick in ax.get_ymajorticklabels():
        col = label_to_color.get(tick.get_text(), "#555555")
        tick.set_color(col)
        tick.set_fontweight("bold")

    x_max = ax.get_xlim()[1]
    ax.set_xlim(right=x_max * 1.02)

    ax.set_xlabel("Cosine distance between word embeddings", fontsize=9)
    ax.set_title(
        "Semantic clustering of discriminative rationale words by framing\n"
        "Matched pairs (same strategy chosen) — only brand framing varies",
        fontsize=10, pad=10,
    )

    legend_handles = [
        mpatches.Patch(color=RED,  label=f"Specific-dominant (top {TOP_N_WORDS} words, Tesla framing)"),
        mpatches.Patch(color=BLUE, label=f"Generic-dominant  (top {TOP_N_WORDS} words, anonymous framing)"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=8.5, framealpha=0.9)
    ax.grid(True, axis="x", linestyle="--", alpha=0.2)
    plt.tight_layout()

    out_path = os.path.join(PLOTS_DIR, "eval_framing_word_dendrogram.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved dendrogram → {out_path}")


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Word-level framing dendrogram (Generic vs Specific rationales)"
    )
    parser.add_argument(
        "--stage", type=int, choices=[1, 2], default=1,
        help="1 = compute log-odds & save word CSV  |  2 = embed & draw dendrogram"
    )
    parser.add_argument("--input_dir", default="infer_results")
    args = parser.parse_args()

    if args.stage == 1:
        run_stage1(args.input_dir)
    else:
        run_stage2()
