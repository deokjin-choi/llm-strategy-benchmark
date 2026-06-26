"""
result_analysis/rationale_rds_analysis.py

Rationale Divergence Score (RDS) Analysis
==========================================
For each matched pair (same scenario / repeat / model / temperature /
num_context / context_variant / chosen_strategy), compute the cosine
distance between the Generic and Specific rationale embeddings.

Calibration histogram (three distributions, identical preprocessing):
  - RDS (framing):     Generic vs Specific, same strategy & repeat
  - Noise (lower):     same cell, same framing & strategy, different repeats
  - Ceiling (upper):   same cell, same framing, different strategies

All texts embedded once (unique strings); cosine distance = 1 - cosine similarity.

Outputs
-------
  final_results/summary/rationale_rds_by_context_variant.csv
  final_results/summary/rationale_rds_by_strategy.csv
  final_results/summary/rationale_rds_heatmap.csv
  final_results/summary/rationale_rds_calibration_summary.csv
  final_results/plots/eval_rationale_rds_heatmap.png
  final_results/plots/eval_rationale_rds_calibration_histogram.png

Usage
-----
  python -m result_analysis.rationale_rds_analysis
"""

from __future__ import annotations

import os
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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

NOISE_CELL_COLS = [
    "scenario", "Model", "Temperature", "Num Context",
    "context_variant", "Standard Mapping", "problem_type",
]

CEILING_CELL_COLS = [
    "scenario", "Model", "Temperature", "Num Context",
    "context_variant", "problem_type",
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

BATCH_SIZE  = 128
EMB_MODEL   = "paraphrase-MiniLM-L6-v2"
RANDOM_SEED = 42


# ── helpers ────────────────────────────────────────────────────────────────────
def _clean(text: str) -> str:
    t = mask_brand_terms(text)
    t = re.sub(r"<BRAND>", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    """Keep generic/specific rows; ensure Standard Mapping and cleaned text."""
    d = df[df["problem_type"].isin(["generic", "specific"])].copy()
    d["Rationale"] = d["Rationale"].fillna("").astype(str)
    d["clean_text"] = d["Rationale"].map(_clean)
    if d["Standard Mapping"].isna().any():
        strat_map = (
            d[["scenario", "Chosen Option", "Standard Mapping"]]
            .dropna(subset=["Standard Mapping"])
            .drop_duplicates(subset=["scenario", "Chosen Option"])
            .set_index(["scenario", "Chosen Option"])["Standard Mapping"]
            .to_dict()
        )
        miss = d["Standard Mapping"].isna()
        d.loc[miss, "Standard Mapping"] = d.loc[miss].apply(
            lambda r: strat_map.get((r["scenario"], r["Chosen Option"]), r["Chosen Option"]),
            axis=1,
        )
    return d


def _build_rds_pair_df(df: pd.DataFrame) -> pd.DataFrame:
    """Matched Generic vs Specific pairs (Fig. 5 pipeline)."""
    rows = []
    for key, g in df.groupby(PAIR_GROUP_COLS, dropna=False):
        gen = g.loc[g["problem_type"] == "generic", "clean_text"].tolist()
        spec = g.loc[g["problem_type"] == "specific", "clean_text"].tolist()
        m = min(len(gen), len(spec))
        if m == 0:
            continue
        key_dict = dict(zip(PAIR_GROUP_COLS, key))
        strategy = g["Standard Mapping"].iloc[0]
        for i in range(m):
            rows.append({
                **key_dict,
                "strategy": strategy,
                "text_a": gen[i],
                "text_b": spec[i],
                "pair_type": "rds",
            })
    return pd.DataFrame(rows)


def _build_noise_pair_df(df: pd.DataFrame, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    One random repeat pair per cell (same framing, same strategy).
    Cell = NOISE_CELL_COLS.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for key, g in df.groupby(NOISE_CELL_COLS, dropna=False):
        by_repeat = (
            g.groupby("repeat", dropna=False)["clean_text"]
            .first()
        )
        if len(by_repeat) < 2:
            continue
        idx = rng.choice(len(by_repeat), size=2, replace=False)
        repeats = by_repeat.index.tolist()
        r1, r2 = repeats[idx[0]], repeats[idx[1]]
        rows.append({
            **dict(zip(NOISE_CELL_COLS, key)),
            "repeat_a": r1,
            "repeat_b": r2,
            "text_a": by_repeat.loc[r1],
            "text_b": by_repeat.loc[r2],
            "pair_type": "noise",
        })
    return pd.DataFrame(rows)


def _build_ceiling_pair_df(df: pd.DataFrame, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    One random cross-strategy pair per cell (same framing).
    Cell = CEILING_CELL_COLS.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for key, g in df.groupby(CEILING_CELL_COLS, dropna=False):
        strategies = g["Standard Mapping"].dropna().unique().tolist()
        if len(strategies) < 2:
            continue
        s_idx = rng.choice(len(strategies), size=2, replace=False)
        s_a, s_b = strategies[s_idx[0]], strategies[s_idx[1]]
        ga = g[g["Standard Mapping"] == s_a]
        gb = g[g["Standard Mapping"] == s_b]
        ra = ga.iloc[int(rng.integers(len(ga)))]
        rb = gb.iloc[int(rng.integers(len(gb)))]
        rows.append({
            **dict(zip(CEILING_CELL_COLS, key)),
            "strategy_a": s_a,
            "strategy_b": s_b,
            "text_a": ra["clean_text"],
            "text_b": rb["clean_text"],
            "pair_type": "ceiling",
        })
    return pd.DataFrame(rows)


def _embed_unique_texts(texts: List[str], model) -> Dict[str, np.ndarray]:
    """Embed each unique string once; return text → L2-normalised vector."""
    unique = list(dict.fromkeys(texts))
    print(f"  Unique texts to embed: {len(unique):,}")
    raw = model.encode(unique, batch_size=BATCH_SIZE, show_progress_bar=True)
    raw = raw / (np.linalg.norm(raw, axis=1, keepdims=True) + 1e-12)
    return dict(zip(unique, raw))


def _cosine_distance_from_dict(text_a: str, text_b: str, emb: Dict[str, np.ndarray]) -> float:
    a = emb[text_a]
    b = emb[text_b]
    return float(1.0 - np.dot(a, b))


def _add_distances(pairs: pd.DataFrame, emb: Dict[str, np.ndarray], col: str = "distance") -> pd.DataFrame:
    out = pairs.copy()
    out[col] = [
        _cosine_distance_from_dict(a, b, emb)
        for a, b in zip(out["text_a"], out["text_b"])
    ]
    return out


def _summarize(dist: pd.Series, name: str) -> dict:
    return {
        "distribution": name,
        "n_pairs": len(dist),
        "mean": round(dist.mean(), 4),
        "median": round(dist.median(), 4),
        "std": round(dist.std(), 4),
        "p25": round(dist.quantile(0.25), 4),
        "p75": round(dist.quantile(0.75), 4),
    }


def _plot_rds_histogram_with_references(
    rds: np.ndarray,
    noise_median: float,
    ceiling_median: float,
    summary: pd.DataFrame,
    out_path: str,
) -> None:
    """
    RDS histogram only; noise/ceiling shown as vertical reference lines (median).
    Median is preferred over mean here: noise is zero-inflated (T=0 repeats) and
    embedding-distance distributions are typically right-skewed.
    """
    rds_med = float(np.median(rds))
    rds_mean = float(np.mean(rds))

    fig, ax = plt.subplots(figsize=(8.5, 5))
    x_max = max(float(np.percentile(rds, 99.5)), ceiling_median * 1.15, 0.45)
    bins = np.linspace(0, x_max, 45)

    ax.hist(
        rds, bins=bins, density=True, alpha=0.72, color="#e74c3c", edgecolor="white",
        linewidth=0.3, label=f"RDS (n={len(rds):,})",
    )

    ax.axvline(noise_median, color="#7f8c8d", linestyle="--", linewidth=1.8,
               label=f"Noise baseline (median={noise_median:.3f})")
    ax.axvline(rds_med, color="#c0392b", linestyle="-", linewidth=1.2, alpha=0.55,
               label=f"RDS median={rds_med:.3f}")
    ax.axvline(ceiling_median, color="#2471a3", linestyle="--", linewidth=1.8,
               label=f"Strategy ceiling (median={ceiling_median:.3f})")

    ax.set_xlim(0, x_max)
    ax.set_xlabel("Cosine distance between rationale embeddings", fontsize=9)
    ax.set_ylabel("Density", fontsize=9)
    ax.set_title(
        "Rationale Divergence Score (RDS) with calibration references\n"
        "Dashed lines: repeat-noise lower bound and cross-strategy upper bound (medians)",
        fontsize=10, pad=10,
    )
    ax.legend(fontsize=8, loc="upper right", framealpha=0.92)
    ax.grid(True, axis="y", linestyle="--", alpha=0.25)

    note = (
        f"RDS mean={rds_mean:.3f}; reference medians from one random pair per condition cell "
        f"(noise n={int(summary.loc[summary.distribution=='noise','n_pairs'].values[0]):,}, "
        f"ceiling n={int(summary.loc[summary.distribution=='ceiling','n_pairs'].values[0]):,})"
    )
    ax.text(0.01, 0.98, note, transform=ax.transAxes, fontsize=7, va="top", color="#555555")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_calibration_histogram(
    rds: np.ndarray,
    noise: np.ndarray,
    ceiling: np.ndarray,
    summary: pd.DataFrame,
    out_path: str,
) -> None:
    """Overlapping percent histograms for noise / RDS / strategy-ceiling calibration."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x_max = max(0.50, float(rds.max()), float(noise.max()), float(ceiling.max()))
    bins = np.linspace(0, x_max, 50)

    def pct_weights(data: np.ndarray) -> np.ndarray:
        return np.full(len(data), 100.0 / len(data))

    series = [
        ("noise", noise, "#95a5a6", "#7f8c8d", "Repeat noise", 0.50),
        ("rds", rds, "#e74c3c", "#c0392b", "Framing (RDS)", 0.55),
        ("ceiling", ceiling, "#2980b9", "#2471a3", "Strategy ceiling", 0.50),
    ]
    for z, (dist_name, data, fill_color, edge_color, label, alpha) in enumerate(series, start=1):
        med = float(summary.loc[summary.distribution == dist_name, "median"].values[0])
        ax.hist(
            data,
            bins=bins,
            weights=pct_weights(data),
            histtype="bar",
            color=fill_color,
            edgecolor=edge_color,
            linewidth=0.45,
            alpha=alpha,
            label=f"{label} (median={med:.3f})",
            zorder=z,
        )
        ax.axvline(med, color=edge_color, linestyle="--", linewidth=1.1, alpha=0.85, zorder=z + 3)

    ax.set_xlim(0, x_max)
    ax.set_xlabel("Cosine distance between rationale embeddings", fontsize=9)
    ax.set_ylabel("Percent of pairs (%)", fontsize=9)
    ax.set_title("Framing vs. repeat noise vs. strategy ceiling", fontsize=11, pad=8)
    ax.legend(fontsize=8.5, loc="upper right", framealpha=0.92)
    ax.grid(True, axis="y", linestyle="--", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ── main ───────────────────────────────────────────────────────────────────────
def run(input_dir: str = "infer_results") -> None:
    from sentence_transformers import SentenceTransformer

    os.makedirs(SUMMARY_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR,   exist_ok=True)

    # ── 1. load ───────────────────────────────────────────────────────────────
    print("Loading rationale data …")
    df = _prepare_df(load_rationale_data(input_dir))
    print(f"  Rows (generic+specific): {len(df):,}")

    # ── 2. build pair tables ───────────────────────────────────────────────────
    print("\nBuilding pair tables …")
    rds_df     = _build_rds_pair_df(df)
    noise_df   = _build_noise_pair_df(df)
    ceiling_df = _build_ceiling_pair_df(df)
    print(f"  RDS pairs:     {len(rds_df):,}")
    print(f"  Noise pairs:   {len(noise_df):,}")
    print(f"  Ceiling pairs: {len(ceiling_df):,}")

    all_texts = (
        rds_df["text_a"].tolist() + rds_df["text_b"].tolist()
        + noise_df["text_a"].tolist() + noise_df["text_b"].tolist()
        + ceiling_df["text_a"].tolist() + ceiling_df["text_b"].tolist()
    )

    # ── 3. embed once ──────────────────────────────────────────────────────────
    print(f"\nEmbedding unique rationales with {EMB_MODEL} (single pass) …")
    model = SentenceTransformer(EMB_MODEL)
    emb_dict = _embed_unique_texts(all_texts, model)

    rds_df     = _add_distances(rds_df,     emb_dict, "rds")
    noise_df   = _add_distances(noise_df,   emb_dict, "distance")
    ceiling_df = _add_distances(ceiling_df, emb_dict, "distance")

    print(f"\n  RDS     mean={rds_df['rds'].mean():.4f}  median={rds_df['rds'].median():.4f}")
    print(f"  Noise   mean={noise_df['distance'].mean():.4f}  median={noise_df['distance'].median():.4f}")
    print(f"  Ceiling mean={ceiling_df['distance'].mean():.4f}  median={ceiling_df['distance'].median():.4f}")

    # ── 4. calibration summary ─────────────────────────────────────────────────
    summary = pd.DataFrame([
        _summarize(rds_df["rds"], "rds"),
        _summarize(noise_df["distance"], "noise"),
        _summarize(ceiling_df["distance"], "ceiling"),
    ])
    cal_path = os.path.join(SUMMARY_DIR, "rationale_rds_calibration_summary.csv")
    summary.to_csv(cal_path, index=False)
    print(f"\n── Calibration summary ───────────────────────────────────────────")
    print(summary.to_string(index=False))
    print(f"Saved → {cal_path}")

    hist_path = os.path.join(PLOTS_DIR, "eval_rationale_rds_calibration_histogram.png")
    _plot_calibration_histogram(
        rds_df["rds"].values,
        noise_df["distance"].values,
        ceiling_df["distance"].values,
        summary,
        hist_path,
    )
    print(f"Saved calibration histogram → {hist_path}")

    ref_hist_path = os.path.join(PLOTS_DIR, "eval_rationale_rds_histogram_with_refs.png")
    _plot_rds_histogram_with_references(
        rds_df["rds"].values,
        float(summary.loc[summary.distribution == "noise", "median"].values[0]),
        float(summary.loc[summary.distribution == "ceiling", "median"].values[0]),
        summary,
        ref_hist_path,
    )
    print(f"Saved RDS-only histogram with reference lines → {ref_hist_path}")

    npz_path = os.path.join(SUMMARY_DIR, "rationale_rds_calibration_distances.npz")
    np.savez(
        npz_path,
        rds=rds_df["rds"].values,
        noise=noise_df["distance"].values,
        ceiling=ceiling_df["distance"].values,
    )

    # ── 5. RDS aggregates (Fig. 5 heatmap) ────────────────────────────────────
    rds_cv = (rds_df.groupby("context_variant")["rds"]
              .agg(mean_rds="mean", median_rds="median", n_pairs="count")
              .reset_index())
    rds_cv.to_csv(os.path.join(SUMMARY_DIR, "rationale_rds_by_context_variant.csv"), index=False)

    rds_strat = (rds_df.groupby("strategy")["rds"]
                 .agg(mean_rds="mean", median_rds="median", n_pairs="count")
                 .sort_values("mean_rds", ascending=False)
                 .reset_index())
    rds_strat.to_csv(os.path.join(SUMMARY_DIR, "rationale_rds_by_strategy.csv"), index=False)

    heatmap_df = (rds_df.groupby(["context_variant", "strategy"])["rds"]
                  .mean()
                  .reset_index()
                  .pivot(index="strategy", columns="context_variant", values="rds"))
    col_order = [c for c in CONTEXT_ORDER if c in heatmap_df.columns]
    heatmap_df = heatmap_df[col_order]
    heatmap_df.columns = [CONTEXT_LABELS.get(c, c) for c in heatmap_df.columns]
    heatmap_df.to_csv(os.path.join(SUMMARY_DIR, "rationale_rds_heatmap.csv"))

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.heatmap(
        heatmap_df, ax=ax, annot=True, fmt=".3f", cmap="YlOrRd",
        linewidths=0.4, linecolor="#dddddd",
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
    heatmap_path = os.path.join(PLOTS_DIR, "eval_rationale_rds_heatmap.png")
    plt.savefig(heatmap_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved heatmap → {heatmap_path}")


def replot_calibration() -> None:
    """Regenerate histograms from saved distances (no re-embedding)."""
    npz_path = os.path.join(SUMMARY_DIR, "rationale_rds_calibration_distances.npz")
    cal_path = os.path.join(SUMMARY_DIR, "rationale_rds_calibration_summary.csv")
    if not os.path.exists(npz_path) or not os.path.exists(cal_path):
        raise FileNotFoundError("Run full analysis first to create .npz and summary CSV.")

    os.makedirs(PLOTS_DIR, exist_ok=True)
    data = np.load(npz_path)
    summary = pd.read_csv(cal_path)

    hist_path = os.path.join(PLOTS_DIR, "eval_rationale_rds_calibration_histogram.png")
    _plot_calibration_histogram(
        data["rds"], data["noise"], data["ceiling"], summary, hist_path,
    )
    ref_hist_path = os.path.join(PLOTS_DIR, "eval_rationale_rds_histogram_with_refs.png")
    _plot_rds_histogram_with_references(
        data["rds"],
        float(summary.loc[summary.distribution == "noise", "median"].values[0]),
        float(summary.loc[summary.distribution == "ceiling", "median"].values[0]),
        summary,
        ref_hist_path,
    )
    print(f"Saved → {hist_path}")
    print(f"Saved → {ref_hist_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Rationale Divergence Score analysis")
    parser.add_argument("--input_dir", default="infer_results")
    parser.add_argument("--plot-only", action="store_true",
                        help="Regenerate histograms from saved .npz (no re-embedding)")
    args = parser.parse_args()

    if args.plot_only:
        replot_calibration()
    else:
        run(args.input_dir)
