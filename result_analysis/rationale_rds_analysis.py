"""
result_analysis/rationale_rds_analysis.py

Rationale Divergence Score (RDS) Analysis
==========================================
For each matched pair (same model / temperature / scenario / num_context /
context_variant / Standard Mapping archetype), compute the cosine distance
between the Generic and Specific rationale embeddings. Repeats are pooled
within each cell (not aligned by repeat index). Only the seven valid
archetypes in ``valid_strategies`` are included (same filter as §5.3–5.4).

Calibration histogram (three distributions, identical preprocessing):
  - RDS (framing):     Generic vs Specific, same Standard Mapping archetype
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
  final_results/plots/eval_rationale_rds_strategy_boxplot.png
  final_results/plots/eval_rationale_rds_calibration_histogram.png

Usage
-----
  python -m result_analysis.rationale_rds_analysis
"""

from __future__ import annotations

import os
import re
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

try:
    from result_analysis.rationale_analysis import load_rationale_data, mask_brand_terms, valid_strategies
except ImportError:
    from rationale_analysis import load_rationale_data, mask_brand_terms, valid_strategies

# ── constants ──────────────────────────────────────────────────────────────────
PAIR_GROUP_COLS = [
    "scenario", "Model", "Temperature",
    "Num Context", "context_variant", "Standard Mapping",
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
    "base",
    "competitive_dynamics",
    "count_fact",
    "opp_focus",
    "randomized_numbers",
]
CONTEXT_LABELS = {
    "base":                  "Base",
    "competitive_dynamics":  "Comp. Dynamics",
    "count_fact":            "Count. Fact",
    "opp_focus":             "Opp. Focus",
    "randomized_numbers":    "Rand. Numbers",
}
CONTEXT_COLORS = {
    "base":                  "#377eb8",
    "competitive_dynamics":  "#ff7f00",
    "count_fact":            "#4daf4a",
    "opp_focus":             "#e41a1c",
    "randomized_numbers":    "#984ea3",
}
STRATEGY_ORDER = list(valid_strategies)

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
    """Keep generic/specific rows with valid archetypes; ensure cleaned text."""
    d = df[df["problem_type"].isin(["generic", "specific"])].copy()
    d["Standard Mapping"] = d["Standard Mapping"].fillna("N/A")
    d = d[d["Standard Mapping"].isin(valid_strategies)].copy()
    d["Rationale"] = d["Rationale"].fillna("").astype(str)
    d["clean_text"] = d["Rationale"].map(_clean)
    return d


def _build_rds_pair_df(df: pd.DataFrame) -> pd.DataFrame:
    """Matched Generic vs Specific pairs within condition×strategy cells."""
    rows = []
    for key, g in df.groupby(PAIR_GROUP_COLS, dropna=False):
        gen = g.loc[g["problem_type"] == "generic", "clean_text"].tolist()
        spec = g.loc[g["problem_type"] == "specific", "clean_text"].tolist()
        m = min(len(gen), len(spec))
        if m == 0:
            continue
        key_dict = dict(zip(PAIR_GROUP_COLS, key))
        strategy = key_dict["Standard Mapping"]
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


def _build_rds_heatmap_df(rds_df: pd.DataFrame) -> pd.DataFrame:
    """Rows = context_variant, columns = strategy (same layout as other paper heatmaps)."""
    mat = (
        rds_df.groupby(["context_variant", "strategy"])["rds"]
        .mean()
        .reset_index()
        .pivot(index="context_variant", columns="strategy", values="rds")
    )
    row_order = [v for v in CONTEXT_ORDER if v in mat.index]
    col_order = [s for s in STRATEGY_ORDER if s in mat.columns]
    return mat.reindex(index=row_order, columns=col_order)


def _plot_rds_heatmap(heatmap_df: pd.DataFrame, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    vals = heatmap_df.to_numpy(dtype=float)
    vmin = float(np.nanmin(vals)) if np.isfinite(vals).any() else 0.0
    vmax = float(np.nanmax(vals)) if np.isfinite(vals).any() else 1.0
    # YlGnBu: same family as Strategy Ratio heatmaps; easier on the eyes than YlOrRd.
    im = ax.imshow(vals, aspect="auto", cmap="YlGnBu", vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(heatmap_df.shape[1]))
    ax.set_xticklabels(heatmap_df.columns.tolist(), rotation=25, ha="right", fontsize=9)
    ax.set_yticks(np.arange(heatmap_df.shape[0]))
    ax.set_yticklabels(heatmap_df.index.tolist(), fontsize=9)
    ax.set_xlabel("Strategy (Chosen Option)", fontsize=9)
    ax.set_ylabel("Context Variant", fontsize=9)
    ax.set_title(
        "Rationale Divergence Score (RDS) by Context Variant × Strategy\n"
        "Matched pairs: same strategy chosen, only brand framing varies",
        fontsize=10, pad=10,
    )
    span = max(vmax - vmin, 1e-9)
    for i in range(heatmap_df.shape[0]):
        for j in range(heatmap_df.shape[1]):
            v = vals[i, j]
            if np.isfinite(v):
                norm = (v - vmin) / span
                txt_color = "white" if norm > 0.62 else "#222222"
                ax.text(j, i, f"{v:.3f}", va="center", ha="center", fontsize=8, color=txt_color)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean RDS (cosine distance)", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _heatmap_df_to_long(heatmap_df: pd.DataFrame) -> pd.DataFrame:
    """Melt context×strategy matrix to long format (one row per cell mean)."""
    long = heatmap_df.stack(future_stack=True).reset_index()
    long.columns = ["context_variant", "strategy", "rds"]
    return long


def _load_rds_heatmap_df(csv_path: str | None = None) -> pd.DataFrame:
    """Load context×strategy RDS matrix from rationale_rds_heatmap.csv."""
    path = csv_path or os.path.join(SUMMARY_DIR, "rationale_rds_heatmap.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"RDS heatmap CSV not found: {path}")
    raw = pd.read_csv(path)
    heatmap_df = _legacy_heatmap_csv_to_matrix(raw)
    if heatmap_df.empty:
        raise ValueError(f"Could not load RDS heatmap data from {path}")
    return heatmap_df


def _plot_rds_strategy_boxplot(heatmap_df: pd.DataFrame, out_path: str) -> None:
    """
    Box plot of RDS by strategy using heatmap cell means only.

    Each strategy has five observations (one mean RDS per context variant),
    taken directly from rationale_rds_heatmap.csv — no re-embedding required.
    """
    cell_long = _heatmap_df_to_long(heatmap_df)
    order = [s for s in STRATEGY_ORDER if s in cell_long["strategy"].unique()]
    ctx_order = [v for v in CONTEXT_ORDER if v in cell_long["context_variant"].unique()]
    palette = {v: CONTEXT_COLORS[v] for v in ctx_order}

    fig, ax = plt.subplots(figsize=(11, 5.5))
    sns.boxplot(
        data=cell_long,
        x="strategy",
        y="rds",
        order=order,
        ax=ax,
        color="#dde7f0",
        linewidth=1.2,
        width=0.55,
        showfliers=False,
        boxprops={"alpha": 0.85},
        zorder=1,
    )
    sns.stripplot(
        data=cell_long,
        x="strategy",
        y="rds",
        hue="context_variant",
        hue_order=ctx_order,
        order=order,
        palette=palette,
        ax=ax,
        size=7,
        jitter=0.08,
        alpha=0.95,
        dodge=False,
        linewidth=0.6,
        edgecolor="white",
        zorder=3,
    )
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        labels,
        title="Context Variant",
        fontsize=8,
        title_fontsize=8,
        loc="upper right",
        framealpha=0.95,
    )
    ax.set_xlabel("Strategy (Chosen Option)", fontsize=9)
    ax.set_ylabel("Mean RDS (cosine distance)", fontsize=9)
    ax.set_title(
        "Rationale Divergence Score (RDS) by Strategy\n"
        "Five context-variant cell means per strategy; matched pairs, brand framing only",
        fontsize=10,
        pad=10,
    )
    ax.tick_params(axis="x", labelrotation=25)
    plt.setp(ax.get_xticklabels(), ha="right")
    ax.grid(True, axis="y", linestyle="--", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _legacy_heatmap_csv_to_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Load heatmap matrix from CSV (supports legacy strategy×context or current context×strategy)."""
    label_to_ctx = {v: k for k, v in CONTEXT_LABELS.items()}
    if "context_variant" in df.columns:
        mat = df.set_index("context_variant")
    elif "strategy" in df.columns:
        mat = df.set_index("strategy").rename(columns=label_to_ctx).T
    elif set(df.index).issubset(set(CONTEXT_ORDER)):
        mat = df.copy()
    else:
        mat = df.rename(columns=label_to_ctx).T
    row_order = [v for v in CONTEXT_ORDER if v in mat.index]
    col_order = [s for s in STRATEGY_ORDER if s in mat.columns]
    return mat.reindex(index=row_order, columns=col_order)


def replot_rds_boxplot() -> None:
    """Regenerate strategy box plot from rationale_rds_heatmap.csv only (no re-embedding)."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    heatmap_df = _load_rds_heatmap_df()
    box_path = os.path.join(PLOTS_DIR, "eval_rationale_rds_strategy_boxplot.png")
    _plot_rds_strategy_boxplot(heatmap_df, box_path)
    print(f"Saved box plot → {box_path}")


def replot_rds_figures() -> None:
    """Regenerate RDS heatmap and strategy box plot from saved CSV (no re-embedding)."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    heatmap_df = _load_rds_heatmap_df()

    long_path = os.path.join(SUMMARY_DIR, "rationale_rds_strategy_context_long.csv")
    _heatmap_df_to_long(heatmap_df).to_csv(long_path, index=False)

    heatmap_path = os.path.join(PLOTS_DIR, "eval_rationale_rds_heatmap.png")
    _plot_rds_heatmap(heatmap_df, heatmap_path)
    print(f"Saved heatmap → {heatmap_path}")

    box_path = os.path.join(PLOTS_DIR, "eval_rationale_rds_strategy_boxplot.png")
    _plot_rds_strategy_boxplot(heatmap_df, box_path)
    print(f"Saved box plot → {box_path}")


def replot_rds_heatmap() -> None:
    """Backward-compatible alias."""
    replot_rds_figures()


def _load_noise_median(default: float = 0.101) -> float:
    cal_path = os.path.join(SUMMARY_DIR, "rationale_rds_calibration_summary.csv")
    if not os.path.exists(cal_path):
        return default
    summary = pd.read_csv(cal_path)
    row = summary.loc[summary.distribution == "noise", "median"]
    return float(row.values[0]) if len(row) else default


def compute_rds_pairs_for_cell(
    input_dir: str,
    *,
    model: str,
    scenario: str,
    strategy: str,
    context_variant: str,
) -> pd.DataFrame:
    """Build and score matched Generic–Specific RDS pairs for one audit case cell."""
    from sentence_transformers import SentenceTransformer

    df = _prepare_df(load_rationale_data(input_dir))
    mask = (
        (df["Model"] == model)
        & (df["scenario"] == scenario)
        & (df["Standard Mapping"] == strategy)
        & (df["context_variant"] == context_variant)
    )
    pairs = _build_rds_pair_df(df.loc[mask])
    if pairs.empty:
        return pairs

    texts = pairs["text_a"].tolist() + pairs["text_b"].tolist()
    emb_model = SentenceTransformer(EMB_MODEL)
    emb_dict = _embed_unique_texts(texts, emb_model)
    return _add_distances(pairs, emb_dict, "rds")


def _summarize_rds_case_cell(rds_cell_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for temp, g in rds_cell_df.groupby("Temperature", dropna=False):
        dist = g["rds"]
        rows.append({
            "Temperature": float(temp),
            "n_pairs": int(len(dist)),
            "mean_rds": round(float(dist.mean()), 4),
            "median_rds": round(float(dist.median()), 4),
            "p75_rds": round(float(dist.quantile(0.75)), 4),
            "max_rds": round(float(dist.max()), 4),
            "share_above_0_12": round(float((dist > 0.12).mean()), 4),
        })
    return pd.DataFrame(rows).sort_values("Temperature")


def plot_rds_matched_choice_case_cell(
    rds_cell_df: pd.DataFrame,
    *,
    model: str,
    scenario: str,
    strategy: str,
    context_variant: str,
    save_path: str,
    temperatures: Sequence[float] = (0.0, 0.7),
    noise_median: float | None = None,
) -> pd.DataFrame:
    """
    Violin + strip plot of pair-level RDS for a localized matched-choice case cell.
    Panels are split by temperature (aligned with Case II JSD deep-dive layout).
    """
    try:
        from result_analysis.model_behavioral_profile import _short_model_name
    except ImportError:
        from model_behavioral_profile import _short_model_name

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    noise_med = float(noise_median if noise_median is not None else _load_noise_median())

    sub = rds_cell_df.copy()
    sub["Temperature"] = sub["Temperature"].astype(float)
    temps = [float(t) for t in temperatures]
    sub = sub[sub["Temperature"].apply(lambda x: any(abs(x - t) < 1e-6 for t in temps))]
    if sub.empty:
        raise ValueError("No RDS pairs in case cell for requested temperatures.")

    sub["temp_label"] = sub["Temperature"].map(lambda x: f"T={float(x):g}")
    temp_labels = [f"T={t:g}" for t in temps]
    palette = {"T=0": "#e74c3c", "T=0.7": "#c0392b"}

    fig, axes = plt.subplots(1, len(temps), figsize=(4.6 * len(temps), 4.2), sharey=True)
    if len(temps) == 1:
        axes = np.array([axes])

    y_max = max(float(sub["rds"].max()) * 1.08, noise_med * 2.2, 0.35)

    for ax, temp in zip(axes, temps):
        g = sub[sub["Temperature"].apply(lambda x: abs(x - temp) < 1e-6)]
        label = f"T={temp:g}"
        if len(g) == 0:
            ax.set_title(f"{label} (no pairs)")
            continue

        sns.violinplot(
            data=g,
            y="rds",
            color=palette.get(label, "#e74c3c"),
            ax=ax,
            inner="box",
            linewidth=1.0,
            width=0.72,
            cut=0,
        )
        sns.stripplot(
            data=g,
            y="rds",
            color="#4a1f1f",
            ax=ax,
            size=2.4,
            alpha=0.28,
            jitter=0.18,
            zorder=3,
        )
        ax.axhline(
            noise_med,
            color="#7f8c8d",
            linestyle="--",
            linewidth=1.4,
            label=f"Repeat-noise median ({noise_med:.3f})",
        )
        med = float(g["rds"].median())
        ax.axhline(med, color="#2471a3", linestyle=":", linewidth=1.2, alpha=0.9)
        ax.scatter([0], [float(g["rds"].max())], color="#111111", s=28, zorder=5)
        ax.text(
            0.12,
            float(g["rds"].max()),
            f"max={float(g['rds'].max()):.3f}",
            fontsize=7.5,
            va="center",
        )
        ax.set_xticks([0])
        ax.set_xticklabels([label], fontsize=9)
        ax.set_ylim(0, y_max)
        ax.set_ylabel("RDS (cosine distance)" if ax is axes[0] else "")
        ax.set_title(f"n={len(g):,}; mean={float(g['rds'].mean()):.3f}", fontsize=9)
        ax.grid(True, axis="y", linestyle="--", alpha=0.25)

    handles = [
        mpatches.Patch(color="#e74c3c", alpha=0.75, label="Pair-level RDS"),
        plt.Line2D([0], [0], color="#7f8c8d", linestyle="--", linewidth=1.4, label="Repeat-noise median"),
        plt.Line2D([0], [0], color="#2471a3", linestyle=":", linewidth=1.2, label="Cell median"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=3, fontsize=8)
    fig.suptitle(
        f"Rationale sensitivity: matched-choice RDS distribution\n"
        f"{_short_model_name(model)} — {scenario} — {strategy} × {context_variant}",
        fontsize=10.5,
        y=1.12,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return _summarize_rds_case_cell(sub)


def plot_rds_case_cell_from_infer(
    input_dir: str,
    *,
    model: str,
    scenario: str,
    strategy: str,
    context_variant: str,
    save_path: str,
    summary_path: str | None = None,
    temperatures: Sequence[float] = (0.0, 0.7),
) -> pd.DataFrame:
    """End-to-end: score and plot RDS for one §5.6 rationale case cell."""
    pairs = compute_rds_pairs_for_cell(
        input_dir,
        model=model,
        scenario=scenario,
        strategy=strategy,
        context_variant=context_variant,
    )
    if pairs.empty:
        raise ValueError(f"No matched RDS pairs for {model} / {scenario} / {strategy} / {context_variant}")

    summary = plot_rds_matched_choice_case_cell(
        pairs,
        model=model,
        scenario=scenario,
        strategy=strategy,
        context_variant=context_variant,
        save_path=save_path,
        temperatures=temperatures,
    )
    if summary_path:
        os.makedirs(os.path.dirname(summary_path) or ".", exist_ok=True)
        summary.insert(0, "Model", model)
        summary.insert(1, "scenario", scenario)
        summary.insert(2, "strategy", strategy)
        summary.insert(3, "context_variant", context_variant)
        summary.to_csv(summary_path, index=False)
    return summary


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

    heatmap_df = _build_rds_heatmap_df(rds_df)
    heatmap_df.to_csv(os.path.join(SUMMARY_DIR, "rationale_rds_heatmap.csv"))
    _heatmap_df_to_long(heatmap_df).to_csv(
        os.path.join(SUMMARY_DIR, "rationale_rds_strategy_context_long.csv"), index=False,
    )

    heatmap_path = os.path.join(PLOTS_DIR, "eval_rationale_rds_heatmap.png")
    _plot_rds_heatmap(heatmap_df, heatmap_path)
    print(f"Saved heatmap → {heatmap_path}")

    box_path = os.path.join(PLOTS_DIR, "eval_rationale_rds_strategy_boxplot.png")
    _plot_rds_strategy_boxplot(heatmap_df, box_path)
    print(f"Saved box plot → {box_path}")


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
                        help="Regenerate plots from saved outputs (no re-embedding)")
    parser.add_argument("--heatmap-only", action="store_true",
                        help="Regenerate RDS heatmap and box plot from saved CSV")
    parser.add_argument("--boxplot-only", action="store_true",
                        help="Regenerate RDS strategy box plot from rationale_rds_heatmap.csv only")
    args = parser.parse_args()

    if args.boxplot_only:
        replot_rds_boxplot()
    elif args.heatmap_only:
        replot_rds_figures()
    elif args.plot_only:
        replot_calibration()
        replot_rds_figures()
    else:
        run(args.input_dir)
