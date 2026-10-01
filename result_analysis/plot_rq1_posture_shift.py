"""
Figures for RQ1 (JIIS revision).

  eval_rq1_context_distribution.png : stacked strategy shares per variant
                                      (base, opp_focus, count_fact, randomized_numbers)
  eval_rq1_posture_shift.png        : Δp = p(variant) - p(base) with bootstrap 95% CI
                                      for Prospector and Defender options only

Reads final_results/summary/rq1_strategy_shift.csv (rq1_posture_shift.py).

Usage
-----
  python -m result_analysis.plot_rq1_posture_shift
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SUMMARY_DIR = "./final_results/summary"
PLOTS_DIR = "./final_results/plots"

ORDER = [
    ("Technology Leadership", "Prospector"),
    ("Diversification", "Prospector"),
    ("Fast Follower", "Analyzer"),
    ("Open Innovation", "Unassigned"),
    ("Niche Focus", "Defender"),
    ("Maintain", "Defender"),
    ("Retrenchment", "Defender"),
]
COLORS = {
    "Technology Leadership": "#08519c",
    "Diversification": "#6baed6",
    "Fast Follower": "#41ab5d",
    "Open Innovation": "#bdbdbd",
    "Niche Focus": "#d94801",
    "Maintain": "#fd8d3c",
    "Retrenchment": "#fdd0a2",
}
DARK_LABEL = {"Technology Leadership", "Niche Focus", "Fast Follower"}
VARIANTS = ["opp_focus", "count_fact", "randomized_numbers"]
VARIANT_COLORS = {"opp_focus": "#08519c", "count_fact": "#d94801", "randomized_numbers": "#737373"}


def _clean_axes(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def plot_distribution(d: pd.DataFrame) -> str:
    strategies = [s for s, _ in ORDER]
    shares = {"base": d[d["variant"] == VARIANTS[0]].set_index("strategy")["p_base"]}
    for v in VARIANTS:
        shares[v] = d[d["variant"] == v].set_index("strategy")["p_variant"]

    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    rows = ["base", *VARIANTS]
    y = np.arange(len(rows))[::-1]
    for yi, r in zip(y, rows):
        left = 0.0
        for s in strategies:
            w = float(shares[r][s])
            ax.barh(yi, w, left=left, color=COLORS[s], edgecolor="white", linewidth=0.6)
            if w >= 0.06:
                ax.text(left + w / 2, yi, f"{w:.2f}", ha="center", va="center", fontsize=7.5,
                        color="white" if s in DARK_LABEL else "#222")
            left += w
    ax.set_yticks(y)
    ax.set_yticklabels(rows, fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Share of choices")
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[s]) for s in strategies]
    ax.legend(handles, [f"{s} ({p})" for s, p in ORDER], fontsize=7.5, ncol=2,
              loc="upper center", bbox_to_anchor=(0.45, -0.2), frameon=False)
    _clean_axes(ax)
    return _save(fig, "eval_rq1_context_distribution.png")


def plot_posture_shift(d: pd.DataFrame) -> str:
    groups = [
        ("Prospector", ["Technology Leadership", "Diversification"]),
        (None, ["Fast Follower", "Open Innovation"]),
        ("Defender", ["Niche Focus", "Maintain", "Retrenchment"]),
    ]
    strategies = [s for _, ss in groups for s in ss]
    ys = np.arange(len(strategies))[::-1].astype(float)

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    height = 0.26
    for j, v in enumerate(VARIANTS):
        dv = d[d["variant"] == v].set_index("strategy").loc[strategies]
        vals = dv["delta"].to_numpy()
        err = np.vstack([vals - dv["ci_lower"].to_numpy(), dv["ci_upper"].to_numpy() - vals])
        ax.barh(ys + (1 - j) * height, vals, height=height, color=VARIANT_COLORS[v], xerr=err,
                ecolor="#333", capsize=1.5, error_kw={"linewidth": 0.7}, label=v)
    ax.axvline(0, color="#444", linewidth=0.8)

    xmax = 0.30
    start = 0
    for name, ss in groups:
        top = ys[start] + 0.5
        bottom = ys[start + len(ss) - 1] - 0.5
        start += len(ss)
        if name is None:
            continue
        ax.axhspan(bottom, top, color="#e8eef6" if name == "Prospector" else "#fbeee6", zorder=0)
        if name == "Prospector":
            ax.text(xmax - 0.005, bottom + 0.1, name, ha="right", va="bottom", fontsize=9,
                    style="italic", color="#444")
        else:
            ax.text(xmax - 0.005, top - 0.1, name, ha="right", va="top", fontsize=9,
                    style="italic", color="#444")
    ax.set_yticks(ys)
    ax.set_yticklabels(strategies, fontsize=9)
    ax.set_ylim(ys[-1] - 0.6, ys[0] + 0.6)
    ax.set_xlim(-0.16, xmax)
    ax.set_xlabel(r"Change in share from base, $\Delta p$ (95% CI)")
    ax.legend(fontsize=8, loc="lower right", frameon=False)
    _clean_axes(ax)
    return _save(fig, "eval_rq1_posture_shift.png")


def _save(fig, name: str) -> str:
    fig.tight_layout()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    out = os.path.join(PLOTS_DIR, name)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    d = pd.read_csv(os.path.join(SUMMARY_DIR, "rq1_strategy_shift.csv"))
    for out in (plot_distribution(d), plot_posture_shift(d)):
        print(f"Saved → {out}")


if __name__ == "__main__":
    main()
