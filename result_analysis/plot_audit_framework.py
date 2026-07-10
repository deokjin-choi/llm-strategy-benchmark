"""
Paper figure: scenario-based decision sensitivity audit framework (§6.4).

Output
------
  final_results/plots/eval_decision_sensitivity_audit_framework.png

Usage
-----
  python -m result_analysis.plot_audit_framework
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

PLOTS_DIR = "./final_results/plots"


def _box(ax, x, y, w, h, text, *, fc="#f4f7fb", ec="#2c3e50", lw=1.15, fontsize=8.0, weight="normal"):
    patch = FancyBboxPatch(
        (x - w / 2, y - h / 2),
        w,
        h,
        boxstyle="round,pad=0.015,rounding_size=0.025",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
        zorder=2,
    )
    ax.add_patch(patch)
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        color="#1a1a1a",
        linespacing=1.28,
        zorder=3,
    )
    return patch


def _arrow(ax, x1, y1, x2, y2, color="#4a5568"):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=1.1,
            color=color,
            zorder=1,
            shrinkA=0,
            shrinkB=0,
        )
    )


def plot_audit_framework(
    save_dir: str = PLOTS_DIR,
    filename: str = "eval_decision_sensitivity_audit_framework.png",
) -> str:
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11.5, 8.0))
    ax.set_xlim(0, 11.5)
    ax.set_ylim(0, 8.0)
    ax.axis("off")

    # Background bands (subtle)
    for y0, h, c in [
        (6.35, 1.05, "#eef2f7"),
        (4.55, 1.55, "#f3f6fa"),
        (2.05, 2.15, "#eef5fb"),
        (0.25, 1.45, "#fff8f1"),
    ]:
        ax.add_patch(Rectangle((0.35, y0), 10.8, h, facecolor=c, edgecolor="none", zorder=0, alpha=0.9))

    # Title
    ax.text(
        5.75,
        7.78,
        "Scenario-Based Decision Sensitivity Audit Framework",
        ha="center",
        va="top",
        fontsize=13,
        fontweight="bold",
        color="#1a202c",
    )
    ax.text(
        5.75,
        7.45,
        "Pre-deployment protocol for LLM-based R&D strategic decision support",
        ha="center",
        va="top",
        fontsize=9,
        color="#4a5568",
    )

    # Step 1
    ax.text(0.55, 7.15, "Step 1", fontsize=7.5, color="#2c5282", fontweight="bold", va="center")
    _box(
        ax,
        5.75,
        6.85,
        9.6,
        0.72,
        "Freeze the audit design\n"
        "Fixed R&D dilemma   ·   Closed strategy menu (7 archetypes)   ·   Documented decoding settings",
        fc="#ffffff",
        ec="#2c5282",
        fontsize=8.2,
        weight="bold",
    )
    _arrow(ax, 5.75, 6.45, 5.75, 6.05)

    # Steps 2–3
    ax.text(0.55, 5.85, "Steps 2–3", fontsize=7.5, color="#2c5282", fontweight="bold", va="center")
    ax.text(
        5.75,
        5.90,
        "Controlled perturbations (dilemma & menu held fixed)",
        ha="center",
        va="center",
        fontsize=8.3,
        fontweight="bold",
        color="#2c5282",
    )
    _box(
        ax,
        3.15,
        5.15,
        4.55,
        1.05,
        "Firm-identity contrast\n"
        "Generic (anonymous)  vs  Specific (named firm)\n"
        "Artifact:  Δp = p(Specific) − p(Generic)",
        fc="#ffffff",
        ec="#2b6cb0",
        fontsize=7.7,
    )
    _box(
        ax,
        8.35,
        5.15,
        4.55,
        1.05,
        "Semantic context stress grid\n"
        "base · opportunity · competition · constraint\n"
        "(+ numerical control)   Artifact: JSD / rank vs base",
        fc="#ffffff",
        ec="#2b6cb0",
        fontsize=7.7,
    )
    _arrow(ax, 5.75, 4.58, 5.75, 4.18)

    # Steps 4–5: four axes
    ax.text(0.55, 3.95, "Steps 4–5", fontsize=7.5, color="#2c5282", fontweight="bold", va="center")
    ax.text(
        5.75,
        4.00,
        "Measure four sensitivity axes",
        ha="center",
        va="center",
        fontsize=8.3,
        fontweight="bold",
        color="#2c5282",
    )

    axis_specs = [
        (2.0, "A. Context\nsensitivity\n\nJSD · entropy\nSpearman vs base"),
        (4.5, "B. Firm-identity\nsensitivity\n\nStrategy-level Δp\nCI · FDR"),
        (7.0, "C. Context×identity\nmoderation\n\nΔp by context\nInteraction vs base"),
        (9.5, "D. Rationale\nsensitivity\n\nLog-odds · RDS\nvs noise / ceiling"),
    ]
    for x, txt in axis_specs:
        _box(ax, x, 2.95, 2.2, 1.7, txt, fc="#ffffff", ec="#3182ce", fontsize=7.3)

    _arrow(ax, 5.75, 2.05, 5.75, 1.65)

    # Step 6
    ax.text(0.55, 1.40, "Step 6", fontsize=7.5, color="#c05621", fontweight="bold", va="center")
    _box(
        ax,
        5.75,
        0.95,
        9.6,
        1.15,
        "Human-in-the-loop gate  →  Pass / remediate / withhold deployment\n"
        "Required reports:  Generic–Specific Δp summary  ·  multi-context stress summary  ·  matched-choice rationale note\n"
        "Reviewer actions: flag discounted downside risks · verify decoding settings · document audit trail",
        fc="#ffffff",
        ec="#c05621",
        fontsize=7.6,
        weight="bold",
    )

    out = os.path.join(save_dir, filename)
    plt.tight_layout(pad=0.3)
    plt.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


if __name__ == "__main__":
    path = plot_audit_framework()
    print(f"Saved → {path}")
