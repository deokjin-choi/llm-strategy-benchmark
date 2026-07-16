"""
Generate swarm (beeswarm) deep-dive plot for the §5.6 rationale case cell.

Keeps the original violin plot in rationale_rds_analysis.plot_rds_matched_choice_case_cell;
this script writes a separate figure with a ``_swarm`` suffix.

Usage
-----
  python -m result_analysis.plot_rds_case_cell_swarm
  python -m result_analysis.plot_rds_case_cell_swarm --input_dir infer_results
"""

from __future__ import annotations

import argparse
import os
import re

from result_analysis.rationale_rds_analysis import (
    PLOTS_DIR,
    plot_rds_matched_choice_case_cell_swarm,
    plot_rds_matched_choice_case_cell_swarm_envelope,
)

try:
    from result_analysis.model_behavioral_profile import _short_model_name
    from result_analysis.rds_by_strategy_ci import load_rds_pairs_with_distances
except ImportError:
    from model_behavioral_profile import _short_model_name
    from rds_by_strategy_ci import load_rds_pairs_with_distances


def _safe_filename(s: str) -> str:
    return re.sub(r"[^\w\-.]+", "_", s)


# §5.6 Case III (rationale sensitivity)
DEFAULT_MODEL = "Qwen/Qwen2.5-14B-Instruct"
DEFAULT_SCENARIO = "2_roadster_launch"
DEFAULT_STRATEGY = "Maintain"
DEFAULT_CONTEXT_VARIANT = "count_fact"
DEFAULT_TEMPERATURES = (0.0, 0.7)


def run(
    input_dir: str = "infer_results",
    *,
    model: str = DEFAULT_MODEL,
    scenario: str = DEFAULT_SCENARIO,
    strategy: str = DEFAULT_STRATEGY,
    context_variant: str = DEFAULT_CONTEXT_VARIANT,
    temperatures: tuple[float, ...] = DEFAULT_TEMPERATURES,
    plots_dir: str = PLOTS_DIR,
    envelope: bool = False,
    envelope_bw_factor: float = 0.38,
) -> str:
    pairs = load_rds_pairs_with_distances(input_dir=input_dir)
    mask = (
        (pairs["Model"] == model)
        & (pairs["scenario"] == scenario)
        & (pairs["strategy"] == strategy)
        & (pairs["context_variant"] == context_variant)
    )
    cell_pairs = pairs.loc[mask]
    if cell_pairs.empty:
        raise ValueError(
            f"No matched RDS pairs for {model} / {scenario} / {strategy} / {context_variant}"
        )

    suffix = "swarm_envelope" if envelope else "swarm"
    save_path = os.path.join(
        plots_dir,
        f"eval_deepdive_rds_matched_choice_{suffix}__"
        f"{_safe_filename(_short_model_name(model))}__"
        f"{_safe_filename(scenario)}__"
        f"{_safe_filename(strategy)}__"
        f"{_safe_filename(context_variant)}.png",
    )
    plot_kwargs = dict(
        model=model,
        scenario=scenario,
        strategy=strategy,
        context_variant=context_variant,
        save_path=save_path,
        temperatures=temperatures,
    )
    if envelope:
        summary = plot_rds_matched_choice_case_cell_swarm_envelope(
            cell_pairs,
            envelope_bw_factor=envelope_bw_factor,
            **plot_kwargs,
        )
    else:
        summary = plot_rds_matched_choice_case_cell_swarm(cell_pairs, **plot_kwargs)
    print(summary.to_string(index=False))
    print(f"Saved → {save_path}")
    return save_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Swarm deep-dive plot for matched-choice RDS case cell (Fig. 12 alt.)",
    )
    parser.add_argument("--input_dir", default="infer_results")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--scenario", default=DEFAULT_SCENARIO)
    parser.add_argument("--strategy", default=DEFAULT_STRATEGY)
    parser.add_argument("--context_variant", default=DEFAULT_CONTEXT_VARIANT)
    parser.add_argument(
        "--temperatures",
        nargs="+",
        type=float,
        default=list(DEFAULT_TEMPERATURES),
        help="Temperature values to panel (default: 0.0 0.7)",
    )
    parser.add_argument("--plots_dir", default=PLOTS_DIR)
    parser.add_argument(
        "--envelope",
        action="store_true",
        help="Draw KDE envelope around swarm (writes *_swarm_envelope*.png)",
    )
    parser.add_argument(
        "--envelope-bw-factor",
        type=float,
        default=0.38,
        help="KDE bandwidth multiplier for envelope curve (default: 0.38)",
    )
    args = parser.parse_args()

    run(
        args.input_dir,
        model=args.model,
        scenario=args.scenario,
        strategy=args.strategy,
        context_variant=args.context_variant,
        temperatures=tuple(args.temperatures),
        plots_dir=args.plots_dir,
        envelope=args.envelope,
        envelope_bw_factor=args.envelope_bw_factor,
    )
