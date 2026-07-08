"""
Bootstrap 95% CI for JSD from base (scenario-level context variants vs base).

Also exports Table 2 companion metrics (entropy, jsd_from_base, spearman_vs_base).

Outputs
-------
  final_results/summary/jsd_from_base_ci.csv

Usage
-----
  python -m result_analysis.jsd_from_base_ci
"""

from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

from result_analysis.overall_results import compute_ratio_metrics, js_divergence, load_overall_ratio
from result_analysis.rationale_analysis import valid_strategies

SUMMARY_DIR = "./final_results/summary"
BOOTSTRAP_N = 10_000
BOOTSTRAP_SEED = 42

STANDARDIZATION_MAP = {
    "C → Open Innovation": "Open Innovation",
    "A → Technology Leadership": "Technology Leadership",
    "C -> Open Innovation": "Open Innovation",
    "A -> Technology Leadership": "Technology Leadership",
}


def _load_pooled_strategy_codes(input_dir: str = "infer_results") -> tuple[pd.DataFrame, np.ndarray]:
    """Load pooled inference rows; return DataFrame and int-coded strategy arrays per scenario."""
    frames = []
    for file_path in glob.glob(os.path.join(input_dir, "*scenarios*.csv")):
        df = pd.read_csv(file_path, usecols=["Standard Mapping"])
        file_name = os.path.basename(file_path).replace(".csv", "")
        if file_name.endswith("_scenarios"):
            scenario_type = "base"
        else:
            scenario_type = file_name.split("_", 1)[-1]
            if scenario_type.startswith("scenarios_"):
                scenario_type = scenario_type.replace("scenarios_", "")

        df["scenario_type"] = scenario_type
        df["Standard Mapping"] = df["Standard Mapping"].replace(STANDARDIZATION_MAP)
        df = df[df["Standard Mapping"].isin(valid_strategies)].copy()
        frames.append(df)

    if not frames:
        raise FileNotFoundError(f"No scenario CSV files found in {input_dir}")

    combined = pd.concat(frames, ignore_index=True)
    codes = {s: i for i, s in enumerate(valid_strategies)}
    combined["code"] = combined["Standard Mapping"].map(codes).astype(np.int8)
    return combined, np.array(valid_strategies)


def _dist_from_codes(codes: np.ndarray, n_strategies: int = 7) -> np.ndarray:
    n = len(codes)
    if n == 0:
        return np.zeros(n_strategies, dtype=float)
    return np.bincount(codes, minlength=n_strategies) / n


def bootstrap_jsd_ci(
    base_codes: np.ndarray,
    variant_codes: np.ndarray,
    *,
    n_boot: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float, float]:
    """Nonparametric bootstrap 95% CI for JSD(base, variant) from row resampling."""
    rng = np.random.default_rng(seed)
    nb, nv = len(base_codes), len(variant_codes)
    js_samples = np.empty(n_boot, dtype=float)

    for b in range(n_boot):
        bc = base_codes[rng.integers(0, nb, nb)]
        vc = variant_codes[rng.integers(0, nv, nv)]
        js_samples[b] = js_divergence(_dist_from_codes(bc), _dist_from_codes(vc))

    return (
        float(np.mean(js_samples)),
        float(np.percentile(js_samples, 2.5)),
        float(np.percentile(js_samples, 97.5)),
    )


def compute_jsd_from_base_ci(input_dir: str = "infer_results") -> pd.DataFrame:
    """Point estimates from pooled ratios; JSD 95% CI from bootstrap resampling."""
    ratio_path = os.path.join(SUMMARY_DIR, "analysis_overall_ratio.csv")
    if not os.path.exists(ratio_path):
        raise FileNotFoundError(
            f"{ratio_path} not found. Run make_summary or run_all_analysis first."
        )

    metrics = compute_ratio_metrics(load_overall_ratio(ratio_path)).reset_index()
    pooled, _ = _load_pooled_strategy_codes(input_dir)
    base_codes = pooled.loc[pooled["scenario_type"] == "base", "code"].to_numpy()

    ci_rows = []
    for scenario in metrics["scenario"]:
        if scenario == "base":
            ci_rows.append({"scenario": scenario, "jsd_ci_lower": 0.0, "jsd_ci_upper": 0.0})
            continue
        var_codes = pooled.loc[pooled["scenario_type"] == scenario, "code"].to_numpy()
        _, lo, hi = bootstrap_jsd_ci(base_codes, var_codes)
        ci_rows.append({"scenario": scenario, "jsd_ci_lower": lo, "jsd_ci_upper": hi})

    out = metrics.merge(pd.DataFrame(ci_rows), on="scenario")
    out["jsd_95ci"] = out.apply(
        lambda r: "—" if r["scenario"] == "base"
        else f"[{r['jsd_ci_lower']:.4f}, {r['jsd_ci_upper']:.4f}]",
        axis=1,
    )
    return out


def run(input_dir: str = "infer_results") -> pd.DataFrame:
    os.makedirs(SUMMARY_DIR, exist_ok=True)
    table = compute_jsd_from_base_ci(input_dir)
    out_path = os.path.join(SUMMARY_DIR, "jsd_from_base_ci.csv")
    table.to_csv(out_path, index=False)
    print(f"Saved → {out_path}")
    print(table[["scenario", "entropy", "jsd_from_base", "jsd_95ci", "spearman_vs_base"]].to_string(index=False))
    return table


if __name__ == "__main__":
    run()
