"""
RQ1: expected-direction check for context variants (JIIS revision).

For each context variant v in {opp_focus, count_fact, randomized_numbers},
computes on pooled distributions (same pooling as tvd_from_base_ci.py):
  - per-strategy share change  Δp_k = p_k(v) - p_k(base)
  - posture-share change       Δπ_X = sum_{k in X} Δp_k
        X = Prospector {Technology Leadership, Diversification}
            Defender   {Niche Focus, Maintain, Retrenchment}
  - TVD(v, base)
with nonparametric bootstrap 95% CIs (responses resampled within v and base).

Numerical-stability reference: TVD between odd- and even-numbered repeats of
the base variant (same pooling), i.e. the shift produced by sampling alone.

competitive_dynamics is excluded from the revision and is not analysed here.

Outputs
-------
  final_results/summary/rq1_posture_shift.csv
  final_results/summary/rq1_strategy_shift.csv
  final_results/summary/rq1_split_half_tvd.csv

Usage
-----
  python -m result_analysis.rq1_posture_shift
"""

from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

try:
    from result_analysis.jsd_from_base_ci import STANDARDIZATION_MAP
    from result_analysis.rationale_analysis import valid_strategies
except ImportError:
    from jsd_from_base_ci import STANDARDIZATION_MAP
    from rationale_analysis import valid_strategies

SUMMARY_DIR = "./final_results/summary"
BOOTSTRAP_N = 10_000
BOOTSTRAP_SEED = 42

VARIANTS = ["opp_focus", "count_fact", "randomized_numbers"]
POSTURES = {
    "Prospector": ["Technology Leadership", "Diversification"],
    "Defender": ["Niche Focus", "Maintain", "Retrenchment"],
}


def _variant_from_path(path: str) -> str:
    name = os.path.basename(path).replace(".csv", "")
    return name.replace("scenarios_", "") if name.startswith("scenarios_") else "base"


def load_codes(input_dir: str = "infer_results") -> pd.DataFrame:
    frames = []
    for path in glob.glob(os.path.join(input_dir, "*scenarios*.csv")):
        variant = _variant_from_path(path)
        if variant not in ["base", *VARIANTS]:
            continue
        d = pd.read_csv(path, usecols=["Standard Mapping", "repeat"])
        d["variant"] = variant
        d["Standard Mapping"] = d["Standard Mapping"].replace(STANDARDIZATION_MAP)
        d = d[d["Standard Mapping"].isin(valid_strategies)].copy()
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    codes = {s: i for i, s in enumerate(valid_strategies)}
    df["code"] = df["Standard Mapping"].map(codes).astype(np.int8)
    df["repeat"] = pd.to_numeric(df["repeat"], errors="coerce")
    return df


def _dist(codes: np.ndarray) -> np.ndarray:
    return np.bincount(codes, minlength=len(valid_strategies)) / len(codes)


def _tvd(p: np.ndarray, q: np.ndarray) -> float:
    return 0.5 * float(np.abs(p - q).sum())


def _posture_index() -> dict[str, np.ndarray]:
    idx = {s: i for i, s in enumerate(valid_strategies)}
    return {g: np.array([idx[s] for s in members]) for g, members in POSTURES.items()}


def bootstrap_shift(base: np.ndarray, var: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Return (n_boot, 7) bootstrap replicates of p(var) - p(base)."""
    nb, nv = len(base), len(var)
    out = np.empty((BOOTSTRAP_N, len(valid_strategies)))
    for b in range(BOOTSTRAP_N):
        out[b] = _dist(var[rng.integers(0, nv, nv)]) - _dist(base[rng.integers(0, nb, nb)])
    return out


def run(input_dir: str = "infer_results") -> None:
    os.makedirs(SUMMARY_DIR, exist_ok=True)
    df = load_codes(input_dir)
    base = df.loc[df["variant"] == "base", "code"].to_numpy()
    p_base = _dist(base)
    post_idx = _posture_index()
    rng = np.random.default_rng(BOOTSTRAP_SEED)

    strat_rows, post_rows = [], []
    for v in VARIANTS:
        var = df.loc[df["variant"] == v, "code"].to_numpy()
        obs = _dist(var) - p_base
        boot = bootstrap_shift(base, var, rng)
        for j, s in enumerate(valid_strategies):
            strat_rows.append({
                "variant": v, "strategy": s,
                "p_base": p_base[j], "p_variant": p_base[j] + obs[j],
                "delta": obs[j],
                "ci_lower": np.percentile(boot[:, j], 2.5),
                "ci_upper": np.percentile(boot[:, j], 97.5),
            })
        for g, ix in post_idx.items():
            bs = boot[:, ix].sum(axis=1)
            post_rows.append({
                "variant": v, "posture": g, "delta": obs[ix].sum(),
                "ci_lower": np.percentile(bs, 2.5), "ci_upper": np.percentile(bs, 97.5),
            })
        tvd_boot = 0.5 * np.abs(boot).sum(axis=1)
        post_rows.append({
            "variant": v, "posture": "TVD", "delta": 0.5 * np.abs(obs).sum(),
            "ci_lower": np.percentile(tvd_boot, 2.5), "ci_upper": np.percentile(tvd_boot, 97.5),
        })

    b = df[df["variant"] == "base"]
    odd = b.loc[b["repeat"] % 2 == 1, "code"].to_numpy()
    even = b.loc[b["repeat"] % 2 == 0, "code"].to_numpy()
    split = pd.DataFrame([{
        "variant": "base", "n_odd": len(odd), "n_even": len(even),
        "split_half_tvd": _tvd(_dist(odd), _dist(even)),
    }])

    pd.DataFrame(strat_rows).to_csv(os.path.join(SUMMARY_DIR, "rq1_strategy_shift.csv"), index=False)
    pd.DataFrame(post_rows).to_csv(os.path.join(SUMMARY_DIR, "rq1_posture_shift.csv"), index=False)
    split.to_csv(os.path.join(SUMMARY_DIR, "rq1_split_half_tvd.csv"), index=False)
    print(pd.DataFrame(post_rows).round(4).to_string(index=False))
    print(pd.DataFrame(strat_rows).round(4).to_string(index=False))
    print(split.to_string(index=False))


if __name__ == "__main__":
    run()
