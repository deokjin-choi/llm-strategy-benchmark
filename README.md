# llm-strategy-benchmark

A scenario-based benchmark for measuring **decision sensitivity** in large language models used for strategic R&D decision support.

Most LLM evaluations certify a single answer under a fixed prompt. This project holds a strategic dilemma and a closed menu of strategy archetypes constant, then measures how recommendations and rationales shift when only semantic context, firm identity (Generic vs. Specific), or decoding settings change. The intended use is pre-deployment audit: profile a model on realistic prompt variation before it is used in organizational strategy work.

The associated manuscript is *A Scenario-Based Audit Framework for Decision Sensitivity in LLM Strategic Decision Support* (under review). This repository is the analysis code and scenario pack behind that study.

## What the benchmark does

- Six historically grounded R&D scenarios, each with a fixed dilemma and a closed strategy menu.
- Two firm-identity framings: **Generic** (unnamed firm) and **Specific** (Tesla).
- Four context variants on top of a base prompt: competition, constraint, opportunity, and numerical perturbation.
- Repeated inference across open-weight models and temperatures (`0.0`, `0.7`).
- Metrics for context shift, firm-identity shift, context–identity moderation, and matched-choice rationale divergence.
- A four-step, human-gated audit protocol assembled from those axes.

## Repository layout

```text
.
├── run_inference.py          # Entry point: prompt construction and model calls
├── configs/                  # Environment, model list, and run parameters
├── infer_pipeline/           # Prompt builder and API helpers
├── input_scenario/           # Base scenarios and context-variant JSON files
├── scripts/                  # vLLM / Ollama launch and result-combine helpers
├── result_analysis/          # Metrics, tables, and figure scripts
└── final_results/            # Aggregated summaries and plots used in the paper
```

| Path | Contents |
|---|---|
| `input_scenario/` | `scenarios.json` (base) plus `scenarios_competitive_dynamics.json`, `scenarios_count_fact.json`, `scenarios_opp_focus.json`, and `scenarios_randomized_numbers.json` |
| `configs/` | `models.yaml`, `param.yaml`, `environment.yaml` |
| `result_analysis/` | Context TVD, framing Δp, RDS, localization, and audit-framework plots |
| `final_results/plots/` | Figures corresponding to the manuscript |
| `final_results/summary/` | Tabular aggregates behind those figures |

Raw per-call inference dumps live locally under `infer_results/` and are not part of the published tree (they are large). Aggregated results needed to reproduce the figures are in `final_results/`.

## Reproduce an analysis pass

1. Place or generate inference CSVs under `infer_results/` (see `run_inference.py` and `configs/param.yaml` for the environment: Ollama, vLLM CUDA, or vLLM ROCm).
2. Build summaries and figures from `result_analysis/` (start with `run_all_analysis.py`).
3. Check outputs in `final_results/`.

Scenario JSON can be inspected without running models. Each file keeps the same dilemma and option set; only the injected context blocks differ.

## Data availability

Analysis code and aggregated results in this repository support the figures and tables. The full set of model inference outputs is not archived here because of its volume; it is available from the corresponding author upon reasonable request.
