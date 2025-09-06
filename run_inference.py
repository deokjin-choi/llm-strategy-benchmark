"""
run_inference.py
Main entry point for running LLM inference on business scenarios.
- Reads configs/param.yaml and configs/models.yaml
- Supports home (Ollama), company (vLLM CUDA), amd_cloud (vLLM ROCm)
- Saves results to infer_results/{ENV}/
"""

import requests, itertools, pandas as pd, json, time, os, glob, argparse
import logging, tqdm, re, yaml
from multiprocessing import Pool, cpu_count

from infer_pipeline.utils import try_parse_json, build_prompt, call_ollama_api

HEADERS = {"Content-Type": "application/json"}

# ---------------------------
# 1) Config 로드
# ---------------------------
def load_params():
    with open("configs/param.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

PARAMS = load_params()
ENV = PARAMS["environment"]

# ---------------------------
# 2) Runner function
# ---------------------------
def run_single_case(case_params):
    scenario_name, scenario_data, problem_type, model, temp, max_tok, subset, out_dir, url, repeats = case_params
    model_name_short = model.split("/")[-1]

    extra_label = "; ".join(subset) if subset else "no_context"
    filename = f"{scenario_name}__{problem_type}__{model_name_short}__T{temp}__MaxTok{max_tok}__Ctx{''.join(str(hash(extra_label)))}__.csv"
    filepath = os.path.join(out_dir, filename)

    if os.path.exists(filepath):
        logging.info(f"Already exists, skipping: {filename}")
        return

    logging.info(f"Starting: {filename}")

    problem_text = scenario_data[f"problem_{problem_type}"]
    prompt = build_prompt(
        problem=problem_text,
        included_tags=subset,
        context_blocks=scenario_data["context_blocks"],
        execution_options=scenario_data["execution_options"],
    )

    results_for_case = []
    for repeat in range(repeats):
        content = "ERROR: Failed to get response."
        try:
            if ENV == "home":  # Ollama 전용
                # utils.py에 있는 call_ollama_api 사용
                content = call_ollama_api(
                    args.model,  # "mistral"
                    prompt,
                    temp,
                    max_tok,
                    url
                )
            else:  # vLLM (OpenAI 호환)
                payload = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": "You are an expert in technology strategy. Keep answers concise and structured."},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": temp,
                    "max_tokens": max_tok,
                    "stream": False,
                }
                r = requests.post(url, headers=HEADERS, json=payload, timeout=90)
                r.raise_for_status()
                content = r.json()["choices"][0]["message"]["content"]

        except Exception as e:
            content = f"ERROR: {e}"

        parsed, parse_err = try_parse_json(content)
        results_for_case.append(
            {
                "scenario": scenario_name,
                "problem_type": problem_type,
                "repeat": repeat + 1,
                "Model": model,
                "Temperature": temp,
                "Max Tokens": max_tok,
                "Context Tags": extra_label,
                "Num Context": len(subset),
                "Prompt": prompt,
                "Raw Output": content,
                "Parse Error": parse_err,
                "Chosen Option": parsed.get("chosen_option") if parsed else None,
                "Standard Mapping": parsed.get("standard_mapping") if parsed else None,
                "Rationale": json.dumps(parsed.get("rationale"), ensure_ascii=False) if parsed else None,
                "Key Signals Used": "; ".join(parsed.get("key_signals_used"))
                if (parsed and isinstance(parsed.get("key_signals_used"), list))
                else None,
            }
        )
        time.sleep(0.01)

    df_case = pd.DataFrame(results_for_case)
    df_case.to_csv(filepath, index=False)
    logging.info(f"Saved {len(df_case)} repeats to {filepath}")

# ---------------------------
# 3) Main Experiment Runner
# ---------------------------
def run_all_scenarios(json_file, out_dir, model, url):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    with open(json_file, "r", encoding="utf-8") as f:
        SCENARIOS = json.load(f)

    jobs = []
    for scenario_name, scenario_data in SCENARIOS.items():
        ctx_tags = list(scenario_data["context_blocks"].keys())
        ctx_combos = list(itertools.chain.from_iterable(itertools.combinations(ctx_tags, r) for r in range(len(ctx_tags) + 1)))
        for problem_type in ["generic", "specific"]:
            if f"problem_{problem_type}" not in scenario_data:
                continue
            temps = PARAMS["temperatures"][ENV]
            for temp in temps:
                max_tok = PARAMS["max_tokens"]
                for subset in ctx_combos:
                    jobs.append((scenario_name, scenario_data, problem_type, model, temp, max_tok, subset, out_dir, url, PARAMS["repeats"][ENV]))

    logging.info(f"Total jobs: {len(jobs)}")

    if PARAMS["parallel"][ENV]:
        with Pool(processes=min(cpu_count(), PARAMS.get("max_parallel_processes", 8))) as pool:
            list(tqdm.tqdm(pool.imap_unordered(run_single_case, jobs), total=len(jobs)))
    else:
        for job in tqdm.tqdm(jobs):
            run_single_case(job)

# ---------------------------
# 4) Entry
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model key (mistral, qwen, llama, gemma, deepseek, yi)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # models.yaml에서 endpoint 가져오기
    with open("configs/models.yaml", "r", encoding="utf-8") as f:
        models_cfg = yaml.safe_load(f)

    if args.model not in models_cfg:
        raise ValueError(f"Unknown model key: {args.model}")

    model_name = models_cfg[args.model]["name"]
    url = models_cfg[args.model]["endpoints"].get(f"vllm-{ENV}", models_cfg[args.model]["endpoints"].get("ollama"))

    # 결과 저장 경로
    json_files = glob.glob("input_scenario/*.json")
    for jf in json_files:
        base = os.path.splitext(os.path.basename(jf))[0]
        out_dir = os.path.join("infer_results", ENV, base)
        run_all_scenarios(jf, out_dir, model_name, url)
