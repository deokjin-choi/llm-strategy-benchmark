# scenario_propose.py (Parallel Processing Version, Updated)

import requests, itertools, pandas as pd, json, re, time, os, math, glob
from multiprocessing import Pool
import logging

# -----------------------------
# 0) vLLM endpoints (updated models)
# -----------------------------
MODEL_ENDPOINTS = {
    "mistralai/Mistral-7B-Instruct-v0.3": "http://localhost:3001/v1/chat/completions",
    "Qwen/Qwen2.5-14B-Instruct": "http://localhost:3002/v1/chat/completions",
    "meta-llama/Meta-Llama-3.1-8B-Instruct": "http://localhost:3003/v1/chat/completions",
    "google/gemma-2-9b-it": "http://localhost:3004/v1/chat/completions",
    "deepseek-ai/deepseek-7b-instruct": "http://localhost:3005/v1/chat/completions",
    "01-ai/Yi-9B-Chat": "http://localhost:3006/v1/chat/completions",
}
HEADERS = {"Content-Type": "application/json"}

REPEATS_PER_CASE = 30  # 🔥 반복 횟수 확장

# ---------------------------
# 1) Prompt Builder
# ---------------------------
def build_prompt(problem: str, included_tags: list[str], context_blocks: dict, execution_options: dict) -> str:
    if included_tags:
        ctx_text = "Additional context (subset may be empty):\n" + "\n".join(
            f"- {tag} {context_blocks[tag]}" for tag in included_tags
        )
    else:
        ctx_text = "Additional context (subset may be empty):\n- (none)"
    
    option_text = "Candidate execution options (choose EXACTLY ONE):\n"
    for k, v in execution_options.items():
        option_text += f"  {k}) {v['name']}\n"
    
    option_text += "\nStandard strategy mappings:\n"
    for k, v in execution_options.items():
        option_text += f"  {k} → {v['mapping']}\n"
    
    schema_text = (
        "Return STRICT JSON with keys exactly: "
        '{"chosen_option": "A|B|C|D|E|F|G", '
        '"standard_mapping": "...", '
        '"rationale": "<3-4 concise sentences>", '
        '"key_signals_used": ["<copy EXACTLY the full tag+short title from the provided context blocks>"]}\n'
        "\n"
        "CRITICAL INSTRUCTIONS FOR 'key_signals_used':\n"
        "- It must be an array of strings.\n"
        "- Each string MUST be the EXACT FULL TAG AND SHORT TITLE from the provided context blocks.\n"
        "- Example: If the context block is '[Market] EV share: <0.02% of global new car sales.', then output exactly '[Market] EV share: <0.02% of global new car sales.'\n"
        "- Do NOT shorten to just 'Market'.\n"
        "- Do NOT copy the full descriptive explanation, only the tag+short title.\n"
        "- Do NOT invent or modify tags.\n"
        "- If no context was used, return an empty array [].\n"
        "Do not include any extra keys or prose outside JSON."
    )

    return (
        "You are a senior technology strategy analyst. Given the fixed problem and a subset of context blocks (which may be empty), "
        "select the single most appropriate execution option and justify it clearly.\n\n"
        f"Problem (ALWAYS INCLUDED):\n{problem}\n\n"
        f"{ctx_text}\n\n{option_text}\n{schema_text}"
    )

# ---------------------------
# 2) Helper functions
# ---------------------------
def try_parse_json(s: str):
    try:
        return json.loads(s), None
    except Exception as e:
        try:
            i, j = s.find("{"), s.rfind("}")
            if i != -1 and j != -1 and j > i:
                return json.loads(s[i:j+1]), None
        except Exception:
            pass
        return None, str(e)

# ---------------------------
# 3) Runner function
# ---------------------------
def run_single_case(case_params):
    scenario_name, scenario_data, problem_type, model, temp, max_tok, subset, out_dir = case_params
    
    model_name_short = model.split('/')[-1]
    url = MODEL_ENDPOINTS[model]
    
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
        execution_options=scenario_data["execution_options"]
    )

    results_for_case = []
    for repeat in range(REPEATS_PER_CASE):
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are an expert in technology strategy. Keep answers concise and structured."},
                {"role": "user", "content": prompt}
            ],
            "temperature": temp,
            "max_tokens": max_tok,
            "stream": False,
        }
        
        content = "ERROR: Failed to get response."
        try:
            r = requests.post(url, headers=HEADERS, json=payload, timeout=90)
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"]
        except Exception as e:
            content = f"ERROR: {e}"
        
        parsed, parse_err = try_parse_json(content)
        
        results_for_case.append({
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
            "Key Signals Used": "; ".join(parsed.get("key_signals_used")) if (parsed and isinstance(parsed.get("key_signals_used"), list)) else None,
        })
        time.sleep(0.01)
    
    df_case = pd.DataFrame(results_for_case)
    df_case.to_csv(filepath, index=False)
    logging.info(f"Saved {len(df_case)} repeats to {filepath}")

# ---------------------------
# 4) Main Experiment Runner
# ---------------------------
def run_all_scenarios_parallel(json_file, out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    with open(json_file, 'r', encoding='utf-8') as f:
        SCENARIOS = json.load(f)

    jobs = []
    for scenario_name, scenario_data in SCENARIOS.items():
        ctx_tags = list(scenario_data["context_blocks"].keys())
        ctx_combos = list(itertools.chain.from_iterable(itertools.combinations(ctx_tags, r) for r in range(len(ctx_tags) + 1)))
        
        for problem_type in ["generic", "specific"]:
            if f"problem_{problem_type}" not in scenario_data:
                continue
            for model in MODEL_ENDPOINTS.keys():
                for temp in [0, 0.7]:   # 🔥 두 가지 temperature
                    max_tok = 256
                    for subset in ctx_combos:
                        jobs.append((scenario_name, scenario_data, problem_type, model, temp, max_tok, subset, out_dir))

    logging.info(f"Total jobs to run ({json_file}): {len(jobs)}")
    
    num_processes = min(os.cpu_count(), len(MODEL_ENDPOINTS))
    with Pool(processes=num_processes) as pool:
        pool.map(run_single_case, jobs)

    logging.info(f"✅ All scenarios completed for {json_file}.")

# ---------------------------
# 5) Run all JSONs
# ---------------------------
def run_all_jsons():
    json_files = glob.glob("*.json")
    for jf in json_files:
        base = os.path.splitext(os.path.basename(jf))[0]
        if base == "scenarios":
            out_dir = os.path.join("scenario_infer_results", "base")
        else:
            out_dir = os.path.join("scenario_infer_results", base)
        run_all_scenarios_parallel(jf, out_dir)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(processName)s - %(levelname)s - %(message)s')
    run_all_jsons()
