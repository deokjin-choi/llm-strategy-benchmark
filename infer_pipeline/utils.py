import yaml, os
import json, re
import requests, json

def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# -----------------------------
# 환경/모델 config 로드
# -----------------------------
CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "configs")
ENV_CONFIG = load_yaml(os.path.join(CONFIG_DIR, "environment.yaml"))
MODEL_CONFIG = load_yaml(os.path.join(CONFIG_DIR, "models.yaml"))

# -----------------------------
# 유틸 함수
# -----------------------------
def get_active_models(env: str):
    """환경별 실행 가능한 모델 리스트 반환"""
    env_cfg = ENV_CONFIG[env]
    model_scope = env_cfg["models"]   # "small" or "full"

    active_models = []
    for key, cfg in MODEL_CONFIG.items():
        # 모델이 비활성화면 스킵
        if not cfg.get("enabled", True):
            continue

        endpoint = cfg["endpoints"].get(env_cfg["runtime"])
        if endpoint:  # null이면 해당 환경 불가
            active_models.append({
                "key": key,
                "name": cfg["name"],
                "endpoint": endpoint,
                "runtime": env_cfg["runtime"]
            })

    return active_models

def get_model_endpoint(model_key: str, env: str):
    """환경에 맞는 특정 모델 endpoint 반환"""
    env_cfg = ENV_CONFIG[env]
    runtime = env_cfg["runtime"]

    if model_key not in MODEL_CONFIG:
        raise ValueError(f"Unknown model: {model_key}")

    endpoint = MODEL_CONFIG[model_key]["endpoints"].get(runtime)
    if not endpoint:
        raise ValueError(f"Model {model_key} not supported on {env} ({runtime})")

    return endpoint

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
        '{"chosen_option": "<MUST be exactly ONE of A, B, C, D, E, F, G (choose ONLY ONE)>", '
        '"standard_mapping": "...", '
        '"rationale": "<3-4 concise sentences>", '
        '"key_signals_used": ["<copy EXACTLY the full tag+short title from the provided context blocks>"]}\n'
        "\n"
        "CRITICAL INSTRUCTIONS:\n"
        "- Your ENTIRE output must be ONLY a valid JSON object.\n"
        "- Do NOT include explanations, preambles, markdown, or text outside the JSON.\n"
        "- The first character must be '{' and the last character must be '}'.\n"
        "\n"
        "RULES FOR 'key_signals_used':\n"
        "- It must be an array of strings.\n"
        "- Each string MUST be the EXACT FULL TAG AND SHORT TITLE from the provided context blocks.\n"
        "- Example: If the context block is '[Market] EV share: <0.02% of global new car sales.', "
        "then output exactly '[Market] EV share: <0.02% of global new car sales.'\n"
        "- Do NOT shorten to just 'Market'.\n"
        "- Do NOT copy the full descriptive explanation, only the tag+short title.\n"
        "- Do NOT invent or modify tags.\n"
        "- If no context was used, return an empty array [].\n"
    )

    return (
        "You are a senior technology strategy analyst. Given the fixed problem and a subset of context blocks (which may be empty), "
        "select the single most appropriate execution option and justify it clearly.\n\n"
        f"Problem (ALWAYS INCLUDED):\n{problem}\n\n"
        f"{ctx_text}\n\n{option_text}\n{schema_text}"
    )


def try_parse_json(s: str):
    try:
        return json.loads(s), None
    except Exception as e:
        match = re.search(r"\{.*\}", s, re.DOTALL)
        if match:
            try:
                return json.loads(match.group()), None
            except Exception as inner_e:
                return None, f"Regex extracted but parse failed: {inner_e}"
        return None, f"Parse failed: {e}"
    

HEADERS = {"Content-Type": "application/json"}

def call_ollama_api(model: str, prompt: str, temp: float, max_tok: int, url: str) -> str:
    """
    Call Ollama API (/api/generate).
    - Handles NDJSON (newline-delimited JSON) streaming output
    - Concatenates all "response" parts into one string
    """
    payload = {
        "model": model,  # e.g., "mistral"
        "prompt": prompt,
        "options": {
            "temperature": temp,
            "num_predict": max_tok
        }
    }

    try:
        r = requests.post(url, headers=HEADERS, json=payload, timeout=90, stream=True)
        r.raise_for_status()
    except Exception as e:
        return f"ERROR: {e}"

    content_parts = []
    for line in r.iter_lines():
        if line:
            try:
                data = json.loads(line.decode("utf-8"))
                if "response" in data:
                    content_parts.append(data["response"])
            except Exception:
                continue
    return "".join(content_parts).strip()