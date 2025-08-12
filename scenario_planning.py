# scenario_planning_gb_flatjson.py
# Good/Bad 기반 시나리오 플래닝 자동화 (배열 금지, 평평한 JSON)
# - 입력: scenarios.json (problem_generic/specific + context_blocks)
# - 출력: 조합별 시나리오 CSV + 대표 3대 시나리오 분류 JSON

import os, json, time, hashlib, logging, itertools, re
import requests
import pandas as pd
from typing import Dict, List, Tuple

# -----------------------------
# 0) vLLM 환경 설정 (Input -> Output)
# Input : 없음(상수)
# Process : 엔드포인트/파라미터 정의
# Output : 전역 상수들
# -----------------------------
MODEL_ENDPOINTS = {
    "mistralai/Mistral-7B-Instruct-v0.3": "http://localhost:3001/v1/chat/completions",
    "Qwen/Qwen2.5-14B-Instruct": "http://localhost:3002/v1/chat/completions",
    "meta-llama/Meta-Llama-3.1-8B-Instruct": "http://localhost:3003/v1/chat/completions",
}
HEADERS = {"Content-Type": "application/json"}

SCENARIO_FILE = "scenarios.json"
RESULTS_DIR = "./scenario_planning_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

TEMPERATURE = 0.0
MAX_TOKENS = 512
TIMEOUT = 90

# 조합/분류 상한 (필요시 조절)
COMBO_LIMIT = None     # 예: 128
CLASSIFY_MAX = 48      # 분류에 넘길 최대 후보 수

# -----------------------------
# 유틸: JSON 관용 파서 (Input -> Output)
# Input : 문자열 s
# Process: 완전파싱 실패 시 본문 내 첫 '{'~마지막 '}' 구간 재시도
# Output: (dict|None, error_message|None)
# -----------------------------
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

# -----------------------------
# 카테고리 매핑 (Input -> Output)
# Input : 카테고리 문자열
# Process: 보고서 친화 라벨로 매핑
# Output: 라벨 문자열
# -----------------------------
CATEGORY_NAME_MAP = {
    "Market": "market situation",
    "Technology": "technology environment",
    "Finance": "financial status",
    "Competitors": "competitive landscape",
    "Competition": "competitive landscape",
    "Policy": "policy and regulation",
    "Regulation": "policy and regulation",
    "Customer Response": "customer response",
    "Manufacturing": "manufacturing and supply chain",
    "Strategy": "strategic context",
}

# -----------------------------
# 카테고리 추출 (Input -> Output)
# Input : 컨텍스트 태그 문자열 예: "[Market] Global ..."
# Process: 대괄호 내부 텍스트 추출
# Output: 카테고리 문자열 (없으면 "General")
# -----------------------------
def extract_category(tag: str) -> str:
    m = re.match(r"\s*\[(.*?)\]\s*", tag)
    if m:
        return m.group(1).strip()
    return "General"

# -----------------------------
# 컨텍스트 그룹핑 (Input -> Output)
# Input : context_blocks(dict)
# Process: 카테고리별로 (tag,text) 리스트로 그룹화
# Output: grouped_context(dict[str, list[(tag,text)]])
# -----------------------------
def group_context(context_blocks: Dict[str, str]) -> Dict[str, List[Tuple[str, str]]]:
    grouped = {}
    for tag, text in context_blocks.items():
        cat = extract_category(tag)
        grouped.setdefault(cat, []).append((tag, text))
    return grouped

# -----------------------------
# 포함 섹션 생성 (Input -> Output)
# Input : grouped_context
# Process: 존재하는 카테고리 기반으로 섹션명 생성 + risks/opportunities 추가
# Output: 섹션명 리스트
# -----------------------------
def build_include_sections(grouped: Dict[str, List[Tuple[str, str]]]) -> List[str]:
    sections = []
    for cat in grouped.keys():
        label = CATEGORY_NAME_MAP.get(cat, cat.lower())
        if label not in sections:
            sections.append(label)
    for must in ["risks", "opportunities"]:
        if must not in sections:
            sections.append(must)
    return sections

# -----------------------------
# Good/Bad 조합 생성 (Input -> Output)
# Input : ctx_tags(list[str])
# Process: 각 태그별 True(GOOD)/False(BAD) 모든 조합 생성
# Output: generator of dict {tag: bool}
# -----------------------------
def iter_good_bad_combos(ctx_tags: List[str]):
    for bits in itertools.product([True, False], repeat=len(ctx_tags)):
        yield {t: b for t, b in zip(ctx_tags, bits)}

# -----------------------------
# Good/Bad 문자열 (Input -> Output)
# Input : bool
# Process: True->GOOD, False->BAD
# Output: "GOOD"|"BAD"
# -----------------------------
def human_gb(b: bool) -> str:
    return "GOOD" if b else "BAD"

# -----------------------------
# 조합 시그니처 (Input -> Output)
# Input : combo(dict[tag->bool])
# Process: 정렬된 태그=0/1 문자열 → SHA1 10자리 해시
# Output: scenario_id(짧은 해시)
# -----------------------------
def combo_signature(combo: Dict[str, bool]) -> str:
    items = [f"{k}={int(v)}" for k, v in sorted(combo.items(), key=lambda kv: kv[0])]
    s = "|".join(items)
    h = hashlib.sha1(s.encode("utf-8")).hexdigest()[:10]
    return h

# -----------------------------
# 프롬프트 생성 (Input -> Output)
# Input : problem(str), context_blocks(dict), combo(dict[tag->bool]), scenario_id(str)
# Process: 카테고리 동적 섹션 + 동일 카테고리 다중항목 모두 표기(GOOD/BAD 라벨)
#          배열([]) 금지, <> 금지, 값 문자열에서 쉼표 사용 지양 지시
# Output: prompt(str)
# -----------------------------
def build_prompt_scenario(problem: str,
                          context_blocks: Dict[str, str],
                          combo: Dict[str, bool],
                          scenario_id: str) -> str:
    grouped = group_context(context_blocks)
    include_sections = build_include_sections(grouped)

    ctx_lines = []
    for cat, items in grouped.items():
        for idx, (tag, text) in enumerate(items, start=1):
            gb = human_gb(combo[tag])
            cat_idx = f"{cat}-{idx}" if len(items) > 1 else f"{cat}"
            ctx_lines.append(f"- [{cat_idx}] {tag} — {gb}: {text}")

    include_text = "\n".join(f"- {sec}" for sec in include_sections)
    ctx_text = "\n".join(ctx_lines)

    # 평평한 JSON 스키마 (배열 없음, <> 금지, 값 문자열에서 쉼표 사용 지양)
    return f"""
You are a senior scenario planning analyst.

Problem:
{problem}

Scenario ID:
{scenario_id}

Context conditions for the company (each marked GOOD or BAD):
{ctx_text}

Write a plausible scenario narrative for this exact set of conditions.

Requirements:
- Do NOT give recommendations or action items. Only describe unfolding events.
- Cover the following sections if relevant to provided categories:
{include_text}
- STRICTLY AVOID using angle brackets.
- STRICTLY AVOID using any array or list syntax.
- AVOID commas inside string values if possible.

Return STRICT JSON with exactly these keys and only string values:
{{
  "scenario_id": "{scenario_id}",
  "narrative": "coherent scenario narrative without angle brackets and avoiding commas where possible",
  "risk1": "key risk one without angle brackets and avoiding commas",
  "risk2": "key risk two without angle brackets and avoiding commas",
  "risk3": "key risk three without angle brackets and avoiding commas",
  "opportunity1": "key opportunity one without angle brackets and avoiding commas",
  "opportunity2": "key opportunity two without angle brackets and avoiding commas",
  "opportunity3": "key opportunity three without angle brackets and avoiding commas"
}}
No extra keys. No markdown. No arrays. No angle brackets.
""".strip()

# -----------------------------
# LLM 호출 (Input -> Output)
# Input : model(str), prompt(str)
# Process: vLLM Chat Completions 호출
# Output: content(str)
# -----------------------------
def call_llm(model: str, prompt: str) -> str:
    url = MODEL_ENDPOINTS[model]
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are an expert scenario planning analyst. Be precise, structured, and factual."},
            {"role": "user", "content": prompt}
        ],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "stream": False,
    }
    r = requests.post(url, headers=HEADERS, json=payload, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

# -----------------------------
# 요약 생성(분류 입력용) (Input -> Output)
# Input : text(str), limit_chars(int)
# Process: 공백 정규화 후 길이 제한
# Output: 짧은 요약 문자열
# -----------------------------
def summarize_for_classification(text: str, limit_chars: int = 260) -> str:
    txt = re.sub(r"\s+", " ", (text or "")).strip()
    if len(txt) <= limit_chars:
        return txt
    return txt[:limit_chars].rsplit(" ", 1)[0] + "…"

# -----------------------------
# 3대 시나리오 분류 (Input -> Output)
# Input : model(str), scenario_name(str), items(list[dict{scenario_id, summary, gb}])
# Process: LLM 호출로 most_probable/worst_case/best_case 선정
#          배열 금지, <> 금지, 값에서 쉼표 지양
# Output: dict (분류 결과)
# -----------------------------
def classify_top3(model: str, scenario_name: str, items: List[Dict]) -> Dict:
    subset = items
    if CLASSIFY_MAX is not None and len(items) > CLASSIFY_MAX:
        subset = items[:CLASSIFY_MAX]

    payload_obj = {
        "scenario_name": scenario_name,
        "candidates": [
            {
                "scenario_id": it["scenario_id"],
                "summary": it["summary"],
                "good_bad": {k: ("GOOD" if v else "BAD") for k, v in it["gb"].items()}
            }
            for it in subset
        ]
    }

    prompt = f"""
You will classify scenario narratives into three buckets without using arrays and without angle brackets.

Here are candidates for scenario "{scenario_name}":
{json.dumps(payload_obj, ensure_ascii=False, indent=2)}

Select exactly:
1) Most Probable Scenario (highest plausibility),
2) Worst Case Scenario (most damaging realistic downside),
3) Best Case Scenario (low probability but highest positive impact).

Return STRICT JSON with only string values and NO arrays:
{{
  "most_probable": "scenario_id of the most probable",
  "worst_case": "scenario_id of the worst case",
  "best_case": "scenario_id of the best case",
  "rationale_most_probable": "one line reason avoiding commas",
  "rationale_worst_case": "one line reason avoiding commas",
  "rationale_best_case": "one line reason avoiding commas"
}}
No extra keys. No arrays. No angle brackets.
""".strip()

    raw = call_llm(model, prompt)
    parsed, err = try_parse_json(raw)
    if not parsed:
        logging.warning(f"[CLASSIFY] JSON parse failed: {err}\nRaw: {raw[:400]}")
        return {"error": "parse_failed", "raw": raw}
    return parsed

# -----------------------------
# 단일 시나리오 실행 (Input -> Output)
# Input : scenario_name(str), scenario_data(dict), problem_type("generic"|"specific"), model(str)
# Process: Good/Bad 모든 조합에 대해 시나리오 생성 → CSV 저장 → 분류 JSON 생성
# Output: 파일 저장 경로들(로그)
# -----------------------------
def run_one(scenario_name: str, scenario_data: Dict, problem_type: str, model: str):
    problem_key = f"problem_{problem_type}"
    if problem_key not in scenario_data:
        return

    problem_text = scenario_data[problem_key]
    context_blocks = scenario_data["context_blocks"]
    ctx_tags = list(context_blocks.keys())

    combos_iter = iter_good_bad_combos(ctx_tags)
    combos = list(itertools.islice(combos_iter, COMBO_LIMIT)) if COMBO_LIMIT else list(combos_iter)

    rows = []
    cls_items = []

    logging.info(f"[{scenario_name}/{problem_type}/{model}] combos={len(combos)}")

    for combo in combos:
        sid = combo_signature(combo)
        prompt = build_prompt_scenario(problem_text, context_blocks, combo, sid)

        try:
            raw = call_llm(model, prompt)
        except Exception as e:
            logging.error(f"LLM call error: {e}")
            raw = f"ERROR: {e}"

        parsed, perr = try_parse_json(raw)

        # 결과 레코드
        rows.append({
            "scenario": scenario_name,
            "problem_type": problem_type,
            "model": model,
            "scenario_id": sid,
            "combo_json": json.dumps({k: int(v) for k, v in combo.items()}, ensure_ascii=False),
            "prompt": prompt,
            "raw": raw,
            "parse_error": perr,
            "narrative": parsed.get("narrative") if parsed else None,
            "risk1": parsed.get("risk1") if parsed else None,
            "risk2": parsed.get("risk2") if parsed else None,
            "risk3": parsed.get("risk3") if parsed else None,
            "opportunity1": parsed.get("opportunity1") if parsed else None,
            "opportunity2": parsed.get("opportunity2") if parsed else None,
            "opportunity3": parsed.get("opportunity3") if parsed else None,
        })

        # 분류용 요약
        summary_src = parsed.get("narrative") if parsed else raw
        cls_items.append({
            "scenario_id": sid,
            "gb": combo,
            "summary": summarize_for_classification(summary_src)
        })

        time.sleep(0.01)

    # CSV 저장
    out_csv = os.path.join(RESULTS_DIR, f"{scenario_name}__{problem_type}__{model.split('/')[-1]}.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    logging.info(f"Saved: {out_csv} ({len(rows)} rows)")

    # 3대 시나리오 분류
    try:
        classify = classify_top3(model, scenario_name, cls_items)
    except Exception as e:
        logging.error(f"Classification error: {e}")
        classify = {"error": str(e)}

    out_json = os.path.join(RESULTS_DIR, f"{scenario_name}__{problem_type}__{model.split('/')[-1]}__classification.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(classify, f, ensure_ascii=False, indent=2)
    logging.info(f"Saved: {out_json}")

# -----------------------------
# 전체 실행 (Input -> Output)
# Input : scenarios.json 파일
# Process: 모든 시나리오×문제타입×모델 반복 실행
# Output: CSV/JSON 파일들 생성
# -----------------------------
def run_all():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    if not os.path.exists(SCENARIO_FILE):
        raise FileNotFoundError(f"{SCENARIO_FILE} not found")

    with open(SCENARIO_FILE, "r", encoding="utf-8") as f:
        scenarios = json.load(f)

    for scenario_name, scenario_data in scenarios.items():
        if "context_blocks" not in scenario_data:
            logging.warning(f"[{scenario_name}] No context_blocks, skip")
            continue

        for problem_type in ["generic", "specific"]:
            if f"problem_{problem_type}" not in scenario_data:
                continue

            for model in MODEL_ENDPOINTS.keys():
                try:
                    run_one(scenario_name, scenario_data, problem_type, model)
                except Exception as e:
                    logging.error(f"Failed: {scenario_name}/{problem_type}/{model} - {e}")

if __name__ == "__main__":
    run_all()
