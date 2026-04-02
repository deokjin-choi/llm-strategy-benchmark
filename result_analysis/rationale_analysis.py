#%%
import os, glob, re, time
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

valid_strategies = ["Maintain", "Retrenchment", "Niche Focus",
                    "Diversification", "Open Innovation",
                    "Fast Follower", "Technology Leadership"]

# 숫자/수치 표현 제거를 위한 불용어(필요 시 확장)
NUMERIC_RELATED_STOPWORDS = {
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "hundred", "thousand", "million", "billion", "trillion",
    "percent", "percentage", "year", "years", "month", "months",
    "usd", "dollar", "dollars", "km", "kwh", "mph", "ev"
}

# -------------------------------
# 1) 데이터 로드
# -------------------------------
def load_rationale_data(input_dir="infer_results"):
    all_files = glob.glob(os.path.join(input_dir, "*scenarios*.csv"))
    if not all_files:
        print(f"No scenario files found in {input_dir}")
        return pd.DataFrame()
    dfs = [pd.read_csv(f) for f in all_files]
    df = pd.concat(dfs, ignore_index=True)
    df["Rationale"] = df["Rationale"].fillna("")
    df["Standard Mapping"] = df["Standard Mapping"].fillna("N/A")
    df["Key Signals Used"] = df["Key Signals Used"].fillna("")

    df = df[df["Standard Mapping"].isin(valid_strategies)].copy()

    return df

# -------------------------------
# 2) 설명 품질 지표
# -------------------------------
def compute_explanation_quality(df):
    df["rationale_length"] = df["Rationale"].apply(lambda x: len(x.split()))
    df["avg_sentence_len"] = df["Rationale"].apply(
        lambda x: np.mean([len(s.split()) for s in re.split(r"[.!?]", x) if s.strip()]) if x else 0
    )
    # 어휘 다양성 = unique 단어 수 / 전체 단어 수
    df["lexical_diversity"] = df["Rationale"].apply(
        lambda x: len(set(x.lower().split())) / (len(x.split()) + 1e-6) if x else 0
    )

    summary = df.groupby("Standard Mapping")[["rationale_length", "avg_sentence_len", "lexical_diversity"]].mean()
    print("\n=== Explanation Quality Metrics ===")
    print(summary.round(3).to_string())
    return summary

def plot_explanation_quality(summary):
    save_dir = "./final_results/plots"
    os.makedirs(save_dir, exist_ok=True)

    summary.plot(kind="bar", subplots=True, layout=(1,3), figsize=(14,4), legend=False, rot=20)
    plt.suptitle("Explanation Quality by Strategy", fontsize=14)

    save_path = os.path.join(save_dir, "eval_explanation_quality.png")
    plt.savefig(save_path, dpi=300)
    plt.show()

# -------------------------------
# 3) 전략별 키워드 (TF-IDF)
# -------------------------------
def compute_rationale_keywords(df, top_k=10):
    strategy_texts = df.groupby("Standard Mapping")["Rationale"].apply(lambda x: " ".join(x)).to_dict()
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=5000,
        ngram_range=(2,2)
    )
    X = vectorizer.fit_transform(strategy_texts.values())
    feature_names = vectorizer.get_feature_names_out()
    strategies = list(strategy_texts.keys())

    keywords_dict = {}
    for i, strat in enumerate(strategies):
        row = X[i].toarray().flatten()
        top_idx = row.argsort()[::-1][:top_k]
        keywords_dict[strat] = [(feature_names[j], round(row[j],3)) for j in top_idx]

    df_keywords = pd.DataFrame({
        strat: [kw for kw,score in keywords] for strat, keywords in keywords_dict.items()
    })
    print("\n=== Top Keywords by Strategy (TF-IDF) ===")
    print(df_keywords)
    return df_keywords

def plot_rationale_keywords(df_keywords):
    save_dir = "./final_results/plots"
    os.makedirs(save_dir, exist_ok=True)
    for strat in df_keywords.columns:
        keywords = df_keywords[strat].dropna().tolist()
        fig, ax = plt.subplots(figsize=(6,4))
        ax.barh(range(len(keywords)), [1]*len(keywords))
        ax.set_yticks(range(len(keywords)))
        ax.set_yticklabels(keywords)
        ax.set_title(f"Top Keywords - {strat}")
        save_path = os.path.join(save_dir, f"keywords_{strat.replace(' ','_')}.png")
        plt.savefig(save_path, dpi=300)
        plt.close(fig)

# -------------------------------
# 4) 전략 별 차별적 키워드 (log-odds ratio)
# -------------------------------
def compute_discriminative_keywords(df, top_k=10, ngram_range=(2,2)):
    """
    전략별 Rationale에서 차별적으로 등장하는 키워드를 log-odds ratio 기반으로 추출
    """
    # 전략별 텍스트 합치기
    strategy_texts = df.groupby("Standard Mapping")["Rationale"].apply(
        lambda x: " ".join(str(t).lower() for t in x if isinstance(t, str))
    ).to_dict()

    # 벡터화 (bigram 기본)
    vectorizer = CountVectorizer(stop_words="english", ngram_range=ngram_range)
    X = vectorizer.fit_transform(strategy_texts.values())
    vocab = np.array(vectorizer.get_feature_names_out())
    counts = pd.DataFrame(X.toarray(), index=strategy_texts.keys(), columns=vocab)

    # 전체 합산
    total_counts = counts.sum(axis=0)

    results = {}
    for strat in counts.index:
        strat_counts = counts.loc[strat] + 1  # smoothing
        other_counts = (total_counts - counts.loc[strat]) + 1

        # log-odds 계산
        log_odds = np.log(strat_counts / strat_counts.sum()) - np.log(other_counts / other_counts.sum())

        # Top-k 추출
        top_keywords = pd.DataFrame({
            "keyword": vocab,
            "log_odds": log_odds
        }).sort_values("log_odds", ascending=False).head(top_k)

        results[strat] = top_keywords.reset_index(drop=True)

    return results


# -------------------------------
# 5) 신호–이유 정합성
# -------------------------------
def compute_signal_alignment(df):
    def match_ratio(row):
        signals = [s.lower() for s in row["Key Signals Used"].split(";") if s.strip()]
        rationale = row["Rationale"].lower()
        if not signals:
            return np.nan
        hits = sum(1 for s in signals if s in rationale)
        return hits / len(signals)

    df["signal_match_ratio"] = df.apply(match_ratio, axis=1)
    alignment_summary = df.groupby("Standard Mapping")["signal_match_ratio"].mean()
    print("\n=== Signal-Rationale Alignment ===")
    print(alignment_summary.round(3).to_string())
    return alignment_summary

def plot_signal_alignment(alignment_summary):
    save_dir = "./final_results/plots"
    os.makedirs(save_dir, exist_ok=True)
    alignment_summary.plot(kind="bar", figsize=(7,4), rot=20, color="skyblue")
    plt.title("Signal-Rationale Alignment by Strategy")
    save_path = os.path.join(save_dir, "eval_signal_alignment.png")
    plt.savefig(save_path, dpi=300)
    plt.show()


# -------------------------------
# 6) Tesla vs Generic 차이 분석
# -------------------------------
def filter_homogeneous_selections(
    df: pd.DataFrame,
    group_cols=None,
    min_pair_size: int = 2,
) -> pd.DataFrame:
    """
    Generic vs Specific를 '결정(Chosen Option)'이 동일한 조건에서 비교하기 위해
    - (scenario, repeat, model, temperature, num_context, chosen_option) 기준으로 묶고
    - 그 그룹 안에 generic과 specific이 모두 존재하는 경우만 남김
    """
    if group_cols is None:
        group_cols = ["scenario", "repeat", "Model", "Temperature", "Num Context", "Chosen Option"]

    # 그룹 내에 generic과 specific이 모두 있는 그룹만 남김
    def has_both_types(g):
        s = set(g["problem_type"].tolist())
        return ("generic" in s) and ("specific" in s) and (len(g) >= min_pair_size)

    keep_idx = df.groupby(group_cols, dropna=False).filter(has_both_types).index
    return df.loc[keep_idx].copy()


def compute_informative_log_odds(
    texts_a,
    texts_b,
    ngram_range=(1, 2),
    stop_words="english",
    min_df=2,
    top_k=30,
):
    """
    Monroe et al. 스타일의 informative log-odds ratio (+ prior) 구현(간단 버전).

    - prior(α)는 두 집단 전체 합산 빈도를 사용 (informative prior)
    - 출력: a에서 상대적으로 높은 키워드 top_k, b에서 상대적으로 높은 키워드 top_k
    """
    # Vectorize counts
    custom_stop = {"tesla","Tesla","company","Company","brand"}  # 필요시 더 추가
    if stop_words == "english":
        stop_words = list(ENGLISH_STOP_WORDS.union(custom_stop).union(NUMERIC_RELATED_STOPWORDS))
    elif isinstance(stop_words, (list, set)):
        stop_words = list(set(stop_words).union(custom_stop).union(NUMERIC_RELATED_STOPWORDS))

    # 숫자가 포함된 토큰을 원천적으로 제외 (알파벳 토큰만 허용)
    vec = CountVectorizer(
        stop_words=stop_words,
        ngram_range=ngram_range,
        min_df=min_df,
        token_pattern=r"(?u)\b[a-zA-Z]{2,}\b",
    )

    # ✅ vocab을 A+B 전체에서 학습
    vec.fit(texts_a + texts_b)

    Xa = vec.transform(texts_a)
    Xb = vec.transform(texts_b)

    vocab = np.array(vec.get_feature_names_out())

    ca = np.asarray(Xa.sum(axis=0)).flatten()
    cb = np.asarray(Xb.sum(axis=0)).flatten()

    # informative prior
    alpha = ca + cb
    alpha0 = alpha.sum()

    ca0 = ca.sum()
    cb0 = cb.sum()

    # smoothing 포함 log-odds
    # log((c_i + α_i) / (c0 + α0 - (c_i + α_i)))
    # difference between groups
    denom_a = (ca0 + alpha0) - (ca + alpha)
    denom_b = (cb0 + alpha0) - (cb + alpha)

    # 안정성 위해 0 방지
    eps = 1e-12
    log_odds_a = np.log((ca + alpha + eps) / (denom_a + eps))
    log_odds_b = np.log((cb + alpha + eps) / (denom_b + eps))
    delta = log_odds_a - log_odds_b

    # z-score 근사 (Monroe): 1/(c_i+α_i) + 1/(other_i+α_other_i) 형태를 단순화
    # 여기선 prior가 informative라 alpha가 크므로 delta만 써도 실무상 충분한 경우가 많음.
    # 원하면 variance까지 넣어 z-score로 바꾸셔도 됩니다.
    df_out = pd.DataFrame({"keyword": vocab, "delta_log_odds": delta})
    df_out = df_out.sort_values("delta_log_odds", ascending=False)

    top_a = df_out.head(top_k).reset_index(drop=True)
    top_b = df_out.tail(top_k).sort_values("delta_log_odds").reset_index(drop=True)

    return top_a, top_b

def _bh_fdr(p_values):
    """
    Benjamini-Hochberg FDR 보정.
    입력: 1D array-like p-values
    출력: q-values (원 순서)
    """
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order]
    q = np.empty(n, dtype=float)

    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = (ranked[i] * n) / rank
        prev = min(prev, val)
        q[i] = prev

    q_original = np.empty(n, dtype=float)
    q_original[order] = np.clip(q, 0.0, 1.0)
    return q_original


def permutation_test_discriminative_keywords(
    df: pd.DataFrame,
    group_cols=None,
    ngram_range=(2, 2),
    min_df=2,
    n_perm=300,
    top_k=20,
    random_state=42,
    mask_brand=True,
    show_progress=True,
):
    """
    Paired permutation test (generic/specific 라벨 스왑)로
    키워드 차이가 우연인지 검정.

    설계 포인트:
    - 동일 조건 페어를 유지한 채(problem_type만 스왑) permutation 수행
    - 관측 통계량:
        1) keyword별 |delta_log_odds|
        2) global: mean(|delta_log_odds|)
    """
    if group_cols is None:
        group_cols = ["scenario", "repeat", "Model", "Temperature", "Num Context", "Chosen Option"]

    d = df.copy()
    d = d[d["problem_type"].isin(["generic", "specific"])].copy()
    if d.empty:
        raise ValueError("No generic/specific rows found for permutation test.")

    # 1) 페어 구성 (그룹 내부에서 generic/specific를 1:1로 최대 매칭)
    pairs = []
    grouped = d.groupby(group_cols, dropna=False)
    for _, g in grouped:
        gs = g.loc[g["problem_type"] == "generic", "Rationale"].fillna("").astype(str).tolist()
        ss = g.loc[g["problem_type"] == "specific", "Rationale"].fillna("").astype(str).tolist()
        m = min(len(gs), len(ss))
        for i in range(m):
            pairs.append((gs[i], ss[i]))  # (generic_text, specific_text)

    if len(pairs) == 0:
        raise ValueError("No paired generic/specific samples were constructed.")

    # 2) 텍스트 전처리 (선택적으로 브랜드 마스킹)
    generic_texts = [g for g, _ in pairs]
    specific_texts = [s for _, s in pairs]

    if mask_brand:
        generic_texts = [mask_brand_terms(t) for t in generic_texts]
        specific_texts = [mask_brand_terms(t) for t in specific_texts]

    # 3) 벡터화 (vocab 고정)
    custom_stop = {"tesla", "Tesla", "company", "Company", "brand"}
    stop_words = list(ENGLISH_STOP_WORDS.union(custom_stop).union(NUMERIC_RELATED_STOPWORDS))
    vec = CountVectorizer(
        stop_words=stop_words,
        ngram_range=ngram_range,
        min_df=min_df,
        token_pattern=r"(?u)\b[a-zA-Z]{2,}\b",
    )
    vec.fit(generic_texts + specific_texts)
    vocab = np.array(vec.get_feature_names_out())

    # 고정 행렬(속도 개선)
    Xg = vec.transform(generic_texts)   # generic
    Xs = vec.transform(specific_texts)  # specific
    total = np.asarray((Xg + Xs).sum(axis=0)).flatten()  # alpha로도 사용
    D = (Xs - Xg).tocsr()  # pair별 차이 벡터

    def delta_from_counts(ca, cb):
        alpha = total
        alpha0 = alpha.sum()
        ca0 = ca.sum()
        cb0 = cb.sum()

        denom_a = (ca0 + alpha0) - (ca + alpha)
        denom_b = (cb0 + alpha0) - (cb + alpha)
        eps = 1e-12
        log_odds_a = np.log((ca + alpha + eps) / (denom_a + eps))
        log_odds_b = np.log((cb + alpha + eps) / (denom_b + eps))
        return log_odds_a - log_odds_b

    # 관측값
    ca_obs = np.asarray(Xs.sum(axis=0)).flatten()
    cb_obs = np.asarray(Xg.sum(axis=0)).flatten()
    obs_delta = delta_from_counts(ca_obs, cb_obs)
    obs_global = float(np.mean(np.abs(obs_delta)))

    obs_df = pd.DataFrame({"keyword": vocab, "delta_log_odds": obs_delta})
    obs_df = obs_df.sort_values("delta_log_odds", ascending=False).reset_index(drop=True)

    # 상/하위 키워드를 "정렬 후 위치"가 아니라 "실제 키워드"로 선택
    target_keywords = pd.concat([obs_df.head(top_k), obs_df.tail(top_k)], axis=0)["keyword"].unique().tolist()
    vocab_to_idx = {k: i for i, k in enumerate(vocab)}
    target_idx = np.array([vocab_to_idx[k] for k in target_keywords if k in vocab_to_idx], dtype=int)

    # permutation 루프
    rng = np.random.default_rng(random_state)
    n_pairs = len(pairs)
    extreme_counts = np.zeros(len(target_idx), dtype=int)
    perm_global_stats = np.zeros(n_perm, dtype=float)

    t0 = time.perf_counter()
    progress_every = max(1, n_perm // 20)  # 5% 단위 진행률

    for b in range(n_perm):
        # flip=True면 라벨 스왑: sign=-1, 아니면 sign=+1
        signs = np.where(rng.integers(0, 2, size=n_pairs).astype(bool), -1.0, 1.0)
        diff = np.asarray(signs @ D).flatten()  # (specific - generic) counts
        ca = (total + diff) / 2.0
        cb = total - ca

        perm_delta = delta_from_counts(ca, cb)
        perm_global_stats[b] = float(np.mean(np.abs(perm_delta)))

        # two-sided p: |perm| >= |obs|
        extreme_counts += (np.abs(perm_delta[target_idx]) >= np.abs(obs_delta[target_idx])).astype(int)

        if show_progress and ((b + 1) % progress_every == 0 or (b + 1) == n_perm):
            elapsed = time.perf_counter() - t0
            rate = (b + 1) / elapsed if elapsed > 0 else 0.0
            remain = (n_perm - (b + 1)) / rate if rate > 0 else float("inf")
            print(
                f"[Permutation] {b+1}/{n_perm} "
                f"({(b+1)/n_perm*100:.1f}%) "
                f"elapsed={elapsed:.1f}s "
                f"eta={remain:.1f}s",
                flush=True
            )

    p_vals = (extreme_counts + 1) / (n_perm + 1)
    q_vals = _bh_fdr(p_vals)

    keyword_stats = pd.DataFrame({
        "keyword": vocab[target_idx],
        "obs_delta_log_odds": obs_delta[target_idx],
        "abs_obs_delta": np.abs(obs_delta[target_idx]),
        "p_value": p_vals,
        "q_value_fdr": q_vals,
        "direction": np.where(obs_delta[target_idx] > 0, "Specific>Generic", "Generic>Specific"),
    }).sort_values(["q_value_fdr", "abs_obs_delta"], ascending=[True, False]).reset_index(drop=True)

    global_p = float((np.sum(perm_global_stats >= obs_global) + 1) / (n_perm + 1))

    return {
        "n_pairs": n_pairs,
        "n_vocab": len(vocab),
        "obs_global_mean_abs_delta": obs_global,
        "perm_global_mean_abs_delta_mean": float(np.mean(perm_global_stats)),
        "perm_global_p_value": global_p,
        "perm_global_stats": perm_global_stats,
        "keyword_stats": keyword_stats,
    }


def plot_permutation_audit_figures(
    audit_res: dict,
    perm_res: dict,
    save_dir: str = "./final_results/plots",
):
    """
    논문용 Figure 생성:
      1) Global permutation null distribution + observed statistic
      2) Context coverage comparison (raw vs masked)
    """
    os.makedirs(save_dir, exist_ok=True)

    # -------------------------------
    # Figure 1: Global permutation distribution
    # -------------------------------
    perm_stats = np.asarray(perm_res.get("perm_global_stats", []), dtype=float)
    obs_global = float(perm_res["obs_global_mean_abs_delta"])
    p_val = float(perm_res["perm_global_p_value"])

    fig, ax = plt.subplots(figsize=(8, 4.8))
    if perm_stats.size > 0:
        ax.hist(perm_stats, bins=30, color="#cfd8dc", edgecolor="#607d8b", alpha=0.9)
    ax.axvline(obs_global, color="#d32f2f", linestyle="--", linewidth=2, label=f"Observed = {obs_global:.4f}")
    ax.set_title("Permutation Null Distribution of Global Separation")
    ax.set_xlabel("mean(|delta_log_odds|)")
    ax.set_ylabel("Frequency")
    ax.legend(loc="upper left")
    ax.text(
        0.98, 0.95,
        f"p = {p_val:.4g}\nPairs = {perm_res['n_pairs']:,}\nVocab = {perm_res['n_vocab']:,}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#b0bec5")
    )
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "eval_rationale_perm_global_distribution.png"), dpi=300)
    plt.close(fig)

    # -------------------------------
    # Figure 2: Context coverage comparison
    # -------------------------------
    cov_raw = audit_res["cov_summary_raw"]
    cov_mask = audit_res["cov_summary_masked"]
    generic_raw = float(cov_raw.get("generic", np.nan))
    specific_raw = float(cov_raw.get("specific", np.nan))
    generic_mask = float(cov_mask.get("generic", np.nan))
    specific_mask = float(cov_mask.get("specific", np.nan))

    labels = ["Raw", "Brand-masked"]
    generic_vals = [generic_raw, generic_mask]
    specific_vals = [specific_raw, specific_mask]
    x = np.arange(len(labels))
    w = 0.35

    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.bar(x - w / 2, generic_vals, width=w, label="Generic", color="#1976d2", alpha=0.9)
    ax.bar(x + w / 2, specific_vals, width=w, label="Specific", color="#c62828", alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Context Coverage")
    ax.set_title("Context Coverage by Problem Type")
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "eval_rationale_context_coverage_comparison.png"), dpi=300)
    plt.close(fig)

def parse_context_short_titles(context_tags: str):
    """
    Context Tags 문자열에서 tag+short title만 리스트로 추출
    예: "[Finance] Large factory acquisition: $42M.; [Market] Fuel cost savings: ..."

    반환 예:
    [
      "[Finance] Large factory acquisition: $42M.",
      "[Market] Fuel cost savings: A new premium sedan was projected to save about $1,800 over 6 years."
    ]
    """
    if not isinstance(context_tags, str) or not context_tags.strip():
        return []

    # 세미콜론 기준 분리 후, 앞뒤 공백 제거
    parts = [p.strip() for p in context_tags.split(";") if p.strip()]

    # 각 파트는 이미 "[Tag] short title: ..." 구조인 경우가 많음
    # 마지막 '.' 제거는 하지 않음(원문 보존이 유리)
    return parts


def build_context_vocabulary(df_subset: pd.DataFrame):
    """
    df_subset에 포함된 row들의 Context Tags로부터 '컨텍스트 기반 어휘 집합' 생성
    """
    
    # 토큰에는 전체 컨텍스트 문구(예: "[Finance] Large factory acquisition: $42M.")도 포함
    tokens = []
    for s in df_subset["Context Tags"].fillna("").tolist():
        shorts = parse_context_short_titles(s)
        tokens.extend(shorts)

    # 키워드 audit 목적상 단어 레벨 vocab도 같이 만듦
    joined = " ".join(tokens).lower()
    # 아주 단순 토큰화(필요시 더 정교하게)
    vocab = set(re.findall(r"[a-z0-9]+", joined))
    return vocab, tokens


# ===============================
# (추가) 5.5 분석용: Brand masking + Context coverage
# ===============================

def mask_brand_terms(text: str, brand_terms=None) -> str:
    """
    브랜드 cue(예: tesla)를 마스킹해서 'tesla 도배' 현상을 제거한 뒤에도
    차별 키워드가 남는지 확인하기 위한 전처리.
    """
    if not isinstance(text, str):
        return ""
    if brand_terms is None:
        brand_terms = [
            r"\btesla\b",
            r"\bmodel\s*s\b",
            r"\bmodel\s*3\b",
            r"\bmodel\s*x\b",
            r"\bmodel\s*y\b",
            r"\belon\b",
            r"\bmusk\b",
            r"\bsupercharger\b",
        ]
    out = text.lower()
    for pat in brand_terms:
        out = re.sub(pat, " <BRAND> ", out, flags=re.IGNORECASE)
    out = re.sub(r"\s+", " ", out).strip()
    return out


def compute_context_coverage(df: pd.DataFrame, use_masked_rationale: bool = False) -> pd.DataFrame:
    """
    row 단위로:
    - context vocab(단어 집합)과
    - rationale vocab(단어 집합)의 overlap 비율을 계산

    coverage = |tokens(rationale) ∩ tokens(context)| / |tokens(rationale)|

    주의: stopword 제거 등은 단순화(일관성 유지 목적).
    """
    d = df.copy()

    def tokenize(s: str):
        return set(re.findall(r"[a-z0-9]+", str(s).lower()))

    coverages = []
    for _, row in d.iterrows():
        ctx_parts = parse_context_short_titles(row.get("Context Tags", ""))
        ctx_vocab = tokenize(" ".join(ctx_parts))

        rat = row.get("Rationale", "")
        if use_masked_rationale:
            rat = mask_brand_terms(rat)
        rat_vocab = tokenize(rat)

        if len(rat_vocab) == 0:
            coverages.append(np.nan)
            continue

        overlap = len(rat_vocab & ctx_vocab)
        coverages.append(overlap / len(rat_vocab))

    d["context_coverage"] = coverages
    return d


def run_tesla_generic_audit(
    df: pd.DataFrame,
    ngram_range=(2, 2),
    min_df=2,
    top_k=30,
    homogeneous_only=True,
    mask_brand=True,
):
    """
    1) homogeneous selections만 필터(optional)
    2) (optional) 브랜드 마스킹 후 log-odds
    3) context coverage 계산 및 집단 비교 요약 출력
    """
    d = df.copy()
    d = d[d["problem_type"].isin(["generic", "specific"])].copy()

    if homogeneous_only:
        d = filter_homogeneous_selections(d)

    # coverage (원문/마스킹 둘 다)
    d_cov_raw = compute_context_coverage(d, use_masked_rationale=False)
    d_cov_mask = compute_context_coverage(d, use_masked_rationale=True)

    cov_summary_raw = d_cov_raw.groupby("problem_type")["context_coverage"].mean()
    cov_summary_mask = d_cov_mask.groupby("problem_type")["context_coverage"].mean()

    # log-odds 입력 텍스트
    if mask_brand:
        texts_specific = d.loc[d["problem_type"] == "specific", "Rationale"].fillna("").astype(str).apply(mask_brand_terms).tolist()
        texts_generic  = d.loc[d["problem_type"] == "generic",  "Rationale"].fillna("").astype(str).apply(mask_brand_terms).tolist()
    else:
        texts_specific = d.loc[d["problem_type"] == "specific", "Rationale"].fillna("").astype(str).tolist()
        texts_generic  = d.loc[d["problem_type"] == "generic",  "Rationale"].fillna("").astype(str).tolist()
    top_specific, top_generic = compute_informative_log_odds(
        texts_a=texts_specific,
        texts_b=texts_generic,
        ngram_range=ngram_range,
        min_df=min_df,
        top_k=top_k
    )

    return {
        "df_used": d,
        "cov_summary_raw": cov_summary_raw,
        "cov_summary_masked": cov_summary_mask,
        "top_specific": top_specific,
        "top_generic": top_generic,
    }



# -------------------------------
# Main
# -------------------------------

#%%
if __name__ == "__main__":
    df = load_rationale_data("infer_results")

    if not df.empty:
        # 1) 설명 품질
        # summary_quality = compute_explanation_quality(df)
        # plot_explanation_quality(summary_quality)

        # 2) 전략별 키워드
        # df_keywords = compute_rationale_keywords(df, top_k=10)
        # if not df_keywords.empty:
        #     plot_rationale_keywords(df_keywords)

        # # 3) 차별적 키워드
        # discrim_keywords = compute_discriminative_keywords(df, top_k=10, ngram_range=(2,2))
        # print("\n=== Discriminative Keywords by Strategy (log-odds) ===")
        # for strat, kws in discrim_keywords.items():
        #     print(f"\n[{strat}]")
        #     print(kws)

        # # 4) 신호-이유 정합성
        # alignment_summary = compute_signal_alignment(df)
        # plot_signal_alignment(alignment_summary)

        # 5) Tesla vs Generic 차이 분석

        # (추가) 5.5 Tesla vs Generic audit
        res = run_tesla_generic_audit(
            df,
            ngram_range=(2,2),
            min_df=2,
            top_k=30,
            homogeneous_only=True, # 균일한 선택만 분석
            mask_brand=True,   # 핵심: tesla 도배 제거 후에도 차이가 남는지
        )

        print("\n=== Context Coverage (raw rationale) ===")
        print(res["cov_summary_raw"].round(4).to_string())

        print("\n=== Context Coverage (brand-masked rationale) ===")
        print(res["cov_summary_masked"].round(4).to_string())

        print("\n=== Top Discriminative Keywords: Specific > Generic (brand-masked) ===")
        print(res["top_specific"].to_string(index=False))

        print("\n=== Top Discriminative Keywords: Generic > Specific (brand-masked) ===")
        print(res["top_generic"].to_string(index=False))

        # 6) Paired permutation test (엄밀 검정)
        #    - 동일 그룹 페어를 유지한 채 generic/specific 라벨만 랜덤 스왑
        #    - 키워드별/전역(global) 차이가 우연인지 검정
        perm_res = permutation_test_discriminative_keywords(
            res["df_used"],
            ngram_range=(2, 2),
            min_df=2,
            n_perm=300,
            top_k=20,
            random_state=42,
            mask_brand=True,
            show_progress=True,
        )

        print("\n=== Permutation Test (paired, brand-masked) ===")
        print(f"n_pairs: {perm_res['n_pairs']}, n_vocab: {perm_res['n_vocab']}")
        print(
            f"global mean(|delta_log_odds|): {perm_res['obs_global_mean_abs_delta']:.6f} "
            f"(perm_mean={perm_res['perm_global_mean_abs_delta_mean']:.6f}, "
            f"p={perm_res['perm_global_p_value']:.6g})"
        )
        print("\n--- Significant keywords (q < 0.05) ---")
        sig = perm_res["keyword_stats"][perm_res["keyword_stats"]["q_value_fdr"] < 0.05]
        if sig.empty:
            print("No keywords passed q < 0.05 with current settings.")
        else:
            print(sig.to_string(index=False))

        # 7) 논문용 Figure 생성
        plot_permutation_audit_figures(
            audit_res=res,
            perm_res=perm_res,
            save_dir="./final_results/plots",
        )
        print("\nSaved figures:")
        print("- ./final_results/plots/eval_rationale_perm_global_distribution.png")
        print("- ./final_results/plots/eval_rationale_context_coverage_comparison.png")

