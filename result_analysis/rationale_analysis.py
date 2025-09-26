import os, glob, re
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
        df_keywords = compute_rationale_keywords(df, top_k=10)
        if not df_keywords.empty:
            plot_rationale_keywords(df_keywords)

        # 3) 차별적 키워드
        discrim_keywords = compute_discriminative_keywords(df, top_k=10, ngram_range=(2,2))
        print("\n=== Discriminative Keywords by Strategy (log-odds) ===")
        for strat, kws in discrim_keywords.items():
            print(f"\n[{strat}]")
            print(kws)

        # 4) 신호-이유 정합성
        alignment_summary = compute_signal_alignment(df)
        plot_signal_alignment(alignment_summary)

