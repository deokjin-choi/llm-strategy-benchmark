import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_strategy_ratio_heatmap(csv_path="./combined_analysis/analysis_overall_ratio.csv"):
    # CSV 불러오기
    df = pd.read_csv(csv_path)
    
    # scenario_type을 index로 설정
    df = df.set_index("scenario_type")
    
    # 히트맵 그리기
    plt.figure(figsize=(10,6))
    sns.heatmap(
        df,
        annot=True, fmt=".2f",
        cmap="YlGnBu",
        cbar_kws={'label': 'Strategy Ratio'}
    )
    title = "Strategy Ratio by Scenario"
    plt.title(title, fontsize=16)
    plt.ylabel("Scenario Type")
    plt.xlabel("Strategy")
    plt.tight_layout()
    
    # 저장 경로 지정
    save_dir = "./combined_analysis/plots"
    os.makedirs(save_dir, exist_ok=True)
    # 제목을 파일명으로 변환 (공백 등 안전하게 처리)
    safe_title = title.replace(" ", "_")
    save_path = os.path.join(save_dir, f"eval_{safe_title}.png")
    
    # 저장 및 출력
    plt.savefig(save_path, dpi=300)
    plt.show()

if __name__ == "__main__":
    csv_path = "./combined_analysis/analysis_overall_ratio.csv"
    plot_strategy_ratio_heatmap(csv_path)
