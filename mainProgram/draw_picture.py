import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor

# ==========================================
# 1. 繪圖任務函數
# ==========================================

def plot_boxplot(df, output_path):
    """繪製四分位分布圖"""
    plt.figure(figsize=(12, 6))
    sns.set(style="whitegrid")
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    sns.boxplot(x='Model', y='Test_MCC', data=df, palette='viridis')
    sns.stripplot(x='Model', y='Test_MCC', data=df, color=".3", size=3, alpha=0.5)
    
    plt.title('不同模型的效能分布 (56 種特徵組合)', fontsize=15)
    plt.xlabel('模型代號', fontsize=12)
    plt.ylabel('Test MCC', fontsize=12)
    plt.xticks(rotation=45)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    return f"✅ 四分位圖已存至: {output_path}"

def plot_distribution_binned(df, output_path):
    """繪製特定區間 (0.1 bin) 的 MCC 統計分布圖"""
    plt.figure(figsize=(10, 6))
    sns.set(style="whitegrid")
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # ✨ 關鍵修改：自定義區間邊界 [0.0, 0.1, 0.2, ..., 1.0]
    custom_bins = np.arange(0, 1.1, 0.1) 
    
    # 繪製直方圖
    sns.histplot(df['Test_MCC'], bins=custom_bins, kde=False, color='skyblue', edgecolor='black')
    
    # 在長條圖上方顯示具體數字
    counts, bin_edges = np.histogram(df['Test_MCC'], bins=custom_bins)
    for count, x in zip(counts, bin_edges):
        if count > 0:
            plt.text(x + 0.05, count + 0.5, str(count), ha='center', va='bottom', fontsize=10)

    plt.title('所有模型組合的 MCC 分布統計 (區間 0.1)', fontsize=15)
    plt.xlabel('Test MCC 區間', fontsize=12)
    plt.ylabel('模型數量 (Number of Models)', fontsize=12)
    plt.xticks(custom_bins) # 強制顯示 0.1, 0.2... 的座標
    
    # 標註平均值
    mean_val = df['Test_MCC'].mean()
    plt.axvline(mean_val, color='red', linestyle='--', label=f'平均值: {mean_val:.3f}')
    plt.legend()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    return f"✅ 區間統計分布圖已存至: {output_path}"

# ==========================================
# 2. 主流程
# ==========================================

if __name__ == "__main__":
    file_path = "../data/mlData/prob_tables/model_performance_mcc.csv"
    output_dir = "../data/mlData/plots/"
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(file_path):
        print(f"❌ 找不到資料檔: {file_path}")
    else:
        df_mcc = pd.read_csv(file_path)
        
        # 準備並行任務
        tasks = [
            (plot_boxplot, df_mcc, os.path.join(output_dir, "model_boxplot.png")),
            (plot_distribution_binned, df_mcc, os.path.join(output_dir, "mcc_distribution_binned.png"))
        ]
        
        print(f"🚀 啟動並行繪圖處理 (Bin Size = 0.1)...")
        
        with ProcessPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(func, data, path) for func, data, path in tasks]
            for future in futures:
                print(future.result())

        print("\n" + "="*40)
        print(f"🎉 任務完成！請查看 plots 資料夾。")
        print("="*40)