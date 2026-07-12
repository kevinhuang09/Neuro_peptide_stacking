import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import matthews_corrcoef
import os

def main(csv_path):
    print(f"📥 正在讀取資料: {csv_path} ...")
    try:
        df = pd.read_csv(csv_path, index_col=0)
    except Exception as e:
        print(f"🚨 讀取 CSV 失敗: {e}")
        return
    
    if 'y' not in df.columns:
        print("🚨 資料中找不到目標欄位 'y'！請確認 CSV 中有 y 標籤。")
        return
    
    y_true = df['y']
    model_cols = [col for col in df.columns if col != 'y']
    
    print(f"🧮 正在計算 {len(model_cols)} 個模型的 MCC 分數...")
    records = []
    
    # 1. 計算每個模型的 MCC 並提取 Feature_Type
    for col in model_cols:
        temp_df = pd.DataFrame({'y_true': y_true, 'y_prob': df[col]}).dropna()
        if temp_df.empty:
            continue

        y_pred = (temp_df['y_prob'] >= 0.5).astype(int)
        mcc = matthews_corrcoef(temp_df['y_true'], y_pred)
        
        # 從欄位名稱 (例如 aac_gbc) 提取 feature_type (aac)
        feature_type = col.split('_')[0]
        records.append({'Feature_Type': feature_type, 'MCC': mcc})
        
    mcc_df = pd.DataFrame(records)
    
    if mcc_df.empty:
        print("🚨 嚴重警告：計算後沒有任何有效的 MCC 數據可供繪圖！請檢查真實標籤 y 是否為空。")
        return
        
    print("📊 正在繪製 Feature Type vs MCC 箱型圖...")
    
    # 2. 繪圖樣式設定
    sns.set_style("whitegrid")
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'PingFang TC', 'Arial Unicode MS'] 
    plt.rcParams['axes.unicode_minus'] = False
    
    # 依照 MCC 中位數由高到低排序，讓圖表更易讀
    median_order = mcc_df.groupby('Feature_Type')['MCC'].median().sort_values(ascending=False).index
    
    # 根據特徵數量自動調整圖片高度，避免擠在一起
    fig_height = max(6, len(median_order) * 0.4)
    plt.figure(figsize=(12, fig_height), dpi=150)
    
    # 3. 繪製箱型圖 (完全還原你截圖中的風格)
    sns.boxplot(
        data=mcc_df, 
        y='Feature_Type', 
        x='MCC', 
        order=median_order,
        palette='Set3',     # 柔和的配色風格
        linewidth=1.2,
        showmeans=True,     # 顯示平均值
        # 平均值標示為白色三角形 (對應你截圖的 ^)
        meanprops={"marker":"^", "markerfacecolor":"white", "markeredgecolor":"black", "markersize": 7},
        # 離群值標示為深灰色菱形 (對應你截圖的 d)
        flierprops={"marker":"d", "markerfacecolor":"dimgray", "markeredgecolor":"dimgray", "markersize": 5}
    )
    
    # 4. 標題與標籤
    plt.title('各 Feature Type 的 MCC 分數表現', fontsize=16, pad=15)
    plt.xlabel('MCC 分數', fontsize=13)
    plt.ylabel('特徵類型 (Feature Type)', fontsize=13)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    
    # 5. 確保資料夾存在並儲存
    output_dir = "../data/mlPicture_NeuroPeptide"
    os.makedirs(output_dir, exist_ok=True)
    out_path = f"{output_dir}/feature_mcc_boxplot.png"
    
    # bbox_inches='tight' 可確保最左邊的標籤不會被圖片切掉
    plt.savefig(out_path, bbox_inches='tight')
    print(f"✅ 圖片已成功儲存至: {out_path}")
    plt.close()

if __name__ == "__main__":
    TARGET_CSV_PATH = "../data/mlData/prob_tables_features_NeuroPeptide/test_prob_(220, 953).csv" 
    if os.path.exists(TARGET_CSV_PATH):
        main(TARGET_CSV_PATH)
    else:
        print(f"🚨 找不到 CSV 檔案！請確認 {TARGET_CSV_PATH} 路徑是否正確。")