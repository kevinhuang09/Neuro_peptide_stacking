import pandas as pd
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor

# 忽略警告資訊
warnings.filterwarnings('ignore')

# 設定路徑
current_dir = os.path.dirname(os.path.abspath(__file__))
parent = os.path.join(current_dir, "..")
sys.path.append(parent)

try:
    from MLProcess.PycaretWrapper import PycaretWrapper
except ImportError:
    print("🚨 找不到 MLProcess.PycaretWrapper，請確保路徑正確。")
    sys.exit(1)

# ==========================================
# 1. 參數與路徑設定
# ==========================================
# 這裡僅讀取 train 資料夾
SPLIT_DATA_PATH = "../data/featureStat/split_features/train/"
PROB_OUTPUT_PATH = "../data/mlData/prob_tables/"
os.makedirs(PROB_OUTPUT_PATH, exist_ok=True)

EXPECTED_TRAIN_SAMPLES = 7875

# 完整的 17 個模型清單
MODEL_LIST = [
    'gbc', 'ridge', 'lr', 'catboost', 'lda', 'ada', 'knn', 'nb', 'et', 
    'lightgbm', 'rf', 'xgboost', 'mlp', 'dt', 'svm', 'qda', 'rbfsvm'
]

# ==========================================
# 2. Worker 函數：僅生成訓練集 OOF 機率
# ==========================================
def process_feature_group(f_prefix):
    """
    僅處理訓練集，生成 OOF 機率表。
    """
    train_file = os.path.join(SPLIT_DATA_PATH, f"{f_prefix.lower()}_train.csv")
    
    if not os.path.exists(train_file):
        print(f"⚠️ 檔案缺失: {train_file}")
        return f_prefix, None

    # 讀取原始數據 (保留原始順序)
    train_df = pd.read_csv(train_file, index_col=0)
    
    pycObj = PycaretWrapper()
    # 進行 Setup (僅針對訓練集)
    pycObj.doSetup(train_df, sessionID=100) 
    
    # 建立存放機率的 DataFrame，Index 繼承自原始檔案
    temp_train_probs = pd.DataFrame(index=train_df.index)

    print(f"⚙️ 正在生成 OOF Meta-Features: {f_prefix} ...")
    
    for model_name in MODEL_LIST:
        try:
            # 建立模型
            model = pycObj.createModelCustom(model_name) 
            
            # 獲取訓練集 OOF 預測值 (Score 欄位即為機率)
            train_pred = pycObj.getOOFPredictions(model) 
            
            if 'Score' in train_pred.columns:
                col_name = f"{f_prefix}_{model_name}"
                
                # 根據 Index 填回，確保資料對齊
                temp_train_probs[col_name] = train_pred['Score']

        except Exception as e:
            print(f"⚠️ 模型 {model_name} 在特徵 {f_prefix} 上失敗: {e}")
            
    return f_prefix, temp_train_probs

# ==========================================
# 3. 主流程
# ==========================================
if __name__ == "__main__":
    # 獲取所有訓練特徵檔案
    all_files = sorted([f for f in os.listdir(SPLIT_DATA_PATH) if f.endswith("_train.csv")])
    feature_prefixes = [f.split('_train.csv')[0] for f in all_files]

    print(f"🚀 啟動並行運算，特徵類別總數: {len(feature_prefixes)}")

    all_train_prob_dfs = []

    # 使用並行處理
    with ProcessPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(process_feature_group, pref) for pref in feature_prefixes]
        
        for future in futures:
            try:
                pref, tr_prob = future.result()
                if tr_prob is not None:
                    all_train_prob_dfs.append(tr_prob)
            except Exception as e:
                print(f"🚨 子進程崩潰: {e}")

    # ==========================================
    # 4. 嚴格順序校驗與合併
    # ==========================================
    if all_train_prob_dfs:
        print("\n--- 開始數據合併與順序校驗 (無 MCC 模式) ---")
        
        # 以第一個特徵組為合併基準 (嚴格不使用 Sort)
        final_train_df = all_train_prob_dfs[0]

        # 逐一檢查後續特徵組的 Index 順序
        for i in range(1, len(all_train_prob_dfs)):
            current_train = all_train_prob_dfs[i]
            
            # 檢查兩者 Index 是否完全一致（順序與內容）
            if not final_train_df.index.equals(current_train.index):
                print(f"🚨 錯誤：'{feature_prefixes[i]}' 的順序與基準不符！")
                sys.exit("停止執行：特徵檔案 ID 順序不統一，請檢查原始資料。")
            
            # 橫向合併欄位
            final_train_df = pd.concat([final_train_df, current_train], axis=1)

        # 讀取 y 標籤
        y_label_train = pd.read_csv(os.path.join(SPLIT_DATA_PATH, all_files[0]), index_col=0)[['y']]
        
        # 最終確認標籤順序
        if not final_train_df.index.equals(y_label_train.index):
            sys.exit("🚨 錯誤：機率表與原始 y 標籤順序不對應！")

        # 加入 y 標籤並檢查筆數
        final_train_df['y'] = y_label_train['y']
        actual_count = len(final_train_df)
        
        if actual_count != EXPECTED_TRAIN_SAMPLES:
            print(f"⚠️ 警告：樣本數 {actual_count} 與預期 {EXPECTED_TRAIN_SAMPLES} 不符。")

        # 儲存結果
        output_file = os.path.join(PROB_OUTPUT_PATH, "train_prob_1020.csv")
        final_train_df.to_csv(output_file)

        print("="*50)
        print(f"🎉 成功！訓練集機率表已儲存至: {output_file}")
        print(f"最終維度: {final_train_df.shape} (特徵數: {final_train_df.shape[1]-1})")
        print("="*50)
    else:
        print("❌ 錯誤：未能生成任何有效的數據。")