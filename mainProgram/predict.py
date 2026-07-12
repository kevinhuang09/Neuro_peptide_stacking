import pandas as pd
import os
import sys
import warnings
from sklearn.metrics import matthews_corrcoef

# 忽略警告
warnings.filterwarnings('ignore')

# 設定路徑
current_dir = os.path.dirname(os.path.abspath(__file__))
parent = os.path.join(current_dir, "..")
sys.path.append(parent)

try:
    from MLProcess.PycaretWrapper import PycaretWrapper
except ImportError:
    print("🚨 找不到 MLProcess.PycaretWrapper")
    sys.exit(1)

# ==========================================
# 1. 路徑設定
# ==========================================
TEST_DATA_PATH = "../data/featureStat/split_features/test/"
MODEL_PATH = "../models/layer0_models/" # 假設你之前 doSaveModel 存出的位置
OUTPUT_PATH = "../data/mlData/performance/"
os.makedirs(OUTPUT_PATH, exist_ok=True)

# 17 個模型清單
MODEL_LIST = [
    'gbc', 'ridge', 'lr', 'catboost', 'lda', 'ada', 'knn', 'nb', 'et', 
    'lightgbm', 'rf', 'xgboost', 'mlp', 'dt', 'svm', 'qda', 'rbfsvm'
]

# ==========================================
# 2. 執行評估
# ==========================================
if __name__ == "__main__":
    # 獲取所有測試特徵檔案
    all_test_files = sorted([f for f in os.listdir(TEST_DATA_PATH) if f.endswith("_test.csv")])
    feature_prefixes = [f.split('_test.csv')[0] for f in all_test_files]

    print(f"🚀 開始評估測試集效能，特徵組數: {len(feature_prefixes)}")

    mcc_results = []
    
    # 建立 Pycaret 物件用於載入模型
    pycObj = PycaretWrapper()

    for f_prefix in feature_prefixes:
        print(f"🔍 正在評估特徵: {f_prefix}")
        test_file = os.path.join(TEST_DATA_PATH, f"{f_prefix.lower()}_test.csv")
        test_df = pd.read_csv(test_file, index_col=0)
        
        # 分離標籤與數據
        y_true = test_df['y']
        X_test = test_df.drop(columns=['y'])

        # ⚠️ 注意：這裡需要載入你在訓練階段存下的模型
        # 假設你的路徑結構是 model_path/feature_name/model_name_tuned.pkl
        load_dir = os.path.join(MODEL_PATH, f_prefix)
        
        try:
            # 載入該特徵組的所有模型
            loaded_models = pycObj.doLoadModel(path=load_dir, fileNameList=MODEL_LIST, b_isFinalizedModel=False)
            
            for model, model_name in zip(loaded_models, MODEL_LIST):
                # 使用 Wrapper 內的預測函式
                pred_df = pycObj.predictModelCustom(model, data=X_test)
                
                # 計算 MCC
                mcc = matthews_corrcoef(y_true, pred_df['Label'])
                
                mcc_results.append({
                    'Feature_Type': f_prefix,
                    'Model_Name': model_name,
                    'Test_MCC': mcc
                })
                
        except Exception as e:
            print(f"⚠️ 特徵 {f_prefix} 模型載入或預測失敗: {e}")

    # ==========================================
    # 3. 產出表格
    # ==========================================
    if mcc_results:
        mcc_df = pd.DataFrame(mcc_results)
        # 按 MCC 由高到低排序
        mcc_df = mcc_df.sort_values(by='Test_MCC', ascending=False)
        
        # 儲存 CSV
        output_file = os.path.join(OUTPUT_PATH, "layer0_test_mcc_report.csv")
        mcc_df.to_csv(output_file, index=False)
        
        print("\n" + "="*50)
        print(f"📊 評估完成！效能報表已儲存至: {output_file}")
        print(f"前五名表現最佳組合：")
        print(mcc_df.head(5))
        print("="*50)
    else:
        print("❌ 未能產生任何 MCC 結果。")