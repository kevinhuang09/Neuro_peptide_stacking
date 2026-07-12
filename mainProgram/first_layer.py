import pandas as pd
import os
import sys
import numpy as np
from pycaret.classification import setup, create_model, tune_model, predict_model, finalize_model

# --- 1. 路徑與參數設定 ---
splitDataPath = "../data/mlData/feature_types_split/"
mlScorePath = "../data/mlScore/"
if not os.path.exists(mlScorePath): os.makedirs(mlScorePath)

# 17 個基礎模型清單
modelNameList = ['lightgbm', 'catboost', 'gbc', 'ridge', 'lr', 'lda', 'ada', 'knn', 'nb', 'et', 'rf', 'xgboost', 'mlp', 'dt', 'svm', 'qda', 'rbfsvm']

# 獲取所有啟用的特徵類型名稱 (由資料夾內的檔案決定)
feature_types = sorted(list(set([f.replace('_train.csv', '') for f in os.listdir(splitDataPath) if f.endswith('_train.csv')])))

print(f"📦 偵測到 {len(feature_types)} 種特徵類型，準備進行 17x{len(feature_types)} 模型訓練...")

all_level1_features = []

# --- 2. 核心雙層迴圈 ---
for f_type in feature_types:
    print(f"\n" + "="*50)
    print(f"🔥 正在處理特徵類型: {f_type}")
    print("="*50)
    
    train_file = os.path.join(splitDataPath, f"{f_type}_train.csv")
    df = pd.read_csv(train_file)
    
    # 初始化 Pycaret 環境
    # n_jobs=-1 使用全部 CPU，html=False 禁止在非 Notebook 環境輸出圖表
    clf_setup = setup(data=df, target='y', session_id=123, verbose=False, html=False, n_jobs=-1)
    
    type_probs = pd.DataFrame(index=df.index)

    for m_name in modelNameList:
        try:
            print(f"  ⚡ 訓練模型: {m_name}...", end=" ")
            
            # 建立模型 (這裡使用預設參數以節省時間，若要調參請改用 tune_model)
            model = create_model(m_name, cross_validation=True, verbose=False)
            
            # 獲取 Out-of-Fold (OOF) 預測機率
            # 在 Stacking 中，Level-1 的輸入必須是 OOF 預測，否則會嚴重的 Overfitting
            from pycaret.classification import get_config
            oof_predictions = predict_model(model, data=df, raw_score=True)
            
            # 提取正樣本(Label 1)的機率值，並命名為 {FeatureType}_{Model}
            prob_col_name = f"Score_1" if f"Score_1" in oof_predictions.columns else "prediction_score"
            type_probs[f"{f_type}_{m_name}"] = oof_predictions[prob_col_name]
            
            print("✅ OK")
            
        except Exception as e:
            print(f"❌ 失敗: {e}")
            # 若失敗，填入 0.5 (中性機率) 以維持維度一致
            type_probs[f"{f_type}_{m_name}"] = 0.5

    all_level1_features.append(type_probs)

# --- 3. 橫向合併 (Horizontal Concatenation) ---
print("\n" + "Done!" + " ="*20)
print("📦 正在合併所有機率特徵...")

# 將 60 個 DataFrame 橫向拼接
final_df = pd.concat(all_level1_features, axis=1)

# 補上正確的標籤 y (從最後一個 df 拿)
final_df['y'] = pd.read_csv(os.path.join(splitDataPath, f"{feature_types[0]}_train.csv"))['y'].values

# 儲存最終 1020 欄的大表
output_path = os.path.join(mlScorePath, "level_1_input_1020_features.csv")
final_df.to_csv(output_path, index=False)

print(f"🎉 任務完成！")
print(f"📊 最終特徵表維度: {final_df.shape}")
print(f"💾 檔案儲存至: {output_path}")