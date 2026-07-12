import os
os.environ["OMP_NUM_THREADS"] = "12"
os.environ["MKL_NUM_THREADS"] = "12"
os.environ["OPENBLAS_NUM_THREADS"] = "12"
os.environ["VECLIB_MAXIMUM_THREADS"] = "12"
os.environ["NUMEXPR_NUM_THREADS"] = "12"


import pandas as pd
import os
import sys
import json

# 處理路徑問題，確保能抓到 userPackage
current_dir = os.path.dirname(os.path.abspath(__file__))
parent = os.path.join(current_dir, "..")
sys.path.append(parent)

from userPackage.Package_Encode import EncodeAllFeatures

# --- 1. 路徑與設定 ---
# 讀取 Step 3 產出的 1020 (893) 維機率表
prob_data_path = "../data/mlData/prob_tables/train_prob_1020.csv"
# 這裡我們假設你也有一個對應的獨立測試集機率表，若無，可先設為 None 或指向同一個檔
indp_prob_path = "../data/mlData/prob_tables/indp_prob_1020.csv" 

mlDataPath = "../data/mlData/"
os.makedirs(mlDataPath, exist_ok=True)

print("📖 載入機率特徵表中...")
train_df = pd.read_csv(prob_data_path, index_col=0)

# 如果有獨立測試集則讀取，否則 mock 一個以符合框架介面
if os.path.exists(indp_prob_path):
    indp_df = pd.read_csv(indp_prob_path, index_col=0)
else:
    print("⚠️ 未發現獨立測試集機率表，將使用訓練集進行結構對齊")
    indp_df = train_df.copy()

# --- 2. 初始化框架對象 ---
encodeObj = EncodeAllFeatures()

# --- 3. 執行 Boruta 演算法 (提取架構核心) ---
# borutaMethod 可選 'RF', 'XGB', 'LightGBM'
print("🌿 啟動 Boruta 核心演算法 (使用 XGBoost 作為基底)...")
brtObj = encodeObj.dataBoruta(
    borutaMethod='XGB', 
    runBoruta=True, 
    featRankPath=mlDataPath,
    trainDf=train_df, 
    skipFeatureList=[]  # 這裡不需要 skip，因為全是機率特徵
)

# --- 4. 評估最佳特徵數量 ---
# 這會產出 evalFeatureNumScore.csv 供你參考在哪個維度表現最好
# print("📊 正在評估最佳特徵挑選數量 (5 到 400)...")
# encodeObj.dataEvalFeatureNum(
#     startNum=50, 
#     endNum=min(400, len(train_df.columns)-1), 
#     step=20,
#     featNumScorePath=mlDataPath, 
#     saveCsvPath=mlDataPath,
#     trainDf=train_df, 
#     indpDf=indp_df, 
#     brtObj=brtObj,
#     foldNum=10, 
#     session=100
# )
# print("Boruta complete")
# --- 5. 決定最終數量並產出 CSV ---
# 根據你的經驗或 evalFeatureNumScore.csv 的結果
# 這裡我們先預設取前 300 名，你可以跑完後根據分數回來修改此處
best_num = 270
print(f"🎯 最終決定選取前 {best_num} 個特徵並產出 Final Dataset...")

encodeObj.dataDecidedFeatureNum(
    featureNum=best_num, 
    saveCsvPath=mlDataPath,
    trainDf=train_df, 
    indpDf=indp_df,
    brtObj=brtObj
)

print(f"\n✅ Boruta 流程完成！")
print(f"💾 最終特徵表已存至: {mlDataPath}train_F{best_num}.csv")