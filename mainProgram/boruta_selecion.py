import os
os.environ["OMP_NUM_THREADS"] = "12"
os.environ["MKL_NUM_THREADS"] = "12"
os.environ["OPENBLAS_NUM_THREADS"] = "12"
os.environ["VECLIB_MAXIMUM_THREADS"] = "12"
os.environ["NUMEXPR_NUM_THREADS"] = "12"

import pandas as pd
import os
import sys
import glob
import json

current_dir = os.path.dirname(os.path.abspath(__file__))
parent = os.path.join(current_dir, "..")
sys.path.append(parent)

sys.path.append(current_dir)
import config

from userPackage.Package_Encode import EncodeAllFeatures

dataName   = config.DATA_NAME
process1_t = config.process1_t
process2_t = config.process2_t
PROB_PATH  = f"../data/mlData/split_features_{dataName}_{process1_t}_{process2_t}/"

mlDataPath = "../data/mlData/"
os.makedirs(mlDataPath, exist_ok=True)

def find_latest(pattern):
    cands = glob.glob(os.path.join(PROB_PATH, pattern))
    if not cands:
        raise FileNotFoundError(f"在 {PROB_PATH} 找不到符合 {pattern} 的檔案，"
                                f"請先跑 meta_feature_gent.py")
    return max(cands, key=os.path.getmtime)   # 取最新產生的

prob_data_path = find_latest("train_prob_*.csv")
indp_prob_path = find_latest("test_prob_*.csv")   # 用 test_prob 當結構對齊的獨立集

print(f"📖 載入機率特徵表中...")
print(f"   train : {prob_data_path}")
print(f"   test  : {indp_prob_path}")

train_df = pd.read_csv(prob_data_path, index_col=0)

# 如果有獨立測試集則讀取，否則 mock 一個以符合框架介面
if os.path.exists(indp_prob_path):
    indp_df = pd.read_csv(indp_prob_path, index_col=0)
else:
    print("未發現獨立測試集機率表，將使用訓練集進行結構對齊")
    indp_df = train_df.copy()

train_feats = [c for c in train_df.columns if c != 'y']
test_feats  = [c for c in indp_df.columns  if c != 'y']
missing = set(train_feats) - set(test_feats)
if missing:
    print(f"警告：test 缺少 {len(missing)} 個 train 有的特徵欄，例如 {list(missing)[:5]}")

print(f"   train 維度: {train_df.shape}, test 維度: {indp_df.shape}")
print(f"   共 {len(train_feats)} 個機率特徵 (feature_type × model)")

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
    skipFeatureList=[]   # 全是機率特徵，不需要 skip
)

# --- 4. 評估最佳特徵數量 
eval_end = min(300, len(train_feats))
encodeObj.dataEvalFeatureNum(
    startNum=50,
    endNum=eval_end,
    step=20,
    featNumScorePath=mlDataPath,
    saveCsvPath=mlDataPath,
    trainDf=train_df,
    indpDf=indp_df,
    brtObj=brtObj,
    foldNum=10,
    session=100
)
print("評估完成！請查看 evalFeatureNumScore.csv 決定最佳 featureNum")
print(f"   （檔案位置：{mlDataPath}evalFeatureNumScore.csv）")

# --- 5. 決定最終數量並產出 CSV 
# 看完 evalFeatureNumScore.csv 後，把 best_num 改成分數最高的維度，再解除下面註解跑一次
#
# best_num = 270   # ← 改成你從 evalFeatureNumScore.csv 選出的最佳數量
# print(f"最終決定選取前 {best_num} 個特徵並產出 Final Dataset...")
# encodeObj.dataDecidedFeatureNum(
#     featureNum=best_num,
#     saveCsvPath=mlDataPath,
#     trainDf=train_df,
#     indpDf=indp_df,
#     brtObj=brtObj
# )
# print(f"\n Boruta 流程完成！")
# print(f"最終特徵表已存至: {mlDataPath}train_F{best_num}.csv")
