import matplotlib
matplotlib.use("Agg")  # 禁止跳出繪圖視窗
# 處理路徑問題
import sys, os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent = os.path.join(current_dir, "..")
sys.path.append(parent)

import pandas as pd
import os
from MLProcess.PycaretWrapper import PycaretWrapper
from MLProcess.Stacking import Stacking
from MLProcess.change_binary import changeBinaryFeatureInDf

# --- 1. 設定路徑與參數 ---
featNumber = 250  # 這是你剛篩選完的特徵數
mlDataPath = "../data/mlData/"
tuneModelPath = "../data/tuneModel/"
mlScorePath = "../data/mlScore/"
if not os.path.exists(mlScorePath): os.makedirs(mlScorePath)

# 你的 17 個基礎模型清單
modelNameList = ['lightgbm', 'catboost', 'rbfsvm', 'gbc', 'ridge', 'lr', 'lda', 'ada', 'knn', 'nb', 'et', 'rf', 'xgboost', 'mlp', 'dt', 'svm', 'qda']
# --- 1.5 核心訓練步驟：模型與 Feature 進行訓練 ---
print("正在讀取數據並準備訓練...")
dataTrainDf = pd.read_csv(os.path.join(mlDataPath, f"train_F{featNumber}.csv"))

# if want to use model again can annotate
# dataTrainDf = changeBinaryFeatureInDf(dataTrainDf)

# 初始化 Pycaret 訓練環境
# pycObj = PycaretWrapper()
# pycObj.doSetup(trainData=dataTrainDf)

# print(f"開始針對 {len(modelNameList)} 個模型進行訓練與調參 (n_iter=100)...")
# # 這一步就是「模型跟 Feature 做訓練」，會產生 .pkl 檔案
# pycObj.doTuneModel(searchLibrary='optuna', includeModelList=modelNameList, n_iter=100)

# store model 
# pycObj.doSaveModel(tuneModelPath, b_isFinalizedModel=False)
print("all models already store")



# --- 2. 初始化 Stacking 物件 ---
# 這裡會讀取你的精華特徵集 train_F250.csv
stackObj = Stacking(
    loadModelPath=tuneModelPath,
    modelNameList=modelNameList,
    trainDataPath=os.path.join(mlDataPath, f"train_F{featNumber}.csv"),
    testDataPath=os.path.join(mlDataPath, f"indp_F{featNumber}.csv"),
    clusterNumList=[5]  # 這裡的數字配合流程圖中的分群邏輯
)

# --- 3. 執行 10-Fold OOF 預測 (產生新特徵的核心) ---
print("🚀 開始執行預測並產生評分與特徵...")

# 這個函式會回傳一個 DataFrame，每一欄代表一個模型的預測機率 (Probabilities)
# 這些機率值就是你流程圖中的 "Output Features"
scoreDf = stackObj.genSelfTestResult(drawPlot=False)

# --- 4. 儲存這 550 個特徵 (或目前產出的模型特徵) ---
# part one : score file
score_file = os.path.join(mlDataPath, "best_mcc.csv")
scoreDf.to_csv(score_file)
print(f"The score file store in {score_file}")

# part two : feature file
prob_features_df = pd.DataFrame(stackObj.selfTestProbVectorList).T 

# 重新命名欄位為模型名稱，這就是 Level-1 的輸入特徵
prob_features_df.columns = modelNameList

# 儲存特徵矩陣
feature_file = os.path.join(mlScorePath, 'level_1_input_features.csv')
prob_features_df.to_csv(feature_file)
df = pd.read_csv('../data/mlScore/level_1_input_features.csv', index_col=0)
print(f"🧬 Level-1 特徵矩陣已儲存至: {feature_file}")
rows = prob_features_df.shape[0] 
cols = prob_features_df.shape[1]

print("-" * 30)
print(f"📊 特徵表維度檢查：")
print(f"🔹 Row (樣本數): {rows}")
print(f"🔹 Col (特徵數): {cols}")
print("-" * 30)