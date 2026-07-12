import matplotlib
matplotlib.use('Agg')  # 禁止彈出圖表視窗

# 處理路徑問題
import sys, os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent = os.path.join(current_dir, "..")
sys.path.append(parent)

import json
import pandas as pd

from userPackage.Package_Encode import EncodeAllFeatures
from userPackage.LoadDataset import LoadDataset
from userPackage.FeatureStat import FeatureStat
from MLProcess.Stacking import Stacking

#  移到外面再把舊的加進空的dict裡面
ifeatureDict = {"AAC": True,
                "AAINDEX": False,  # 注意!!需等長
                "CKSAAGP": [True, 5],
                "CTDC": True,
                "CTDD": True,
                "CTDT": True,
                "CTriad": True,  # 會產生feature 343個
                "DDE": True,
                "DPC": True,
                "GAAC": True,
                "GDPC": True,
                "GTPC": True,
                "KSCTriad": [True, 0],  # 會產生feature 343個
                "QSOrder": [False, 30, 0.1],
                "TPC": False,  # 會有8000個feature 不建議開啟
                "SOCN": [False, 3],
                "APAAC": [True, 3, 0.05],
                "Geary": [True, 3],
                "Moran": [True, 3],
                "NMBroto": [True, 3],
                "CKSAAP": [True, 3],  # 會產生feature 1600個
                "BINARY": False,  # 注意!!需等長
                "PAAC": [True, 3, 0.05]
                }

PfeatureDict = {"DDOR": True,
                "RRI": True,
                "SER": True,
                "SEP": True,
                "SE": True,
                "QSO": [True, 3, 0.1]
                }  # gap=3 w=0.1

AMPfeatureDict = {"length": True,
                  "calculate_mw": [True, True],  # [0] = on(True)/off [1] = amide
                  "calculate_charge": [True, 7, True],  # [0] = on(True)/off [1] = ph, [2] = amide
                  "charge_density": [True, 7, True],  # [0] = on(True)/off [1] = ph, [2] = amide
                  "isoelectric_point": [True, True],  # [0] = on(True)/off [1] = amide
                  "instability_index": True,
                  "aromaticity": True,
                  "aliphatic_index": True,
                  "hydrophobic": True,
                  "aasi": True,
                  "abhprk": [True, 5],  # [0] = on(True)/off [1] = window
                  "argos": True,
                  "bulkiness": True,
                  "charge_phys": True,
                  "charge_acid": True,
                  "cougar": [True, 5],  # [0] = on(True)/off [1] = window
                  "ez": [True, 5],  # [0] = on(True)/off [1] = window
                  "flexibility": True,
                  "gravy": True,
                  "levitt_alpha": True,
                  "mss": True,
                  "msw": [True, 5],  # [0] = on(True)/off [1] = window
                  "pepArc": True,
                  "polarity": True,
                  "refractivity": True,
                  "tm_tend": True,
                  "z3": [True, 5],  # [0] = on(True)/off [1] = window
                  "z5": [True, 5],  # [0] = on(True)/off [1] = window
                  "formula": True,  # C,H,N,O,S atom composition
                  "boman_index": True,
                  "eisenberg": True,
                  "hopp_woods": True,
                  "janin": True,
                  "kytedoolittle": True
                  }

OVPfeatureDict = {"OVPC": True,
                  "OVP": [True, 4, 4]  # 兩個數為 N, C 端胺基酸數目  N= C=
                  }

MotifBitVecfeatureDict = {"Usage": False,
                          "motifList": ['FKK', 'LKL', 'KKLL', 'KWK', 'VLK',
                                        'CY'
                                        ''
                                        'CR', 'CRR', 'RFC', 'RRR', 'LKKL']
                          }

centerGDPDict = {"Usage": False, "UseGap": False, "gap_size": -1}  # 若是預測每個 amino acid 的 label 在使用

featureDict = {'iFeature': ifeatureDict,
               'pFeature': PfeatureDict,
               'ampFeature': AMPfeatureDict,
               'ovpFeature': OVPfeatureDict,
               'motifBitVecFeature': MotifBitVecfeatureDict,
               'centerGDPFeature': centerGDPDict}

# 初始化存放 True 特徵的清單
enabled_features = []

for category, sub_dict in featureDict.items():
    for feat_name, value in sub_dict.items():
        # 情況 1: 直接是布林值 (如 "AAC": True)
        if isinstance(value, bool):
            if value is True:
                enabled_features.append(feat_name)
        
        # 情況 2: 是列表 (如 "CKSAAGP": [True, 5])
        elif isinstance(value, list):
            # 檢查列表第一個元素是否為 True
            if len(value) > 0 and value[0] is True:
                enabled_features.append(feat_name)

# 印出結果
print(f"✅ 共有 {len(enabled_features)} 個啟用的特徵：")
print(enabled_features)

mlDataPath = "../data/mlData/feature_types_split/"
tuneModelPath = "../data/tuneModel/"
mlScorePath = "../data/mlScore/"
modelNameList = ['lightgbm', 'catboost', 'rbfsvm', 'gbc', 'ridge', 'lr', 'lda', 'ada', 'knn', 'nb', 'et', 'rf', 'xgboost', 'mlp', 'dt', 'svm', 'qda']

all_feature_dfs = []

def run_single_feature_stacking(f_name, mlDataPath, tuneModelPath, modelNameList):
    """
    這是一個獨立的工作單元，負責處理一個特徵類型。
    """
    train_csv = os.path.join(mlDataPath, f"{f_name}_train.csv")
    indp_csv = os.path.join(mlDataPath, f"{f_name}_indp.csv")
    
    if not os.path.exists(train_csv):
        return None  # 檔案不存在回傳 None

    # 初始化 Stacking 物件
    stackObj = Stacking(
        loadModelPath=tuneModelPath,
        modelNameList=modelNameList,
        trainDataPath=train_csv,
        testDataPath=indp_csv,
        clusterNumList=[5]
    )
    
    # 執行預測
    stackObj.genSelfTestResult(drawPlot=False)
    
    # 轉換成 DataFrame 並命名
    prob_df = pd.DataFrame(stackObj.selfTestProbVectorList).T
    prob_df.columns = [f"{f_name}_{m}" for m in modelNameList]
    
    return prob_df

from concurrent.futures import ProcessPoolExecutor, as_completed

if __name__ == "__main__":  # <--- Windows 執行必加這行
    
    # --- 參數設定 ---
    # 建議設定為 CPU 核心數的一半，例如 8 核心設 4，避免記憶體不足
    MAX_WORKERS = 8
    
    all_feature_dfs = []

    print(f"🔥 開始多核心並行處理，最大執行緒數: {MAX_WORKERS}")

    # 使用 ProcessPoolExecutor 啟動並行引擎
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        
        # 1. 提交所有任務到進程池 (Submit Tasks)
        future_to_feature = {
            executor.submit(run_single_feature_stacking, f, mlDataPath, tuneModelPath, modelNameList): f 
            for f in enabled_features
        }

        # 2. 依照完成順序收集結果 (Collect Results)
        for future in as_completed(future_to_feature):
            f_name = future_to_feature[future]
            try:
                result_df = future.result()
                if result_df is not None:
                    all_feature_dfs.append(result_df)
                    print(f"✅ {f_name} 處理完成 (目前進度: {len(all_feature_dfs)}/{len(enabled_features)})")
            except Exception as e:
                print(f"❌ {f_name} 發生錯誤: {e}")

    # --- 第三步：合併資料 ---
    if all_feature_dfs:
        print("📦 正在橫向合併所有特徵...")
        final_stacked_features = pd.concat(all_feature_dfs, axis=1)
        output_file = os.path.join(mlScorePath, 'level_1_all_1020_features.csv')
        final_stacked_features.to_csv(output_file)
        print(f"🎉 任務完成！維度為: {final_stacked_features.shape}")