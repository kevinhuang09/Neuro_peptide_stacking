import matplotlib
matplotlib.use('Agg')  # 禁止彈出圖表視窗

# 處理路徑問題
import sys, os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent = os.path.join(current_dir, "..")
sys.path.append(parent)

import json

from userPackage.Package_Encode import EncodeAllFeatures
from userPackage.LoadDataset import LoadDataset
from userPackage.FeatureStat import FeatureStat

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

mlDataPath = "../data/mlData/"  # 內含 data 檔案 ex : train_F390.csv, boruta 檔案 ex :Boruta-featRank-RF.csv
paramPath = "../data/param/"  # 內含檔案: featureTypeDict.pkl, normalize.pkl
normalizeMethod = 'standard'
dataName = 'HemoPi_1'
print("path is OK")
ldObj = LoadDataset(minSeqLength=5)
# trainNegSeqDict = ldObj.readFasta("../data/HemoPI_1_neg_main80_.fasta")
# trainPosSeqDict = ldObj.readFasta("../data/HemoPI_1_pos_main80_.fasta")
# indpNegSeqDict = ldObj.readFasta("../data/HemoPI_1_neg_val20_.fasta")
# indpPosSeqDict = ldObj.readFasta("../data/HemoPI_1_pos_val20_.fasta")

trainNegSeqDict = ldObj.readFasta("../data/neg_train.fasta")
trainPosSeqDict = ldObj.readFasta("../data/pos_train.fasta")
indpNegSeqDict = ldObj.readFasta("../data/neg_test.fasta")
indpPosSeqDict = ldObj.readFasta("../data/pos_test.fasta")

print(len(trainNegSeqDict))
print(len(trainPosSeqDict))
print(len(indpNegSeqDict))
print(len(indpPosSeqDict))

print("fasta data is load")

# 清洗掉序列中的 "-" 符號
def clean_sequences(data_dict):
    if data_dict is None: return None
    # 移除序列中的 "-" 並過濾掉空序列
    return {k: v.replace('-', '').upper() for k, v in data_dict.items() if v}

trainNegSeqDict = clean_sequences(trainNegSeqDict)
trainPosSeqDict = clean_sequences(trainPosSeqDict)
indpNegSeqDict = clean_sequences(indpNegSeqDict)
indpPosSeqDict = clean_sequences(indpPosSeqDict)

print("Sequences cleaned: Gaps '-' removed.")

# 1. 計算數量
numTrainPos = len(trainPosSeqDict)
numTrainNeg = len(trainNegSeqDict)
numIndpPos = len(indpPosSeqDict)
numIndpNeg = len(indpNegSeqDict)

# 2. 計算比例 (正樣本 / 負樣本)
trainRatio = numTrainPos / numTrainNeg if numTrainNeg != 0 else 0
indpRatio = numIndpPos / numIndpNeg if numIndpNeg != 0 else 0

# 3. 列印結果
print("="*30)
print(f"{'Dataset Type':<15} | {'Positive':<10} | {'Negative':<10} | {'P/N Ratio':<10}")
print("-"*30)
print(f"{'Training Set':<15} | {numTrainPos:<10} | {numTrainNeg:<10} | {trainRatio:.2f}")
print(f"{'Independent Set':<15} | {numIndpPos:<10} | {numIndpNeg:<10} | {indpRatio:.2f}")
print("="*30)

# 計算總數
print(f"Total Sequences: {numTrainPos + numTrainNeg + numIndpPos + numIndpNeg}")

# indpNegSeqDict2 = ldObj.readFasta("../data/neg_val.fasta")
# indpPosSeqDict2 = ldObj.readFasta("../data/pos_val.fasta")
# print("test data is load")
trainDataDict = {0: trainNegSeqDict, 1: trainPosSeqDict, -1: None}
indpDataDict = {0: indpNegSeqDict, 1: indpPosSeqDict, -1: None}
# indpDataDict2 = {0: indpNegSeqDict2, 1: indpPosSeqDict2, -1: None}  # 第二組 indp test data

encodeObj = EncodeAllFeatures()

encodeObj.dataEncodeSetup(saveFeatureDict=featureDict,  # normalization 前傳出來
                          saveJsonPath=paramPath + f'{dataName}_featureTypeDict.json',  # 把 featureDict 存至 json 檔
                          loadJsonPath=None,  # 讀取 featureDict 的 pkl 檔
                          b_loadJson=False)  # True: 讀取 featureDict 的 pkl 檔 (loadJsonPath), False: 把 featureDict 存至 pkl 檔 (saveJsonPath)

encodeTrainDf = encodeObj.dataEncodeOutPut(dataDict=trainDataDict)
encodeIndpDf1 = encodeObj.dataEncodeOutPut(dataDict=indpDataDict)
# encodeIndpDf2 = encodeObj.dataEncodeOutPut(dataDict=indpDataDict2)  # 第二組 indp test data

# ======================================================================================================================
# normalization
nmlzScalerPath = paramPath + f'{dataName}_{normalizeMethod}Scaler.pkl'

delNanTrainDf = FeatureStat.delNan(data=encodeTrainDf, logPath="../data/mlData/delNanTrain.txt")
delNanIndpDf = FeatureStat.delNan(data=encodeIndpDf1, logPath="../data/mlData/delNanIndp.txt")
# delNanIndpDf2 = FeatureStat.delNan(data=encodeIndpDf2, logPath="../data/mlData/delNanIndp2.txt")
print("missing data is clear")
trainNmlzDf = encodeObj.dataNormalization(encodeTrainDf=delNanTrainDf,
                                          encodeIndpDf=None,  # train scaler存起來 ，indp 另外做
                                          normalization=normalizeMethod,
                                          saveNmlzScalerPklPath=nmlzScalerPath,
                                          loadNmlzScalerPklPath=None,
                                          b_loadPkl=False)  # True: 讀取 NmlzScaler 的 pkl 檔 (loadNmlzScalerPklPath)
# False: 把 NmlzScaler 存至 pkl 檔 (saveNmlzScalerPklPath)

indpNmlzDf1 = encodeObj.dataNormalization(encodeTrainDf=None,
                                          encodeIndpDf=delNanIndpDf,
                                          normalization=normalizeMethod,
                                          saveNmlzScalerPklPath=None,
                                          loadNmlzScalerPklPath=nmlzScalerPath,
                                          b_loadPkl=True)  # indp test set 永遠使用 training set 存好的 NmlzScaler.pkl 檔

# indpNmlzDf2 = encodeObj.dataNormalization(encodeTrainDf=None,
#                                           encodeIndpDf=encodeIndpDf2,
#                                           normalization=normalizeMethod,
#                                           saveNmlzScalerPklPath=None,
#                                           loadNmlzScalerPklPath=nmlzScalerPath,
#                                           b_loadPkl=True)  # indp test set 永遠使用 training set 存好的 NmlzScaler.pkl 檔

# ================================================================================================================
# 把全部 feature 做完 nmlz 的結果存成 csv 檔 (可檢查 feature cutoff 以及 nmlz 的結果)
featureStatPath = '../data/featureStat/'
trainNmlzCsvPath = featureStatPath + f'train_{dataName}_{normalizeMethod}.csv'
indpNmlzCsvPath = featureStatPath + f'indp_{dataName}_{normalizeMethod}.csv'
featureAnalysisXlsxPath = featureStatPath + "featureAnalysis.xlsx"

trainNmlzDf.to_csv(trainNmlzCsvPath)  # 變數名稱不用ed
indpNmlzDf1.to_csv(indpNmlzCsvPath)

# Feature Stat 分析
featureStatObj = FeatureStat(dataPath=trainNmlzCsvPath)
# featureStatObj = FeatureStat(dataDf=trainNmlzDf)         # 也可 input dataframe
featureStatObj.sdAnalysis(
    saveFigPath=featureStatPath + f"sd_analysis_{dataName}_{normalizeMethod}.jpg")  # std deviation 分析結果 output 圖片
featureStatObj.featureValuePct_analysis(saveFinalExcel=featureAnalysisXlsxPath)  # 分析結果 output xlsx 檔
featureStatObj.pepCompositionAnalysis(posFastaPath="../data/pos_train.fasta",
                                     negFastaPath="../data/neg_train.fasta",
                                     saveXlsxPath=featureStatPath + 'pepCompositionAnalysis.xlsx')  # 找出只含有 1, 2 or 3 個 amino acid 的 peptide
# output xlsx 檔

# filteredTrainNmlzDf 為篩選後的 nmlz dataframe, 跑後續 boruta 用  (紀錄剩下幾個 feature)
filteredTrainNmlzDf, removeList = featureStatObj.processData(xlsxPath=featureAnalysisXlsxPath, columnName='top1percent',
                                                             number='+0.98', protectFeatSubstringList=['MotifBitVec'])
featureStatObj.processDataLog(logPath='../data/mlData/')
filterTrainNmlzPath = featureStatPath + f'filtered_train_{dataName}_{normalizeMethod}.csv'  # 移除完 feature 後新的 nmlz dataset 檔, 跑後續 boruta 用
removeFeatureListPath = featureStatPath + f'remove_feature_list_{dataName}_{normalizeMethod}.json'  # 移除掉的 feature 會存在這個文字檔裡
filteredTrainNmlzDf.to_csv(filterTrainNmlzPath)
with open(removeFeatureListPath, 'w') as f:
    json.dump(removeList, f)
# ================================================================================================================
skipFeatureList = [s for s in filteredTrainNmlzDf.columns if s.__contains__("MotifBitVec")]
brtObj = encodeObj.dataBoruta(borutaMethod='XGB', runBoruta=True, featRankPath=mlDataPath,
                              trainDf=filteredTrainNmlzDf, skipFeatureList=skipFeatureList)

encodeObj.dataEvalFeatureNum(startNum=5, endNum=400, step=5,
                             featNumScorePath=mlDataPath, saveCsvPath=mlDataPath,
                             trainDf=filteredTrainNmlzDf, indpDf=indpNmlzDf1, brtObj=brtObj,
                               foldNum=10, session = 100)   #sessionID可修改成任意整數，ex:1,4,10,15...



'''encodeObj.dataDecidedFeatureNum(featureNum=390, saveCsvPath=mlDataPath,
                                trainDf=filteredTrainNmlzDf, indpDf=indpNmlzDf1,
                                brtObj=brtObj) '''  # 決定好 feature 數字請開這個

# --- 報告用：特徵數量統計 ---

# 1. 原始特徵數 (剛 Encode 完)
# 扣除 ID 或 Label 欄位 (假設 columns 中包含這些，若不確定可直接 len)
original_total_features = len(encodeTrainDf.columns) 

# 2. 初步篩選後的特徵數 (經過 SD 或 Top1% 篩選)
filtered_features_count = len(filteredTrainNmlzDf.columns)
removed_count = len(removeList)

# 3. 獲取各類別特徵數 (可選，報告時更有條理)
def count_prefix(df, prefix):
    return len([col for col in df.columns if col.startswith(prefix)])

# 列印報告數值
print("\n" + "="*40)
print(f"{'Feature Selection Summary':^40}")
print("-" * 40)
print(f"Original Total Features:      {original_total_features}")
print(f"Removed (Low variance/High Corr): {removed_count}")
print(f"Filtered Features:            {filtered_features_count}")
print("-" * 40)

# 如果你有跑到 Boruta 或是最後決定的數量 (例如你最後註解寫的 390)
# final_selected = 390 
# print(f"Final Selected (Boruta/Best): {final_selected}")
print("="*40 + "\n")