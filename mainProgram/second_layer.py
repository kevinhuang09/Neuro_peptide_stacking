import matplotlib
matplotlib.use("Agg")  # 禁止跳出繪圖視窗
# 處理路徑問題
import sys, os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent = os.path.join(current_dir, "..")
sys.path.append(parent)

import pandas as pd
import numpy as np
import random
from MLProcess.PycaretWrapper import PycaretWrapper
from MLProcess.Predict import Predict
from MLProcess.Scoring import Scoring
from MLProcess.Stacking import Stacking
from MLProcess.Voting import Voting
from MLProcess.DrawPlot import DrawPlot
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from collections import Counter

def changeBinaryFeatureInDf(dataDf):
    """
    因為 LightGBM 無法接受 binary feature 以及 int64 type 的 feature, 所以人工處理.
    binary feature: 隨機挑 1% 來改變數值 (加減 0.01 or 乘上 0.99/1.01)
    int64 type feature: 手動轉換為 float64 type
    :param dataDf:
    :return:
    """
    pd.options.mode.chained_assignment = None
    threshold = len(dataDf.index.tolist()) * (95 / 100)
    for column in dataDf.columns.to_list():
        dfUniqueValue = dataDf[column].unique()
        count_class = Counter(dataDf[column])
        dfCounterValue = pd.DataFrame.from_dict(count_class, orient='index', columns=["Count"])
        dfCounterValueSum = dfCounterValue['Count'].nlargest(2).sum()
        if (dfCounterValueSum >= threshold) and (column != 'y'):
            value1 = dfCounterValue['Count'].nlargest(2).index.tolist()[0]
            value2 = dfCounterValue['Count'].nlargest(2).index.tolist()[1]
            if 0 in dfUniqueValue:
                convertMaxValue = value1 - 0.01
                convertMinValue = value2 + 0.01
                print('Value of ' + str(column) + ' Converted')
                print(str(value1) + '  --+0.01-->  ' + str(convertMaxValue))
                print(str(value2) + '  --+0.01-->  ' + str(convertMinValue))
            else:
                convertMaxValue = value1 * 0.99
                convertMinValue = value2 * 1.01
                print('Value of ' + str(column) + ' Converted')
                print(str(value1) + '  --*0.99-->  ' + str(convertMaxValue))
                print(str(value2) + '  --*1.01-->  ' + str(convertMinValue))
            value1Index = list(dataDf.loc[dataDf[column] == value1].index[:])
            value2Index = list(dataDf.loc[dataDf[column] == value2].index[:])
            countMax_int = int(np.ceil(len(value1Index) * 0.1))
            countMin_int = int(np.ceil(len(value2Index) * 0.1))
            randomMaxIndex = random.sample(value1Index, countMax_int)
            randomMinIndex = random.sample(value2Index, countMin_int)
            dataDf[column].loc[randomMaxIndex] = dataDf[column].loc[randomMaxIndex].replace(value1, convertMaxValue)
            dataDf[column].loc[randomMinIndex] = dataDf[column].loc[randomMinIndex].replace(value2, convertMinValue)
        if (dataDf[column].dtype.name == 'int64') and (column != 'y'):
            dataDf[column] = dataDf[column].astype('float64')
            print('Type of ' + str(column) + ' Converted')
            print('int64  ---->  float64')

    return dataDf


def voting(dataIndp, loadModelPath, voteNumList, selfTestScoreDf, sortingMeasure):
    dataIndp_X = dataIndp.drop(["y"], axis=1)
    dataIndp_y = dataIndp[["y"]]
    pycVoteObj = PycaretWrapper()
    vote01_predVectorListIndp = []
    voteFrac_probVectorListIndp = []
    voteFrac_predVectorListIndp = []
    voteModelNameList = []
    selfTestModelNameList = []
    selfTestScoreDf = selfTestScoreDf.sort_values(by=sortingMeasure, ascending=False)
    print('voting model selected by ' + str(sortingMeasure) + '!!!')
    selfTestModelIndexList = selfTestScoreDf.index.tolist()
    for selfTestModelIndex in selfTestModelIndexList:
        selfTestModelName = ''.join(selfTestModelIndex)
        selfTestModelNameList.append(selfTestModelName)
    for voteNum in voteNumList:
        fileNameList = selfTestModelNameList[:voteNum]
        finalModelList = pycVoteObj.doLoadModel(loadModelPath, b_isFinalizedModel=True, fileNameList=fileNameList)
        voteObjIndp = Voting(dataX=dataIndp_X, modelList=finalModelList)
        vote01_predVectorIndp = voteObjIndp.vote_predVector()
        voteFrac_predVectorIndp, voteFrac_probVectorIndp = voteObjIndp.vote_probVector(cutoff=0.5)
        vote01_predVectorListIndp.append(vote01_predVectorIndp)
        voteFrac_predVectorListIndp.append(voteFrac_predVectorIndp)
        voteFrac_probVectorListIndp.append(voteFrac_probVectorIndp)
        voteModelNameList.append('voting_' + str(voteNum))
        print('Voting model with ' + str(voteNum) + ' estimators: ' + str(fileNameList))
    '''算分 : 使用 [0, 1] 投票的結果,仍需probVector來計算AUC'''
    scoreObj = Scoring(predVectorList=vote01_predVectorListIndp, probVectorList=voteFrac_probVectorListIndp,
                       answerDf=dataIndp_y, modelNameList=voteModelNameList)
    scoreDf = scoreObj.doScoring(b_optimizedMcc=False,
                                 sortColumn='mcc')
    '''算分 : 使用 prob 投票的結果,仍需probVector來計算AUC'''
    scoreProbObj = Scoring(predVectorList=voteFrac_predVectorListIndp, probVectorList=voteFrac_probVectorListIndp,
                           answerDf=dataIndp_y,
                           modelNameList=voteModelNameList)
    scoreProbDf = scoreProbObj.doScoring(b_optimizedMcc=False,
                                         sortColumn='mcc')

    return scoreDf, scoreProbDf



#def pepProAna

#####################################################################
#   '''確定feature數目後做tunedModel & CV用的'''                       #
#####################################################################

mlDataPath = "../data/mlData/"  # 內含 feature selection 後的 data 檔案 ex : train_F390.csv, boruta 檔案 ex :Boruta-featRank-RF.csv
paramPath = "../data/param/"  # 內含檔案: featureTypeDict.pkl, normalize.pkl
finalModelPath = "../data/finalModel"  # train 好且 finalize 的 model 內含檔案 ex: lr_final.pkl
tuneModelPath = "../data/tuneModel"  # tune 好，但未進行finalize 的 model 內含檔案 ex: lr_tuned.pkl
mlScorePath = "../data/mlScore/"  # 內含 ml model 預測完並算好分的檔案 ex: cvScore.csv, singleModelScore_Indp.csv
# ratioPath = "../data/PeptideProperty/"
# ratioPicPath = "../data/PeptidePropertyPic/"

# avoid the folders is not exist
# auto to build table of contents
for path in [finalModelPath, tuneModelPath, mlScorePath, paramPath]:
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")

featNumber = 250
modelNameList = ['lightgbm', 'catboost', 'rbfsvm', 'gbc', 'ridge', 'lr', 'lda', 'ada', 'knn', 'nb', 'et', 'rf',
                 'xgboost', 'mlp', 'dt', 'svm', 'qda']

# --- 補上這段以確保環境初始化 ---
dataTrainDf = pd.read_csv(os.path.join(mlDataPath, f"train_F{featNumber}.csv"))
# 如果 CSV 第一欄是 index，請視情況加入 index_col=[0]

# 數據清洗 (針對 LightGBM)
dataTrainDf = changeBinaryFeatureInDf(dataTrainDf)

# 初始化 Pycaret 環境
pycObj = PycaretWrapper()
setupDf = pycObj.doSetup(trainData=dataTrainDf)

# if without tune models, using following two 
pycObj.doTuneModel(searchLibrary='optuna', searchAlg='tpe', includeModelList=modelNameList, n_iter=100)
pycObj.doSaveModel(tuneModelPath, b_isFinalizedModel=False)

# --- 3. 初始化 Stacking 物件 ---
# 這裡會讀取你剛產出的 train_F250.csv
stackObj = Stacking(
    loadModelPath=tuneModelPath,
    modelNameList=modelNameList,
    trainDataPath=os.path.join(mlDataPath, f"train_F{featNumber}.csv"),
    testDataPath=os.path.join(mlDataPath, f"indp_F{featNumber}.csv"),
    clusterNumList=[5] # 這裡暫時設一個數字即可，因為我們目標是 OOF features
)

# --- 4. 產生 550 個 Output Features (關鍵步驟) ---
print("正在執行 10-Fold OOF 預測以產生 Level-1 特徵...")

# 這個函式內部會跑 CV，並回傳每個模型對訓練集的預測機率 (OOF Probabilities)
# 這些機率值就是你流程圖中的 "Output Features"
selfTestScoreDf = stackObj.genSelfTestResult(drawPlot=False)

# 儲存結果
selfTestScoreDf.to_csv(os.path.join(mlScorePath, 'level_1_input_features.csv'))

print(f"成功！Level-1 所需的特徵已產出至: {mlScorePath}level_1_input_features.csv")