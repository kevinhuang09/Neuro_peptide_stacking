import matplotlib
matplotlib.use("Agg")  # 禁止跳出繪圖視窗
from concurrent.futures import ProcessPoolExecutor # 1. 導入並行庫
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
from MLProcess.change_binary import changeBinaryFeatureInDf
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from collections import Counter

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
# ... 前面 import 部分保持不變 ...

featNumber = 290
modelNameList = ['lightgbm', 'catboost', 'rbfsvm', 'gbc', 'ridge', 'lr', 'lda', 'ada', 'knn', 'nb', 'et', 'rf',
                 'xgboost', 'mlp', 'dt', 'svm', 'qda']

# 1. 基礎路徑設定 (建議寫在迴圈外)
base_mlScorePath = "../data/mlScore/"
base_finalModelPath = "../data/finalModel/"
base_tuneModelPath = "../data/tuneModel/"

for i in range(1, 21):
    print(f"\n>>>>>>> 第 {i} 次 test running : \n")
    
    # 2. 【關鍵】建立當次專屬的子資料夾路徑
    run_score_path = os.path.join(base_mlScorePath, f"run_{i}/")
    run_final_path = os.path.join(base_finalModelPath, f"run_{i}/")
    run_tune_path = os.path.join(base_tuneModelPath, f"run_{i}/")
    
    for p in [run_score_path, run_final_path, run_tune_path]:
        if not os.path.exists(p):
            os.makedirs(p)

    # --- 資料處理 ---
    dataTrainDf = pd.read_csv(mlDataPath + f'/{featNumber}/' + "train_F" + str(featNumber) + ".csv", index_col=[0])
    dataTrainDf = changeBinaryFeatureInDf(dataTrainDf)
    
    pycObj = PycaretWrapper()
    setupDf = pycObj.doSetup(trainData=dataTrainDf)
    
    pycObj.doTuneModel(searchLibrary='optuna', searchAlg='tpe', includeModelList=modelNameList, foldNum=10,
                       n_iter=100, early_stopping=False, customGridDict=None)

    # 3. 【更正】儲存路徑改為 run_tune_path
    pycObj.doSaveModel(run_tune_path, b_isFinalizedModel=False)
    tunedModelList = pycObj.doLoadModel(run_tune_path, fileNameList=modelNameList, b_isFinalizedModel=False)
    
    tunedModelParamList, scoreRank = pycObj.doCompareModel(fold=10, includeModelList=tunedModelList)
    
    # 4. 【更正】CSV 存到 run_score_path
    scoreRank.to_csv(run_score_path + 'cvScore.csv')

    # --- Independent test ---
    dataIndp = pd.read_csv(mlDataPath + f'/{featNumber}/' + "indp_F" + str(featNumber) + ".csv", index_col=[0])
    dataIndp = changeBinaryFeatureInDf(dataIndp)
    dataIndp_X = dataIndp.drop(["y"], axis=1)
    dataIndp_y = dataIndp[["y"]]
    
    pycObj.doFinalizeModel()
    
    # 5. 【更正】Final Model 路徑改為 run_final_path
    pycObj.doSaveModel(run_final_path, b_isFinalizedModel=True)
    finalModelList = pycObj.doLoadModel(path=run_final_path, fileNameList=modelNameList, b_isFinalizedModel=True)
    
    predObjIndp = Predict(dataX=dataIndp_X, modelList=finalModelList)
    predVectorListIndp, probVectorListIndp = predObjIndp.doPredict()
    
    probVectorDf = pd.DataFrame(probVectorListIndp, index=modelNameList, columns=dataIndp.index).T
    probVectorDf.to_csv(run_score_path + 'probVector.csv')

    scoreObjIndp = Scoring(predVectorList=predVectorListIndp, probVectorList=probVectorListIndp,
                           answerDf=dataIndp_y, modelNameList=modelNameList)
    
    # 6. 【更正】bestCutoff 檔名加上 run 編號避免衝突
    bestCutoffPath = os.path.join(paramPath, f'bestCutoff_run_{i}.json')
    bestPredVectorListIndp = scoreObjIndp.optimizeMcc(cutOffList=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                                                     method='mcc',
                                                     bestCutoffJsonPath=bestCutoffPath)
    
    bestPredVectorDf = pd.DataFrame(bestPredVectorListIndp, index=modelNameList, columns=dataIndp.index).T
    bestPredVectorDf.to_csv(run_score_path + 'predResultDf.csv') # 存入 run 資料夾

    # 7. 【更正】評分結果存到 run_score_path
    scoreDfIndp = scoreObjIndp.doScoring(b_optimizedMcc=True, path=run_score_path + 'singleModelScore_Indp.csv',
                                        sortColumn='mcc')

    # 8. 【更正】繪圖路徑全部更新為 run_score_path，並將 show 設為 False
    scoreObjIndp.plotPredConfidence(predictionsList=probVectorListIndp, trueLabelsDf=dataIndp_y, numBins=10,
                                    modelNameList=modelNameList,
                                    outputExcel=run_score_path + "plotPredConfidence.xlsx",
                                    figSave=True,
                                    figSavePath=run_score_path)
    
    drawObj = DrawPlot(answerDf=dataIndp_y, modelList=finalModelList, modelNameList=modelNameList,
                       predArrList=predVectorListIndp, probArrList=probVectorListIndp)
    
    aucDf = drawObj.drawROC(colorList=None, title=False, titleName='Receiver Operating Characteristic', setDpi=True,
                           legendSize=11, labelSize=20, save=True, saveLoc=run_score_path + 'Multi_Single_Model.png',
                           show=False, # 設為 False 才不會中斷迴圈
                           dpi=300, figSize=(12, 9))

    # 9. 【新增】釋放記憶體 (非常重要，否則跑 20 次電腦會當機)
    import gc
    import matplotlib.pyplot as plt
    plt.close('all') 
    del pycObj, setupDf, dataTrainDf, dataIndp, scoreObjIndp, drawObj
    gc.collect()

print("所有 20 次實驗已完成。")