import init_env

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
from MLProcess.change_binary import changeBinaryFeatureInDf

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

print(f"path set is OK")

featNumber = 270
# test model list
modelNameList = ['mlp', 'rf']

# modelNameList = ['lightgbm', 'catboost', 'rbfsvm', 'gbc', 'ridge', 'lr', 'lda', 'ada', 'knn', 'nb', 'et', 'rf',
#                  'xgboost', 'mlp', 'dt', 'svm', 'qda']

dataTrainDf = pd.read_csv(mlDataPath + f'/{featNumber}/' + "train_F" + str(featNumber) + ".csv",
                          index_col=[0])  # 例如檔名為train_F190.csv

print(f"feature file read is OK")

dataTrainDf = changeBinaryFeatureInDf(dataTrainDf)

print(f"change binary is OK")

pycObj = PycaretWrapper()
setupDf = pycObj.doSetup(trainData=dataTrainDf)

print(f"pycaret warpper setup OK")

# 一般情況建議使用 TPE search
# pycObj.doTuneModel(searchLibrary='optuna', searchAlg='tpe', includeModelList=modelNameList, foldNum=10,
#                    n_iter=100, early_stopping=False, customGridDict=None)

pycObj.doTuneModel(searchLibrary='optuna', searchAlg='tpe', includeModelList=modelNameList, foldNum=2,
                   n_iter=3, early_stopping=False, customGridDict=None)
# 因為樣本數少，所以最後可改為flodNUM = 20

# 以grid search 做 tune model (通常使用於RBFSVM)
# CList = [2**-5, 2**-3, 2**-1, 2**1, 2**3, 2**5, 2**7, 2**9, 2**11, 2**13, 2**15]
# gammaList = [2**-15, 2**-13, 2**-11, 2**-9, 2**-7, 2**-5, 2**-3, 2**-1, 2**1, 2**3]
# customGridDict = {'rbfsvm': {'C': CList, 'gamma': gammaList}}
# pycObj.doTuneModel(searchLibrary='scikit-learn', searchAlg='grid', includeModelList=['rbfsvm'], foldNum=5,
#                    n_iter=100, early_stopping=False', customGridDict=customGridDict)


pycObj.doSaveModel(tuneModelPath, b_isFinalizedModel=False)  # 儲存tune好的model
print(f"model save complete")

tunedModelList = pycObj.doLoadModel(tuneModelPath, fileNameList=modelNameList,
                                    b_isFinalizedModel=False)  # 讀取tune好的model
print(f"load models")

tunedModelParamList, scoreRank = pycObj.doCompareModel(fold=10,
                                                       includeModelList=tunedModelList)  # 用tune好的model去做CV,scoreRank即為CV的表格
scoreRank.to_csv(mlScorePath + 'cvScore.csv')

#####################################################################
#     拿FinalizeModel做independent test                              #
#####################################################################

dataIndp = pd.read_csv(mlDataPath + f'/{featNumber}/' + "indp_F" + str(featNumber) + ".csv",
                       index_col=[0])  # 例如檔名為Indp_F190.csv   ##加上test1/2
dataIndp = changeBinaryFeatureInDf(dataIndp)
dataIndp_X = dataIndp.drop(["y"], axis=1)
dataIndp_y = dataIndp[["y"]]
pycObj.doFinalizeModel()
pycObj.doSaveModel(finalModelPath, b_isFinalizedModel=True)
finalModelList = pycObj.doLoadModel(path=finalModelPath, fileNameList=modelNameList, b_isFinalizedModel=True)
predObjIndp = Predict(dataX=dataIndp_X, modelList=finalModelList)
predVectorListIndp, probVectorListIndp = predObjIndp.doPredict()  # predVectorListIndp：model 產生的原始 0 or 1 預測結果, probVectorListIndp：model 產生的原始 probability 預測結果
# predVectorDf = pd.DataFrame(predVectorListIndp, pepName=modelNameList, columns=dataIndp.pepName).T
probVectorDf = pd.DataFrame(probVectorListIndp, index=modelNameList, columns=dataIndp.index).T
# predVectorDf.to_csv(mlScorePath + 'predVector.csv')
probVectorDf.to_csv(mlScorePath + 'probVector.csv')
scoreObjIndp = Scoring(predVectorList=predVectorListIndp, probVectorList=probVectorListIndp,
                       answerDf=dataIndp_y, modelNameList=modelNameList)
bestPredVectorListIndp = scoreObjIndp.optimizeMcc(cutOffList=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],  # bestPredVectorListIndp：probVectorListIndp 經過最佳 cutoff 轉出的 0 or 1 vector (binary prediction)
                                                  method='mcc',
                                                  bestCutoffJsonPath=f'{paramPath}/bestCutoff.json')
bestPredVectorDf = pd.DataFrame(bestPredVectorListIndp, index=modelNameList, columns=dataIndp.index).T

bestPredVectorDf.to_csv('../data/predResultDf.csv')

scoreDfIndp = scoreObjIndp.doScoring(b_optimizedMcc=True, path=mlScorePath + 'singleModelScore_Indp.csv',
                                     sortColumn='mcc')
#plotPredConfidence
scoreObjIndp.plotPredConfidence(predictionsList=probVectorListIndp, trueLabelsDf=dataIndp_y, numBins=10,
                                modelNameList=modelNameList,
                                outputExcel=mlScorePath + "plotPredConfidence.xlsx",
                                figSave=True,
                                figSavePath=mlScorePath)
drawObj = DrawPlot(answerDf=dataIndp_y, modelList=finalModelList, modelNameList=modelNameList,
                   predArrList=predVectorListIndp, probArrList=probVectorListIndp)
aucDf = drawObj.drawROC(colorList=None, title=False, titleName='Receiver Operating Characteristic', setDpi=True,
                        legendSize=11, labelSize=20, save=True, saveLoc=mlScorePath + 'Multi_Single_Model.png',
                        show=True,
                        dpi=300, figSize=(12, 9))

# #####################################################################
# #           voting & stacking                                       #
# #####################################################################

# clusterNumList = [3, 5, 7, 9, 11]  # [2, 3]
# # test models list
# modelNameList = ['lr']
# # modelNameList = ['rbfsvm', 'gbc', 'ridge', 'lr', 'catboost', 'lda', 'ada', 'knn', 'nb', 'et', 'lightgbm', 'rf',
# #                  'xgboost', 'gpc', 'mlp', 'dt', 'svm', 'qda']
# stackObj = Stacking(loadModelPath=tuneModelPath,
#                     modelNameList=modelNameList,
#                     trainDataPath=mlDataPath + f'/{featNumber}/' + "train_F" + str(featNumber) + ".csv",
#                     testDataPath=mlDataPath + f'/{featNumber}/' + "indp_F" + str(featNumber) + ".csv",
#                     clusterNumList=clusterNumList)
# selfTestScoreDf = stackObj.genSelfTestResult(drawPlot=True)
# selfTestScoreDf.to_csv(mlScorePath + 'selfTestScore.csv')
# stkModelList = stackObj.genStkModel(selfTestScoreLabel='auc', final_estimator=LogisticRegression(), drawPlot=True,
#                                     metric='euclidean', linkageType='ward')
# stkScoreDf = stackObj.stkModelPredictScoring(scoreCsvPath=mlScorePath + 'stkModelScore.csv', drawPlot=True,
#                                              b_isBinary=True)

# voteNumList = [3, 5, 7, 9, 11]
# scoreVoteDfIndp, scoreVoteProbDfIndp = voting(dataIndp=dataIndp,
#                                               loadModelPath=finalModelPath,
#                                               voteNumList=voteNumList,
#                                               selfTestScoreDf=selfTestScoreDf,
#                                               sortingMeasure='mcc')

# scoreVoteDfIndp.to_csv(mlScorePath + 'VoteScorePred.csv')
# scoreVoteProbDfIndp.to_csv(mlScorePath + 'VoteScoreProb.csv')
