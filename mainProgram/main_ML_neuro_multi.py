import matplotlib
matplotlib.use("Agg")  # 禁止跳出繪圖視窗
import sys, os, gc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor

# 處理路徑問題
current_dir = os.path.dirname(os.path.abspath(__file__))
parent = os.path.join(current_dir, "..")
sys.path.append(parent)

# 導入你的自定義模組
from MLProcess.PycaretWrapper import PycaretWrapper
from MLProcess.Predict import Predict
from MLProcess.Scoring import Scoring
from MLProcess.Stacking import Stacking
from MLProcess.Voting import Voting
from MLProcess.DrawPlot import DrawPlot
from MLProcess.change_binary import changeBinaryFeatureInDf

# 定義單次實驗的函數
def run_ml_experiment(i):
    """
    執行第 i 次實驗的完整流程
    """
    print(f"\n>>> [Run {i}] 啟動 (Process ID: {os.getpid()})")
    
    # 參數設定
    featNumber = 290
    modelNameList = ['lightgbm', 'catboost', 'rbfsvm', 'gbc', 'ridge', 'lr', 'lda', 'ada', 
                     'knn', 'nb', 'et', 'rf', 'xgboost', 'mlp', 'dt', 'svm', 'qda']
    
    mlDataPath = "../data/mlData/"
    paramPath = "../data/param/"
    base_mlScorePath = "../data/mlScore/"
    base_finalModelPath = "../data/finalModel/"
    base_tuneModelPath = "../data/tuneModel/"

    # 1. 建立當次專屬資料夾
    run_score_path = os.path.join(base_mlScorePath, f"run_{i}/")
    run_final_path = os.path.join(base_finalModelPath, f"run_{i}/")
    run_tune_path = os.path.join(base_tuneModelPath, f"run_{i}/")
    
    for p in [run_score_path, run_final_path, run_tune_path]:
        os.makedirs(p, exist_ok=True)

    try:
        # --- 2. 資料處理 ---
        train_file = os.path.join(mlDataPath, str(featNumber), f"train_F{featNumber}.csv")
        dataTrainDf = pd.read_csv(train_file, index_col=[0])
        dataTrainDf = changeBinaryFeatureInDf(dataTrainDf)
        
        pycObj = PycaretWrapper()
        setupDf = pycObj.doSetup(trainData=dataTrainDf)
        
        # --- 3. 模型調優 ---
        pycObj.doTuneModel(searchLibrary='optuna', searchAlg='tpe', includeModelList=modelNameList, 
                           foldNum=10, n_iter=100, early_stopping=False)

        pycObj.doSaveModel(run_tune_path, b_isFinalizedModel=False)
        tunedModelList = pycObj.doLoadModel(run_tune_path, fileNameList=modelNameList, b_isFinalizedModel=False)
        
        tunedModelParamList, scoreRank = pycObj.doCompareModel(fold=10, includeModelList=tunedModelList)
        scoreRank.to_csv(os.path.join(run_score_path, 'cvScore.csv'))

        # --- 4. Independent test ---
        indp_file = os.path.join(mlDataPath, str(featNumber), f"indp_F{featNumber}.csv")
        dataIndp = pd.read_csv(indp_file, index_col=[0])
        dataIndp = changeBinaryFeatureInDf(dataIndp)
        dataIndp_X = dataIndp.drop(["y"], axis=1)
        dataIndp_y = dataIndp[["y"]]
        
        pycObj.doFinalizeModel()
        pycObj.doSaveModel(run_final_path, b_isFinalizedModel=True)
        finalModelList = pycObj.doLoadModel(path=run_final_path, fileNameList=modelNameList, b_isFinalizedModel=True)
        
        predObjIndp = Predict(dataX=dataIndp_X, modelList=finalModelList)
        predVectorListIndp, probVectorListIndp = predObjIndp.doPredict()
        
        probVectorDf = pd.DataFrame(probVectorListIndp, index=modelNameList, columns=dataIndp.index).T
        probVectorDf.to_csv(os.path.join(run_score_path, 'probVector.csv'))

        # --- 5. 評分與優化 ---
        scoreObjIndp = Scoring(predVectorList=predVectorListIndp, probVectorList=probVectorListIndp,
                               answerDf=dataIndp_y, modelNameList=modelNameList)
        
        bestCutoffPath = os.path.join(paramPath, f'bestCutoff_run_{i}.json')
        bestPredVectorListIndp = scoreObjIndp.optimizeMcc(
            cutOffList=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            method='mcc',
            bestCutoffJsonPath=bestCutoffPath
        )
        
        bestPredVectorDf = pd.DataFrame(bestPredVectorListIndp, index=modelNameList, columns=dataIndp.index).T
        bestPredVectorDf.to_csv(os.path.join(run_score_path, 'predResultDf.csv'))

        scoreDfIndp = scoreObjIndp.doScoring(b_optimizedMcc=True, 
                                            path=os.path.join(run_score_path, 'singleModelScore_Indp.csv'),
                                            sortColumn='mcc')

        # --- 6. 繪圖 ---
        scoreObjIndp.plotPredConfidence(predictionsList=probVectorListIndp, trueLabelsDf=dataIndp_y, numBins=10,
                                        modelNameList=modelNameList,
                                        outputExcel=os.path.join(run_score_path, "plotPredConfidence.xlsx"),
                                        figSave=True, figSavePath=run_score_path)
        
        drawObj = DrawPlot(answerDf=dataIndp_y, modelList=finalModelList, modelNameList=modelNameList,
                           predArrList=predVectorListIndp, probArrList=probVectorListIndp)
        
        drawObj.drawROC(colorList=None, title=False, setDpi=True,
                        save=True, saveLoc=os.path.join(run_score_path, 'Multi_Single_Model.png'),
                        show=False, dpi=300, figSize=(12, 9))

        print(f"--- [Run {i}] 完成 ---")
        return f"Run {i}: Success"

    except Exception as e:
        print(f"!!! [Run {i}] 發生錯誤: {str(e)}")
        return f"Run {i}: Error ({str(e)})"

    finally:
        # 強制清理記憶體，避免子程序佔用過多 RAM
        plt.close('all')
        gc.collect()

# --- 主程式執行入口 ---
if __name__ == "__main__":
    # 設定並行核心數 (請根據你的 RAM 大小調整，建議先設為 2~4)
    MAX_WORKERS = 10
    total_runs = range(1, 21)

    print("="*50)
    print(f"開始執行並行運算，預計跑 {len(total_runs)} 次實驗")
    print(f"並行核心數設定為: {MAX_WORKERS}")
    print("="*50)

    # 使用 ProcessPoolExecutor 執行多核心任務
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 使用 map 執行
        results = list(executor.map(run_ml_experiment, total_runs))

    print("\n" + "="*50)
    print("所有任務執行完畢。總覽：")
    for res in results:
        print(res)
    print("="*50)