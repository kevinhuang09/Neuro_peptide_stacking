from pycaret.classification import *
import os
import pandas as pd

class PycaretWrapper:
    def __init__(self):
        self.model = None
        self.modelNameList = None
        self.tunedModelList = None
        self.finalModelList = None

    def doSetup(self, trainData=None, testData=None, train_size=0.8, sessionID=None, needTrain=True):
        """
        初始化 Pycaret 設定
        :param trainData: 訓練集 DataFrame (需包含 'y' 標籤)
        :param testData: 獨立測試集 DataFrame
        :param preprocess: 開啟基礎預處理以確保 Target 對齊與類別處理
        """
        if needTrain:
            self.model = setup(data=trainData,
                               test_data=testData, 
                               target='y',
                               session_id=sessionID,
                               preprocess=True,    # 建議開啟，處理基礎特徵對齊
                               normalize=False,    # 外部已做過 Normalization 則關閉
                            #    low_variance_threshold=None,
                               numeric_features=list(trainData.drop(columns=['y']).columns), 
                               silent=True,
                               n_jobs=1)
            return pull()
        else:
            return None

    def doCompareModel(self, fold=5, n_select=None, sort='MCC', turbo=False, includeModelList=None):
        if includeModelList is None:
            includeModelList = ['rbfsvm', 'gbc', 'ridge', 'lr', 'catboost', 'lda', 'ada', 'knn', 'nb', 'et', 'lightgbm', 'rf', 'xgboost', 'gpc', 'mlp', 'dt', 'svm', 'qda']
        if n_select is None:
            n_select = len(includeModelList)
            
        defaultModelParamList = compare_models(fold=fold,
                                               n_select=n_select,
                                               sort=sort,
                                               turbo=turbo,
                                               include=includeModelList)
        scoreRank = pull()
        return defaultModelParamList, scoreRank
    def createModelCustom(self, model_name):
        """
        封裝 PyCaret 的 create_model 函式
        """
        # 建立並返回模型
        model = create_model(model_name)
        return model
    def doTuneModel(self, searchLibrary='optuna', searchAlg='tpe', includeModelList=None, foldNum=5,
                    n_iter=10, early_stopping_max_iters=10, early_stopping=False,
                    customGridDict=None):
        if includeModelList is None:
            includeModelList = ['rbfsvm', 'gbc', 'ridge', 'lr', 'catboost', 'lda', 'ada', 'knn', 'nb', 'et', 'lightgbm',
                                'rf', 'xgboost', 'gpc', 'mlp', 'dt', 'svm', 'qda']
        
        self.modelNameList = includeModelList
        self.tunedModelList = []
        tunerList = []

        for modelName in self.modelNameList:
            # 針對不同模型設定基礎參數
            if modelName == 'lr':
                defaultModel = create_model(modelName, fold=foldNum, max_iter=2000, verbose=False)
            else:
                defaultModel = create_model(modelName, fold=foldNum, verbose=False)
            
            # 獲取對應的自定義網格
            param = customGridDict.get(modelName) if customGridDict else None
            
            tuneModels, tuner = tune_model(defaultModel, 
                                           search_library=searchLibrary, 
                                           choose_better=True, 
                                           optimize='MCC',
                                           return_tuner=True, 
                                           search_algorithm=searchAlg, 
                                           return_train_score=True,
                                           fold=foldNum, 
                                           early_stopping_max_iters=early_stopping_max_iters, 
                                           n_iter=n_iter,
                                           early_stopping=early_stopping, 
                                           custom_grid=param,
                                           verbose=False)
            self.tunedModelList.append(tuneModels)
            tunerList.append(tuner)
            
        return self.tunedModelList, tunerList

    def getOOFPredictions(self, model):
        """
        ✨ 真正無洩漏版本：獲取 Out-of-Fold 預測 (用於 Stacking 第二層訓練)
        注意：不可傳入 data=train_data，否則會發生資料洩漏。
        """
        from pycaret.classification import predict_model
        
        # Pycaret predict_model(model) 不帶 data 參數時，
        # 會自動提取 Cross-validation 過程中各個 Hold-out fold 的預測結果。
        oof_prediction = predict_model(model, verbose=False) 

        # 欄位名稱對齊 (相容不同 Pycaret 版本)
        name_map = {
            'prediction_score': 'Score', 
            'prediction_label': 'Label',
            'Score': 'Score', 
            'Label': 'Label'
        }
        
        for old, new in name_map.items():
            if old in oof_prediction.columns:
                oof_prediction = oof_prediction.rename(columns={old: new})
        
        # 確保必要欄位存在
        if 'Score' not in oof_prediction.columns:
            oof_prediction['Score'] = 0.5
        if 'Label' not in oof_prediction.columns:
            oof_prediction['Label'] = (oof_prediction['Score'] > 0.5).astype(int)

        return oof_prediction[['Score', 'Label']]

    def predictModelCustom(self, model, data):
        """
        用於預測全新的資料 (如獨立測試集 Indp)
        """
        prediction = predict_model(model, data=data, verbose=False)
        
        # 處理 Score 與 Label 欄位名稱
        if 'Score' not in prediction.columns:
            score_cols = [c for c in prediction.columns if 'score' in c.lower()]
            prediction['Score'] = prediction[score_cols[0]] if score_cols else 0.5
        
        if 'Label' not in prediction.columns:
            label_cols = [c for c in prediction.columns if 'label' in c.lower()]
            prediction['Label'] = prediction[label_cols[0]] if label_cols else (prediction['Score'] > 0.5).astype(int)
                
        return prediction[['Score', 'Label']]

    def doFinalizeModel(self):
        """
        ⚠️ 警告：僅在所有實驗完成，要打包最終工具時才使用。
        Stacking 過程中產出機率特徵時，請使用 tunedModelList。
        """
        self.finalModelList = []
        for tunedModel in self.tunedModelList:
            finalModel = finalize_model(tunedModel)
            self.finalModelList.append(finalModel)

    def doSaveModel(self, path, b_isFinalizedModel=True):
        target_list = self.finalModelList if b_isFinalizedModel else self.tunedModelList
        suffix = '_final' if b_isFinalizedModel else '_tuned'
        
        for model, name in zip(target_list, self.modelNameList):
            save_path = os.path.join(path, name + suffix)
            save_model(model, save_path)

    def doLoadModel(self, path, fileNameList=None, b_isFinalizedModel=True):
        if fileNameList is None:
            fileNameList = ['rbfsvm', 'gbc', 'ridge', 'lr', 'catboost', 'lda', 'ada', 'knn', 'nb', 'et', 'lightgbm', 'rf', 'xgboost', 'gpc', 'mlp', 'dt', 'svm', 'qda']
        
        modelList = []
        suffix = '_final' if b_isFinalizedModel else '_tuned'
        
        for fileName in fileNameList:
            loadPath = os.path.join(path, fileName + suffix)
            loaded_pipeline = load_model(loadPath)
            # 提取 Pipeline 中最後一個步驟 (即訓練好的模型)
            resultModel = loaded_pipeline.named_steps.trained_model
            modelList.append(resultModel)
        return modelList