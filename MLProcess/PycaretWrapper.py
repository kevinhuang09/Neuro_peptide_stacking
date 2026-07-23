from pycaret.classification import *
import os
import pandas as pd
import numpy as np


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
                                preprocess=True,   # 建議開啟，處理基礎特徵對齊
                                normalize=False,   # 外部已做過 Normalization 則關閉
                                # low_variance_threshold=None,
                                numeric_features=list(trainData.drop(columns=['y']).columns),
                                silent=True,
                                n_jobs=1)
            return pull()
        else:
            return None

    def doCompareModel(self, fold=5, n_select=None, sort='MCC', turbo=False, includeModelList=None):
        if includeModelList is None:
            includeModelList = ['rbfsvm', 'gbc', 'ridge', 'lr', 'catboost', 'lda', 'ada', 'knn', 'nb', 'et',
                                 'lightgbm', 'rf', 'xgboost', 'gpc', 'mlp', 'dt', 'svm', 'qda']
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
            includeModelList = ['rbfsvm', 'gbc', 'ridge', 'lr', 'catboost', 'lda', 'ada', 'knn', 'nb', 'et',
                                 'lightgbm', 'rf', 'xgboost', 'gpc', 'mlp', 'dt', 'svm', 'qda']
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

    # ==========================================================
    # ★★★ FIX: getOOFPredictions ★★★
    # ------------------------------------------------------------
    # 原本的寫法：
    #     oof_prediction = predict_model(model, verbose=False)
    # 這個呼叫【不帶 data】時，PyCaret 回傳的其實是 setup() 內部切出來
    # 的 hold-out 測試集（預設約 30%）的預測結果，長度遠小於整份
    # train_df，因此在 meta_feature_gent.py 把結果塞回
    # temp_train_probs[col_name] = prob_1_train.values
    # 時，會因為長度對不上 (ex: 1580 vs 5264) 而丟出：
    #     Length of values (1580) does not match length of index (5264)
    #
    # 修正方式：改用 sklearn 的 cross_val_predict，對「整份」
    # train_df 做 K-fold CV，每一筆樣本的機率都來自沒看過它的那個
    # fold，保證輸出長度一定等於 len(train_df)，同時仍是無資料
    # 洩漏的 Out-of-Fold 機率，可以放心當作 Stacking 第二層的訓練特徵。
    # ==========================================================
    def getOOFPredictions(self, model, train_df, cv=5, random_state=100):
        """
        ✨ 真正無洩漏、且長度一定等於 len(train_df) 的 Out-of-Fold 預測
        (用於 Stacking 第二層訓練)

        :param model: create_model() 回傳的 (sklearn相容) 模型
        :param train_df: 該特徵類別的完整訓練集 DataFrame (需含 'y' 欄)
        :param cv: K-fold 折數
        :param random_state: 讓每次切分可重現
        :return: DataFrame(index=train_df.index, columns=['Score', 'Label'])
        """
        from sklearn.model_selection import StratifiedKFold, cross_val_predict

        X = train_df.drop(columns=['y'])
        y = train_df['y']

        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

        try:
            oof_proba = cross_val_predict(
                model, X, y, cv=skf, method='predict_proba', n_jobs=1
            )
            score_1 = oof_proba[:, 1]
        except AttributeError:
            # 少數模型 (例如某些設定下的 ridge) 沒有 predict_proba，
            # 改用 decision_function 並用 min-max 正規化近似成機率
            oof_scores = cross_val_predict(
                model, X, y, cv=skf, method='decision_function', n_jobs=1
            )
            oof_scores = np.asarray(oof_scores).reshape(-1)
            mn, mx = oof_scores.min(), oof_scores.max()
            score_1 = (oof_scores - mn) / (mx - mn) if mx > mn else np.full_like(oof_scores, 0.5)

        oof_label = (score_1 > 0.5).astype(int)

        result = pd.DataFrame(
            {'Score': score_1, 'Label': oof_label},
            index=train_df.index
        )
        # 長度保證與 train_df 完全一致，供呼叫端直接
        # temp_train_probs[col_name] = result['Score'].values 使用
        assert len(result) == len(train_df), \
            f"OOF 預測長度 {len(result)} 與 train_df 長度 {len(train_df)} 不一致"
        return result

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
        os.makedirs(path, exist_ok=True)
        target_list = self.finalModelList if b_isFinalizedModel else self.tunedModelList
        suffix = '_final' if b_isFinalizedModel else '_tuned'
        for model, name in zip(target_list, self.modelNameList):
            save_path = os.path.join(path, name + suffix)
            save_model(model, save_path)

    def doLoadModel(self, path, fileNameList=None, b_isFinalizedModel=True):
        if fileNameList is None:
            fileNameList = ['rbfsvm', 'gbc', 'ridge', 'lr', 'catboost', 'lda', 'ada', 'knn', 'nb', 'et',
                             'lightgbm', 'rf', 'xgboost', 'gpc', 'mlp', 'dt', 'svm', 'qda']
        modelList = []
        suffix = '_final' if b_isFinalizedModel else '_tuned'
        for fileName in fileNameList:
            loadPath = os.path.join(path, fileName + suffix)
            loaded_pipeline = load_model(loadPath)
            # 提取 Pipeline 中最後一個步驟 (即訓練好的模型)
            resultModel = loaded_pipeline.named_steps.trained_model
            modelList.append(resultModel)
        return modelList
