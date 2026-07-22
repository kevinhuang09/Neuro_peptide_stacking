import init_env, os, sys, config
import pandas as pd
import warnings
from concurrent.futures import ProcessPoolExecutor
from data_cleaning import remove_constant_features
# ignore some warnings
warnings.filterwarnings('ignore')
from MLProcess.PycaretWrapper import PycaretWrapper
# ==========================================
# 1. set data path and model list
# ==========================================
dataName = config.DATA_NAME
process1_t = config.process1_t
process2_t = config.process2_t
# SPLIT_DATA_PATH = f"../data/featureStat/split_features_{DATA_NAME}/"
SPLIT_DATA_PATH = f"../data/featureStat/split_features_{dataName}_{process1_t}_{process2_t}/"
# PROB_OUTPUT_PATH = f"../data/mlData/prob_tables_features_{DATA_NAME}/"
PROB_OUTPUT_PATH = f"../data/mlData/split_features_{dataName}_{process1_t}_{process2_t}/"
os.makedirs(PROB_OUTPUT_PATH, exist_ok=True)
MODEL_LIST = [
    'gbc', 'ridge', 'lr', 'catboost', 'lda', 'ada', 'knn', 'nb', 'et',
    'lightgbm', 'rf', 'xgboost', 'mlp', 'dt', 'svm', 'qda', 'rbfsvm'
]

# CV_FOLD 用於 getOOFPredictions 內部的 K-fold 折數，可依需要調整
CV_FOLD = 5


# ==========================================
# 2. tune model one by one(feature types * models)
# ==========================================
def process_feature_group(f_prefix):
    """
    針對單一特徵類別 (例如 AAC) 訓練模型並獲取 Train/Test 機率值
    """
    # train_file = f"../data/featureStat/split_features/train/{f_prefix.lower()}_train.csv"
    # test_file = f"../data/featureStat/split_features/test/{f_prefix.lower()}_test.csv"
    train_file = os.path.join(SPLIT_DATA_PATH, "train", f"{f_prefix}_train.csv")
    test_file = os.path.join(SPLIT_DATA_PATH, "test", f"{f_prefix}_test.csv")
    if not os.path.exists(train_file) or not os.path.exists(test_file):
        return None, None, None

    train_df = pd.read_csv(train_file, index_col=0)
    test_df = pd.read_csv(test_file, index_col=0)

    pycObj = PycaretWrapper()
    # Initial pycaret, fix sessionID ensure experiment can try again
    pycObj.doSetup(train_df, sessionID=100)

    temp_train_probs = pd.DataFrame(index=train_df.index)
    temp_test_probs = pd.DataFrame(index=test_df.index)

    print(f"feature being processing: {f_prefix} ...")
    for model_name in MODEL_LIST:
        try:
            # 1. 建立模型並對 Train 與 Test 進行預測
            model = pycObj.createModelCustom(model_name)

            # ------------------------------------------------------------
            # ★★★ FIX ★★★
            # 原本：train_pred = pycObj.getOOFPredictions(model)
            # 這樣拿到的只是 PyCaret setup() 內部切出來的 hold-out
            # (約 30% train_df)，長度跟 train_df 對不上，才會出現
            # Length of values (1580) does not match length of index (5264)
            #
            # 修正：多帶入 train_df，讓 getOOFPredictions 內部改用
            # K-fold cross_val_predict 對「整份」 train_df 做 OOF 預測，
            # 保證回傳長度一定等於 len(train_df)。
            # ------------------------------------------------------------
            train_pred = pycObj.getOOFPredictions(model, train_df=train_df, cv=CV_FOLD)
            # train_pred = pycObj.predictModelCustom(model, data=train_df)
            test_pred = pycObj.predictModelCustom(model, data=test_df)

            # 2. 動態抓取欄位名稱
            label_col = 'Label' if 'Label' in train_pred.columns else 'prediction_label'
            score_col = 'Score' if 'Score' in train_pred.columns else 'prediction_score'

            if label_col in train_pred.columns and score_col in train_pred.columns:
                col_name = f"{f_prefix}_{model_name}"

                # 3. 無懼型別的強健判定法！
                # 將 label 轉為字串，只要是 '1' 或 '1.0' 就判定為預測 Class 1
                prob_1_train = train_pred.apply(
                    lambda row: row[score_col] if str(row[label_col]).strip() in ['1', '1.0'] else 1.0 - row[score_col],
                    axis=1
                )
                temp_train_probs[col_name] = prob_1_train.values

                prob_1_test = test_pred.apply(
                    lambda row: row[score_col] if str(row[label_col]).strip() in ['1', '1.0'] else 1.0 - row[score_col],
                    axis=1
                )
                temp_test_probs[col_name] = prob_1_test.values
            else:
                print(f"找不到預測欄位，{model_name} 回傳的欄位有: {train_pred.columns.tolist()}")
        except Exception as e:
            print(f"模型 {model_name} 在特徵 {f_prefix} 上失敗: {e}")

    # 確保回傳所有資料供主進程收集
    return f_prefix, temp_train_probs, temp_test_probs


# ==========================================
# 3. concurrent core process
# ==========================================
if __name__ == "__main__":
    train_dir = os.path.join(SPLIT_DATA_PATH, "train")
    all_files = sorted([f for f in os.listdir(train_dir) if f.endswith("_train.csv")])
    feature_prefixes = [f.split('_train.csv')[0] for f in all_files]

    print(f"start to train {len(feature_prefixes) * len(MODEL_LIST)} models")

    all_train_prob_dfs = []
    all_test_prob_dfs = []

    # process feature types
    with ProcessPoolExecutor(max_workers=12) as executor:
        futures = [executor.submit(process_feature_group, pref) for pref in feature_prefixes]
        for future in futures:
            try:
                pref, tr_prob, ts_prob = future.result()
                if tr_prob is not None:
                    all_train_prob_dfs.append(tr_prob)
                    all_test_prob_dfs.append(ts_prob)
            except Exception as e:
                print(f"子進程崩潰: {e}")

    # merge all of prob field
    final_train_prob_df = pd.concat(all_train_prob_dfs, axis=1)
    final_test_prob_df = pd.concat(all_test_prob_dfs, axis=1)

    # read index 0 to get y seq
    sample_y = pd.read_csv(os.path.join(train_dir, all_files[0]), index_col=0)['y']
    final_train_prob_df['y'] = sample_y

    # 讀取測試集的標籤
    test_sample_y = pd.read_csv(
        os.path.join(SPLIT_DATA_PATH, "test", os.listdir(os.path.join(SPLIT_DATA_PATH, "test"))[0]),
        index_col=0
    )['y']
    final_test_prob_df['y'] = test_sample_y

    # remove unique value
    final_train_prob_df, final_test_prob_df = remove_constant_features(final_train_prob_df, final_test_prob_df, dropna=False)

    # store prob table (修正檔名以顯示正確的 test 維度)
    final_train_prob_df.to_csv(os.path.join(PROB_OUTPUT_PATH, f"train_prob_{final_train_prob_df.shape}.csv"))
    final_test_prob_df.to_csv(os.path.join(PROB_OUTPUT_PATH, f"test_prob_{final_test_prob_df.shape}.csv"))

    print("\n" + "=" * 40)
    print(f"complete!")
    print(f"prob output train feature dimension: {final_train_prob_df.shape}")
    print(f"prob output test feature dimension: {final_test_prob_df.shape}")
    print(f"file already store in: {PROB_OUTPUT_PATH}")
    print("=" * 40)
