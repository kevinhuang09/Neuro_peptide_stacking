import init_env, os, sys, config
import numpy as np
import pandas as pd
import warnings
from concurrent.futures import ProcessPoolExecutor
from data_cleaning import remove_constant_features
warnings.filterwarnings('ignore')

# ★ 改用 sklearn 直接做 OOF，繞開 PyCaret 黑盒（避免對 train 自我預測）
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import (GradientBoostingClassifier, AdaBoostClassifier,
                              ExtraTreesClassifier, RandomForestClassifier)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# ==========================================
# 1. set data path and model list
# ==========================================
dataName = config.DATA_NAME
process1_t = config.process1_t
process2_t = config.process2_t
SPLIT_DATA_PATH = f"../data/featureStat/split_features_{dataName}_{process1_t}_{process2_t}/"
PROB_OUTPUT_PATH = f"../data/mlData/split_features_{dataName}_{process1_t}_{process2_t}/"
os.makedirs(PROB_OUTPUT_PATH, exist_ok=True)

MODEL_LIST = [
    'gbc', 'ridge', 'lr', 'catboost', 'lda', 'ada', 'knn', 'nb', 'et',
    'lightgbm', 'rf', 'xgboost', 'mlp', 'dt', 'svm', 'qda', 'rbfsvm'
]

# ★ 用 sklearn 對照 PyCaret 的 17 個模型代號，盡量對齊原本行為
def build_model(model_name):
    m = model_name.lower()
    if m == 'gbc':      return GradientBoostingClassifier(random_state=100)
    if m == 'ridge':    return RidgeClassifier(random_state=100)          # 無 predict_proba，用 decision_function
    if m == 'lr':       return LogisticRegression(max_iter=1000, random_state=100)
    if m == 'catboost': return CatBoostClassifier(verbose=0, random_state=100)
    if m == 'lda':      return LinearDiscriminantAnalysis()
    if m == 'ada':      return AdaBoostClassifier(random_state=100)
    if m == 'knn':      return KNeighborsClassifier()
    if m == 'nb':       return GaussianNB()
    if m == 'et':       return ExtraTreesClassifier(random_state=100)
    if m == 'lightgbm': return LGBMClassifier(random_state=100, verbose=-1)
    if m == 'rf':       return RandomForestClassifier(random_state=100)
    if m == 'xgboost':  return XGBClassifier(random_state=100, use_label_encoder=False, eval_metric='logloss')
    if m == 'mlp':      return MLPClassifier(max_iter=500, random_state=100)
    if m == 'dt':       return DecisionTreeClassifier(random_state=100)
    if m == 'svm':      return SVC(kernel='linear', probability=True, random_state=100)   # PyCaret 'svm'=linear SVM
    if m == 'qda':      return QuadraticDiscriminantAnalysis()
    if m == 'rbfsvm':   return SVC(kernel='rbf', probability=True, random_state=100)
    raise ValueError(f"未知模型代號: {model_name}")

# ★ 取「預測為 class 1 的機率」，處理沒有 predict_proba 的模型（ridge）
def oof_prob_class1(model, X, y, cv):
    try:
        # 多數模型：直接 OOF predict_proba，取 class 1 那欄
        proba = cross_val_predict(model, X, y, cv=cv, method='predict_proba', n_jobs=1)
        # 找出 class 1 對應的欄位（避免類別順序不是 [0,1] 的情況）
        classes = np.unique(y)
        idx1 = list(classes).index(1) if 1 in classes else -1
        return proba[:, idx1]
    except (AttributeError, ValueError):
        # ridge 等沒有 predict_proba：用 decision_function 做 min-max 正規化當作機率替代
        dfc = cross_val_predict(model, X, y, cv=cv, method='decision_function', n_jobs=1)
        dfc = np.asarray(dfc, dtype=float)
        rng = dfc.max() - dfc.min()
        return (dfc - dfc.min()) / rng if rng > 0 else np.full_like(dfc, 0.5)

# ==========================================
# 2. tune model one by one (feature types * models)  ★ train 改成 OOF
# ==========================================
def process_feature_group(f_prefix):
    """針對單一特徵類別 (例如 AAC) 產生 Train(OOF) / Test 機率值"""
    train_file = os.path.join(SPLIT_DATA_PATH, "train", f"{f_prefix}_train.csv")
    test_file  = os.path.join(SPLIT_DATA_PATH, "test",  f"{f_prefix}_test.csv")
    if not os.path.exists(train_file) or not os.path.exists(test_file):
        return None, None, None

    train_df = pd.read_csv(train_file, index_col=0)
    test_df  = pd.read_csv(test_file,  index_col=0)

    # ★ 切 X / y（label 欄叫 'y'）
    y_train = train_df['y'].astype(int).values
    X_train = train_df.drop(columns=['y'])
    X_test  = test_df.drop(columns=['y'])

    temp_train_probs = pd.DataFrame(index=train_df.index)
    temp_test_probs  = pd.DataFrame(index=test_df.index)

    # ★ 固定的 5-fold，讓每個 model 的 OOF 切法一致
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=100)

    print(f"feature being processing: {f_prefix} ...")
    for model_name in MODEL_LIST:
        try:
            col_name = f"{f_prefix}_{model_name}"

            # ---- train：OOF（不洩漏）----
            #   每筆樣本的機率，都來自「沒看過它的那折模型」
            prob_1_train = oof_prob_class1(build_model(model_name),
                                           X_train, y_train, cv)
            temp_train_probs[col_name] = prob_1_train

            # ---- test：用全 train 訓練一次，對 test 預測（test 本來就沒參與訓練）----
            model = build_model(model_name)
            model.fit(X_train, y_train)
            if hasattr(model, 'predict_proba'):
                classes = list(model.classes_)
                idx1 = classes.index(1) if 1 in classes else -1
                prob_1_test = model.predict_proba(X_test)[:, idx1]
            else:  # ridge：decision_function 正規化
                dfc = np.asarray(model.decision_function(X_test), dtype=float)
                rng = dfc.max() - dfc.min()
                prob_1_test = (dfc - dfc.min()) / rng if rng > 0 else np.full_like(dfc, 0.5)
            temp_test_probs[col_name] = prob_1_test

        except Exception as e:
            print(f"模型 {model_name} 在特徵 {f_prefix} 上失敗: {e}")

    return f_prefix, temp_train_probs, temp_test_probs

# ==========================================
# 3. concurrent core process  ★ 完全沿用你原本的 main，一行沒動
# ==========================================
if __name__ == "__main__":
    train_dir = os.path.join(SPLIT_DATA_PATH, "train")
    all_files = sorted([f for f in os.listdir(train_dir) if f.endswith("_train.csv")])
    feature_prefixes = [f.split('_train.csv')[0] for f in all_files]
    print(f"start to train {len(feature_prefixes) * len(MODEL_LIST)} models")

    all_train_prob_dfs = []
    all_test_prob_dfs = []

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

    final_train_prob_df = pd.concat(all_train_prob_dfs, axis=1)
    final_test_prob_df = pd.concat(all_test_prob_dfs, axis=1)

    sample_y = pd.read_csv(os.path.join(train_dir, all_files[0]), index_col=0)['y']
    final_train_prob_df['y'] = sample_y
    test_sample_y = pd.read_csv(os.path.join(SPLIT_DATA_PATH, "test",
                    os.listdir(os.path.join(SPLIT_DATA_PATH, "test"))[0]), index_col=0)['y']
    final_test_prob_df['y'] = test_sample_y

    final_train_prob_df, final_test_prob_df = remove_constant_features(
        final_train_prob_df, final_test_prob_df, dropna=False)

    final_train_prob_df.to_csv(os.path.join(PROB_OUTPUT_PATH, f"train_prob_{final_train_prob_df.shape}.csv"))
    final_test_prob_df.to_csv(os.path.join(PROB_OUTPUT_PATH, f"test_prob_{final_test_prob_df.shape}.csv"))
    print("\n" + "="*40)
    print(f"complete!")
    print(f"prob output train feature dimension: {final_train_prob_df.shape}")
    print(f"prob output test feature dimension: {final_test_prob_df.shape}")
    print(f"file already store in: {PROB_OUTPUT_PATH}")
    print("="*40)
