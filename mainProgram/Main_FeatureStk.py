# ==========================================
# import environment setting
# ==========================================
import init_env, os, sys
import config
import pandas as pd
from concurrent.futures import ProcessPoolExecutor


# import some package from userPackage (encode, feature)
from userPackage.Package_Encode import EncodeAllFeatures
from userPackage.LoadDataset import LoadDataset
from userPackage.FeatureStat import FeatureStat
from data_cleaning import remove_constant_features

featureDict = FeatureStat.build_feature_config()


# import some package from userPackage (encode, feature)
from userPackage.Package_Encode import EncodeAllFeatures
from userPackage.LoadDataset import LoadDataset
from userPackage.FeatureStat import FeatureStat
from data_cleaning import remove_constant_features

featureDict = FeatureStat.build_feature_config()

# ==========================================
# 可調整區：想只跑編碼或只跑切分時修改這裡
# ==========================================
RUN_ENCODE = True   # 對應原 feature_encode.py 的 __main__
RUN_SPLIT = True    # 對應原 feature_split.py 的 __main__

# path set
paramPath = "../data/param/"
mlDataPath = "../data/mlData/"
featureStatPath = "../data/featureStat/"

dataName = config.DATA_NAME
process1_f = config.process1_f
process1_t = config.process1_t
process2_f = config.process2_f
process2_t = config.process2_t

NORMALIZE_METHOD = "standard"


# ==========================================
# 對應 feature_encode.py：平行編碼 worker
# ==========================================
def encode_worker(neg_seqs, pos_seqs, feat_dict, json_path):
    # Notice!!! dict should to add -1, avoid keyerror
    worker_data_dict = {
        0: neg_seqs,
        1: pos_seqs,
        -1: None  # conform to EncodeAllFeatures
    }
    encoder = EncodeAllFeatures()
    encoder.dataEncodeSetup(
        saveFeatureDict=feat_dict,
        saveJsonPath=json_path,
        b_loadJson=False
    )
    return encoder.dataEncodeOutPut(dataDict=worker_data_dict)


def clean(d):
    """clean peptide remove '-'"""
    return {k: v.replace('-', '').upper() for k, v in d.items() if v}


# ==========================================
# 對應 feature_encode.py 的 __main__
# ==========================================
def run_feature_encode():
    os.makedirs(paramPath, exist_ok=True)
    os.makedirs(mlDataPath, exist_ok=True)
    os.makedirs(featureStatPath, exist_ok=True)

    ldObj = LoadDataset(minSeqLength=5)
    print("read fasta file ...")

    # NeuroPeptide
    trainNeg = ldObj.readFasta("../data/neg2_train.fasta")
    trainPos = ldObj.readFasta("../data/pos_train.fasta")
    indpNeg = ldObj.readFasta("../data/neg2_test.fasta")
    indpPos = ldObj.readFasta("../data/pos_test.fasta")

    tNeg, tPos = clean(trainNeg), clean(trainPos)
    iNeg, iPos = clean(indpNeg), clean(indpPos)

    print("Parallel feature extraction")
    with ProcessPoolExecutor(max_workers=12) as executor:
        f_train = executor.submit(
            encode_worker, tNeg, tPos, featureDict,
            paramPath + f'{dataName}_train_feat.json'
        )
        f_indp = executor.submit(
            encode_worker, iNeg, iPos, featureDict,
            paramPath + f'{dataName}_indp_feat.json'
        )
        try:
            encodeTrainDf = f_train.result()
            print("training encoding is complete")
            encodeIndpDf1 = f_indp.result()
            print("test encoding is complete")
        except Exception as e:
            print(f"error message: {e}")
            sys.exit(1)


# ==========================================
# 對應 feature_encode.py 的 __main__
# ==========================================
def run_feature_encode():
    os.makedirs(paramPath, exist_ok=True)
    os.makedirs(mlDataPath, exist_ok=True)
    os.makedirs(featureStatPath, exist_ok=True)

    ldObj = LoadDataset(minSeqLength=5)
    print("read fasta file ...")

    # NeuroPeptide
    trainNeg = ldObj.readFasta("../data/neg2_train.fasta")
    trainPos = ldObj.readFasta("../data/pos_train.fasta")
    indpNeg = ldObj.readFasta("../data/neg2_test.fasta")
    indpPos = ldObj.readFasta("../data/pos_test.fasta")

    tNeg, tPos = clean(trainNeg), clean(trainPos)
    iNeg, iPos = clean(indpNeg), clean(indpPos)

    print("Parallel feature extraction")
    with ProcessPoolExecutor(max_workers=12) as executor:
        f_train = executor.submit(
            encode_worker, tNeg, tPos, featureDict,
            paramPath + f'{dataName}_train_feat.json'
        )
        f_indp = executor.submit(
            encode_worker, iNeg, iPos, featureDict,
            paramPath + f'{dataName}_indp_feat.json'
        )
        try:
            encodeTrainDf = f_train.result()
            print("training encoding is complete")
            encodeIndpDf1 = f_indp.result()
            print("test encoding is complete")
        except Exception as e:
            print(f"error message: {e}")
            sys.exit(1)

    # ==========================================
    # Executing Normalization and Analysis
    # ==========================================
    print("Executing Normalization and Analysis")

    delNanTrainDf = FeatureStat.delNan(data=encodeTrainDf, logPath=mlDataPath + "delNanTrain.txt")
    delNanIndpDf = FeatureStat.delNan(data=encodeIndpDf1, logPath=mlDataPath + "delNanIndp.txt")

    delNanTrainDf.to_csv(f'{featureStatPath}train_{dataName}_{process1_f}_{process2_f}.csv')
    delNanIndpDf.to_csv(f'{featureStatPath}indp_{dataName}_{process1_f}_{process2_f}.csv')
    print(f"save {process1_f} data")

    # diagnosis table without only unique parameter,
    # after feature encoding but before normalization
    df_train = pd.read_csv(
        f'{featureStatPath}train_{dataName}_{process1_f}_{process2_f}.csv', index_col=0
    )
    df_test = pd.read_csv(
        f'{featureStatPath}indp_{dataName}_{process1_f}_{process2_f}.csv', index_col=0
    )
    df_clean_train, df_clean_test = remove_constant_features(df_train, df_test, dropna=False)

    df_clean_train.to_csv(f'{featureStatPath}train_{dataName}_{process1_t}_{process2_f}.csv')
    df_clean_test.to_csv(f'{featureStatPath}indp_{dataName}_{process1_t}_{process2_f}.csv')
    print("drop unique file already saved")

    # Normalization
    normal = "standard"
    nmlzScalerPath = paramPath + f'{dataName}_standardScaler.pkl'
    trainNmlzDf = EncodeAllFeatures.dataNormalization(
        encodeTrainDf=df_clean_train,
        normalization=normal,
        saveNmlzScalerPklPath=nmlzScalerPath,
        b_loadPkl=False
    )
    indpNmlzDf1 = EncodeAllFeatures.dataNormalization(
        encodeIndpDf=df_clean_test,
        normalization=normal,
        loadNmlzScalerPklPath=nmlzScalerPath,
        b_loadPkl=True
    )

    train_csv_path = f'{featureStatPath}train_{dataName}_{process1_t}_{process2_t}.csv'
    indp_csv_path = f'{featureStatPath}indp_{dataName}_{process1_t}_{process2_t}.csv'
    trainNmlzDf.to_csv(train_csv_path)
    indpNmlzDf1.to_csv(indp_csv_path)

    print("feature encoding is complete!!!")
    return train_csv_path, indp_csv_path


# ==========================================
# 對應 feature_split.py：切分單一 prefix 的 worker
# ==========================================
def split_task(f_prefix, train_df, test_df, output_base_path):
    """split from summary table"""
    # exclude y and index
    if f_prefix in ['y', 'Unnamed: 0']:
        return None

    results = []
    for df, set_name in [(train_df, 'train'), (test_df, 'test')]:
        cols = [c for c in df.columns if c.startswith(f_prefix)]
        if not cols:
            continue

        if 'y' in df.columns:
            label = 'y'
        else:
            # 修正原始 bug：找不到 'y' 欄位時 fallback 用最後一欄當 label
            label = df.columns[-1]
            print(f"'y' column not found, auto fallback to '{label}' as label")

        extracted_df = df[cols + [label]]

        save_dir = os.path.join(output_base_path, set_name)
        os.makedirs(save_dir, exist_ok=True)

        save_file = os.path.join(save_dir, f"{f_prefix.lower()}_{set_name}.csv")
        extracted_df.to_csv(save_file)
        results.append(f"{f_prefix}_{set_name}")

    return results


# ==========================================
# 對應 feature_split.py 的 __main__
# ==========================================
def run_feature_split(train_csv_path=None, indp_csv_path=None):
    # 若未傳入路徑（例如單獨跑 split），沿用 config 規則自行拼出路徑
    if train_csv_path is None:
        train_csv_path = os.path.join(
            featureStatPath, f"train_{dataName}_{process1_t}_{process2_t}.csv"
        )
    if indp_csv_path is None:
        indp_csv_path = os.path.join(
            featureStatPath, f"indp_{dataName}_{process1_t}_{process2_t}.csv"
        )

    output_base_path = f"{featureStatPath}split_features_{dataName}_{process1_t}_{process2_t}/"

    print(f"Loading summary table (Data: {dataName})...")
    try:
        full_train = pd.read_csv(train_csv_path, index_col=0)
        full_test = pd.read_csv(indp_csv_path, index_col=0)
    except FileNotFoundError:
        print(f"not found summary table file, please ensure output directory exists : {featureStatPath}")
        print(f"expect file name : train_{dataName}_{process1_t}_{process2_t}.csv")
        sys.exit(1)

    print("auto analysis feature")
    actual_prefixes = set()
    for col in full_train.columns:
        if col not in ['y', 'Unnamed: 0']:
            prefix = col.split('_')[0]
            actual_prefixes.add(prefix)
    prefix_list = sorted(list(actual_prefixes))
    print(f"detected {len(prefix_list)} feature categories: {prefix_list[:10]} ...")

    os.makedirs(output_base_path, exist_ok=True)
    info_file_path = os.path.join(output_base_path, "feature_info.txt")
    try:
        with open(info_file_path, "w", encoding="utf-8") as f:
            f.write(f"Data Name: {dataName}\n")
            f.write(f"Normalize Method: {NORMALIZE_METHOD}\n")
            f.write(f"Total Feature Categories: {len(prefix_list)}\n")
            f.write("-" * 30 + "\n")
            f.write("Feature List:\n")
            for pref in prefix_list:
                f.write(f"- {pref}\n")
        print(f"Feature list has been saved to: {info_file_path}")
    except Exception as e:
        print(f"Failed to write feature_info.txt: {e}")

    print("starting concurrent splitting")
    with ProcessPoolExecutor(max_workers=12) as executor:
        futures = [
            executor.submit(split_task, pref, full_train, full_test, output_base_path)
            for pref in prefix_list
        ]
        done_count = 0
        for future in futures:
            try:
                res = future.result()
                if res:
                    done_count += 1
            except Exception as e:
                print(f"error occurred during the splitting process: {e}")

    print("\n" + "=" * 40)
    print("split success!!!")
    print(f"have {done_count} feature categories")
    print(f"file put in: {output_base_path}")
    print("=" * 40)


# ==========================================
# Main Process：唯一的進入點，取代原本兩份各自的 __main__
# ==========================================
if __name__ == "__main__":
    train_csv_path = None
    indp_csv_path = None

    if RUN_ENCODE:
        train_csv_path, indp_csv_path = run_feature_encode()

    if RUN_SPLIT:
        run_feature_split(train_csv_path, indp_csv_path)