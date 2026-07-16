# import environment setting
import init_env, os, sys

# import config parameters
import config

# set concurrent core
from concurrent.futures import ProcessPoolExecutor

# import some package from userPackage (encode, feature)
from userPackage.Package_Encode import EncodeAllFeatures
from userPackage.LoadDataset import LoadDataset
from userPackage.FeatureStat import FeatureStat

featureDict = FeatureStat.build_feature_config()

# set concurrent worker
def encode_worker(neg_seqs, pos_seqs, feat_dict, json_path):
    # Notice!!! dict should to add -1, avoid keyerror
    worker_data_dict = {
        0: neg_seqs, 
        1: pos_seqs, 
        -1: None  # conform to EncodeAllFeatures
    }

    # initialize the feature encoding class
    encoder = EncodeAllFeatures()
    encoder.dataEncodeSetup(
        saveFeatureDict=feat_dict,
        saveJsonPath=json_path,
        b_loadJson=False
    )
    
    return encoder.dataEncodeOutPut(dataDict=worker_data_dict)

# ==========================================
# Main Process
# ==========================================
if __name__ == "__main__": # set entry point
    paramPath = "../data/param/"
    mlDataPath = "../data/mlData/"
    os.makedirs(paramPath, exist_ok=True)
    os.makedirs(mlDataPath, exist_ok=True)
    dataName = config.DATA_NAME
    # unique
    process1_f = config.process1_f
    process1_t = config.process1_t
    # normal
    process2_f = config.process2_f
    process2_t = config.process2_t
    
    ldObj = LoadDataset(minSeqLength=5)
    
    print("read fasta file ...")
    # NeuroPeptide
    trainNeg = ldObj.readFasta("../data/neg2_train.fasta")
    trainPos = ldObj.readFasta("../data/pos_train.fasta")
    indpNeg = ldObj.readFasta("../data/neg2_test.fasta")
    indpPos = ldObj.readFasta("../data/pos_test.fasta")
    # 📄 檔案 [neg2_train.fasta]: 共有 2632 條序列
    # 📄 檔案 [pos_train.fasta]: 共有 2632 條序列
    # 📄 檔案 [neg2_test.fasta]: 共有 293 條序列
    # 📄 檔案 [pos_test.fasta]: 共有 293 條序列

    # clean peptide remove '-'
    def clean(d): return {k: v.replace('-', '').upper() for k, v in d.items() if v}
    tNeg, tPos = clean(trainNeg), clean(trainPos)
    iNeg, iPos = clean(indpNeg), clean(indpPos)

    print(f"Parallel feature extraction")
    with ProcessPoolExecutor(max_workers=12) as executor:
        # start the calculations for train and indp simultaneously
        f_train = executor.submit(encode_worker, tNeg, tPos, featureDict, paramPath + f'{dataName}_train_feat.json')
        f_indp = executor.submit(encode_worker, iNeg, iPos, featureDict, paramPath + f'{dataName}_indp_feat.json')
        
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
    print(f"Executing Normalization and Analysis")    
    # delete NaN
    delNanTrainDf = FeatureStat.delNan(data=encodeTrainDf, logPath=mlDataPath + "delNanTrain.txt")
    delNanIndpDf = FeatureStat.delNan(data=encodeIndpDf1, logPath=mlDataPath + "delNanIndp.txt")

    # save non normal data
    delNanTrainDf.to_csv(f'../data/featureStat/train_{dataName}_{process1_f}_{process2_f}.csv')
    delNanIndpDf.to_csv(f'../data/featureStat/indp_{dataName}_{process1_f}_{process2_f}.csv')
    print(f"save {process1_f} data")

    """
    diagnosis table without only unique parameter that after faeture encoding
    but before normalization  
    """
    import pandas as pd
    from data_cleaning import remove_constant_features

    df_train = pd.read_csv(f'../data/featureStat/train_{dataName}_{process1_f}_{process2_f}.csv', index_col=0)
    df_test = pd.read_csv(f'../data/featureStat/indp_{dataName}_{process1_f}_{process2_f}.csv', index_col=0)

    df_clean_train, df_clean_test = remove_constant_features(df_train, df_test, dropna = False)

    df_clean_train.to_csv(f'../data/featureStat/train_{dataName}_{process1_t}_{process2_f}.csv')
    df_clean_test.to_csv(f'../data/featureStat/indp_{dataName}_{process1_t}_{process2_f}.csv')
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

    # # remove acc_ridge
    # trainNmlzDf = trainNmlzDf.drop(columns=['aac_ridge'])
    # indpNmlzDf1 = indpNmlzDf1.drop(columns=['aac_ridge'])
    # store result

    trainNmlzDf.to_csv(f'../data/featureStat/train_{dataName}_{process1_t}_{process2_t}.csv')
    indpNmlzDf1.to_csv(f'../data/featureStat/indp_{dataName}_{process1_t}_{process2_t}.csv')

    print(f"feature encoding is complete!!!")