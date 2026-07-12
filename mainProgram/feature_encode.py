# set Agg to backend and without popping up a window
import matplotlib
matplotlib.use("Agg")

# set path 
import sys, os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent = os.path.join(current_dir, "..")
sys.path.append(parent)

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
    dataName = 'NeuroPeptide'

    ldObj = LoadDataset(minSeqLength=5)
    
    print("read fasta file ...")
    # trainNeg = ldObj.readFasta("../data/neg_train.fasta")
    # trainPos = ldObj.readFasta("../data/pos_train.fasta")
    # indpNeg = ldObj.readFasta("../data/neg_test.fasta")
    # indpPos = ldObj.readFasta("../data/pos_test.fasta")

    # hemoPI
    trainNeg = ldObj.readFasta("../data/neg2_train.fasta")
    trainPos = ldObj.readFasta("../data/pos_train.fasta")
    indpNeg = ldObj.readFasta("../data/neg2_test.fasta")
    indpPos = ldObj.readFasta("../data/pos_test.fasta")

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

    type1 = "nonormal"
    # save before normal data
    delNanTrainDf.to_csv(f'../data/featureStat/train_{dataName}_{type1}.csv')
    delNanIndpDf.to_csv(f'../data/featureStat/indp_{dataName}_{type1}.csv')
    print(f"save {type1} data")

    # Normalization
    normal = "standard"
    nmlzScalerPath = paramPath + f'{dataName}_standardScaler.pkl'
    trainNmlzDf = EncodeAllFeatures.dataNormalization(
        encodeTrainDf=delNanTrainDf,
        normalization=normal,
        saveNmlzScalerPklPath=nmlzScalerPath,
        b_loadPkl=False
    )
    indpNmlzDf1 = EncodeAllFeatures.dataNormalization(
        encodeIndpDf=delNanIndpDf,
        normalization=normal,
        loadNmlzScalerPklPath=nmlzScalerPath,
        b_loadPkl=True
    )

    # # remove acc_ridge
    # trainNmlzDf = trainNmlzDf.drop(columns=['aac_ridge'])
    # indpNmlzDf1 = indpNmlzDf1.drop(columns=['aac_ridge'])
    # store result
    trainNmlzDf.to_csv(f'../data/featureStat/train_{dataName}_{normal}.csv')
    indpNmlzDf1.to_csv(f'../data/featureStat/indp_{dataName}_{normal}.csv')

    print(f"feature encoding is complete!!!")