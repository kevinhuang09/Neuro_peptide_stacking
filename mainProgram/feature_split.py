import pandas as pd
import os
import sys
from concurrent.futures import ProcessPoolExecutor

# ==========================================
# set path
# ==========================================
DATA_NAME = 'NeuroPeptide' 
INPUT_PATH = "../data/featureStat/"
# OUTPUT_BASE_PATH = f"../data/featureStat/split_features_{DATA_NAME}/"
OUTPUT_BASE_PATH = f"../data/featureStat/split_features_{DATA_NAME}_nonormal/"
 
NORMALIZE_METHOD = 'standard'

# ==========================================
# 2. detect file name from prefix
# file_name format : ACC_train.csv
# ==========================================
def split_task(f_prefix, train_df, test_df):
    """
    split from summary table
    """
    # exclude y and index
    if f_prefix in ['y', 'Unnamed: 0']:
        return None
        
    results = []
    # process the training set and test set seperately
    for df, set_name in [(train_df, 'train'), (test_df, 'test')]:
        # find all of prefix file 
        cols = [c for c in df.columns if c.startswith(f_prefix)]
        
        # check 'y' column exists
        if cols: 
            if 'y' in df.columns:
                label = 'y'
            else:
                print(f"auto to convert last col to 'y'")
            extracted_df = df[cols + [label]]
            
            # check path if without path auto to build
            save_dir = os.path.join(OUTPUT_BASE_PATH, set_name)
            os.makedirs(save_dir, exist_ok=True)
            
            # file_name convert to lower case
            save_file = os.path.join(save_dir, f"{f_prefix.lower()}_{set_name}.csv")
            extracted_df.to_csv(save_file)
            results.append(f"{f_prefix}_{set_name}")
            
    return results

# ==========================================
# concurrent core process
# ==========================================
if __name__ == "__main__":
    # train_path = os.path.join(INPUT_PATH, f"train_{DATA_NAME}_{NORMALIZE_METHOD}.csv")
    # test_path = os.path.join(INPUT_PATH, f"indp_{DATA_NAME}_{NORMALIZE_METHOD}.csv")
    type1 = "nonormal"
    train_path = os.path.join(INPUT_PATH, f"train_{DATA_NAME}_{type1}.csv")
    test_path = os.path.join(INPUT_PATH, f"indp_{DATA_NAME}_{type1}.csv")


    print(f"Loading summary table (Data: {DATA_NAME})...")
    try:
        # read summary table
        full_train = pd.read_csv(train_path, index_col=0)
        full_test = pd.read_csv(test_path, index_col=0)
    except FileNotFoundError:
        print(f"not found summary table file, please ensure output directory exists : {INPUT_PATH}")
        print(f"except file name : train_{DATA_NAME}_{NORMALIZE_METHOD}.csv")
        sys.exit(1)


    print("auto analysis feature")
    actual_prefixes = set()
    for col in full_train.columns:
        if col not in ['y', 'Unnamed: 0']:
            prefix = col.split('_')[0]
            actual_prefixes.add(prefix)
    
    prefix_list = sorted(list(actual_prefixes))
    print(f"detected {len(prefix_list)} feature categories: {prefix_list[:10]} ...")

    # store feature message
    os.makedirs(OUTPUT_BASE_PATH, exist_ok = True)
    info_file_path = os.path.join(OUTPUT_BASE_PATH, "feature_info.txt")
    try:
        with open(info_file_path, "w", encoding="utf-8") as f:
            f.write(f"Data Name: {DATA_NAME}\n")
            f.write(f"Normalize Method: {NORMALIZE_METHOD}\n")
            f.write(f"Total Feature Categories: {len(prefix_list)}\n")
            f.write("-" * 30 + "\n")
            f.write("Feature List:\n")
            for pref in prefix_list:
                f.write(f"- {pref}\n")
        print(f"Feature list has been saved to: {info_file_path}")
    except Exception as e:
        print(f"Failed to write feature_info.txt: {e}")

    print(f"starting concurrent spliting")
    
    with ProcessPoolExecutor(max_workers=12) as executor:

        futures = [executor.submit(split_task, pref, full_train, full_test) for pref in prefix_list]
        
        done_count = 0
        for future in futures:
            try:
                res = future.result()
                if res:
                    done_count += 1
            except Exception as e:
                print(f" error occurred during the splitting process: {e}")

    # show message 
    print("\n" + "="*40)
    print(f"split suceess!!!")
    print(f"have {done_count} feature categories")
    print(f"file put in: {OUTPUT_BASE_PATH}")
    print("="*40)