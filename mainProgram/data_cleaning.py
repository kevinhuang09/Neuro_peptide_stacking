import pandas as pd
from typing import Union, Tuple, Optional

def remove_constant_features(
    df_train : pd.DataFrame, 
    df_test : Optional[pd.DataFrame] = None,
    dropna : bool = False
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    移除 DataFrame 中只有單一獨特值（常數）的特徵欄位。
    
    Parameters:
    -----------
    df_train : pd.DataFrame
        訓練集資料。
    df_test : pd.DataFrame, optional
        測試集資料（選填）。若提供，將依據訓練集篩選後的特徵欄位同步進行過濾，以確保特徵一致。
    dropna : bool, default False
        計算獨特值個數時，是否將 NaN 視為一種獨立的值。
        若為 False，NaN 會被算作一種值（例如 [1, NaN] 會被算作 2 個 unique 值）；
        若為 True，NaN 不會被算作值（例如 [1, NaN] 只會被算作 1 個 unique 值，從而被判定為常數欄位並移除）。
        
    Returns:
    --------
    Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
        若只輸入 df_train，返回過濾後的 df_train。
        若同時輸入 df_train 與 df_test，返回 (過濾後的 df_train, 過濾後的 df_test)。
    """
    print(f"start to exceute dropna")
    # find all single value from train set
    nunique_train = df_train.nunique_(dropna = dropna)
    constant_cols = nunique_train[nunique_train <= 1].index.tolist()
    
    print(f"訓練集原始特徵數: {df_train.shape[1]}")
    print(f"偵測到常數特徵欄位共 {len(constant_cols)} 個：")
    if len(constant_cols) > 0:
        print(f" └─ {constant_cols}")
    else:
        print(" └─ 無")

    # 3. remove all unique value
    df_train_clean = df_train.drop(columns=constant_cols)
    print(f"訓練集處理後剩餘特徵數: {df_train_clean.shape[1]}")

    # 4. if have indp set, synchronous filtering
    if df_test is not None:
        print(f"\n測試集原始特徵數: {df_test.shape[1]}")
        
        # 確保只留下與處理後訓練集相同的欄位
        remaining_cols = df_train_clean.columns
        df_test_clean = df_test[remaining_cols]
        
        print(f"測試集與訓練集對齊後剩餘特徵數: {df_test_clean.shape[1]}")
        print("--- 執行完畢 ---")
        return df_train_clean, df_test_clean
    print("--- 執行完畢 ---")
    return df_train_clean