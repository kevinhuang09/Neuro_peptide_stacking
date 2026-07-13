"""
diagnosis table without only unique parameter that after faeture encoding
but before normalization  
"""
import pandas as pd

df_train = pd.read_csv('../data/featureStat/train_NeuroPeptide_nonormal.csv', index_col=0)
df_test = pd.read_csv('../data/featureStat/test_NeuroPeptide_nonormal.csv', index_col=0)

print(df_train.shape)   # (sample, feature)
print(df_train.head())
print(df_test.shape)   # (sample, feature)
print(df_test.head())

nunique_train = df_train.nunique(dropna = False)
# dropna = False; nan is also regard as a type
constant_cols_train = nunique_train[nunique_train <= 1].index.tolist()
nunique_test = df_test.nunique(dropna = False)
constant_cols_test = nunique_test[nunique_test <= 1].index.tolist()

print(f"共 {df_train.shape[1]} 個特徵欄位")
print(f"其中只有一種值的欄位有 {len(constant_cols_train)} 個：")
print(constant_cols_train)
print(f"共 {df_test.shape[1]} 個特徵欄位")
print(f"其中只有一種值的欄位有 {len(constant_cols_test)} 個：")
print(constant_cols_test)

df_clean_train = df_train.drop(columns=constant_cols_train)
print(f"移除後剩下 {df_clean_train.shape[1]} 個特徵")
df_clean_test = df_test.drop(columns=constant_cols_test)
print(f"移除後剩下 {df_clean_test.shape[1]} 個特徵")

df_clean_train.to_csv('../data/featureStat/train_Neuro_nonstd_nonunique.csv')
df_clean_test.to_csv('../data/featureStat/test_Neuro_nonstd_nonunique.csv')
print("已存檔")
