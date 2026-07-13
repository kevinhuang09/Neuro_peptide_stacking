"""
diagnosis table without only unique parameter that after faeture encoding
but before normalization  
"""
import pandas as pd

df = pd.read_csv('../data/featureStat/train_NeuroPeptide_standard.csv', index_col=0)
print(df.shape)   # (sample, feature)
print(df.head())

nunique = df.nunique(dropna=False)
# dropna = False; nan is also regard as a type
constant_cols = nunique[nunique <= 1].index.tolist()

print(f"共 {df.shape[1]} 個特徵欄位")
print(f"其中只有一種值的欄位有 {len(constant_cols)} 個：")
print(constant_cols)

df_clean = df.drop(columns=constant_cols)
print(f"移除後剩下 {df_clean.shape[1]} 個特徵")

df_clean.to_csv('../data/featureStat/train_Neuro_std_nonunique.csv')
print("已存檔")
