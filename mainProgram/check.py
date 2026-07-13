import pandas as pd

# 讀取你的總表
df = pd.read_csv("../data/featureStat/train_Neuro_standard.csv", nrows=1)

# 看看所有欄位的開頭
all_prefixes = set([col.split('_')[0] for col in df.columns])
print(f"總表中的特徵類別前綴有：\n{all_prefixes}")
print(f"總共欄位數：{len(df.columns)}")