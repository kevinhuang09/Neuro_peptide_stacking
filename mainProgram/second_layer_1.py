import pandas as pd
import os

# # 1. 設定基礎路徑與新資料夾路徑
# mlDataPath = "../data/mlData/"
# # 建立子資料夾：feature_types_split
# splitDataPath = os.path.join(mlDataPath, "feature_types_split")

# if not os.path.exists(splitDataPath):
#     os.makedirs(splitDataPath)
#     print(f"✅ 已建立新資料夾: {splitDataPath}")

# source_file = os.path.join(mlDataPath, "all_features_for_boruta.csv")

# # 2. 讀取總表
# print(f"正在讀取總表: {source_file}")
# df = pd.read_csv(source_file)

# # 排除標籤列，取得所有特徵名稱
# all_features = [col for col in df.columns if col != 'Target_Label']

# # 3. 識別特徵類型 (AAC, CTDC...)
# feature_types = set([col.rsplit('_', 1)[0] for col in all_features])
# print(f"🔍 偵測到 {len(feature_types)} 種特徵類型")

# # 4. 開始拆分並存入子資料夾
# for f_type in feature_types:
#     related_cols = [col for col in all_features if col.startswith(f_type + "_")]
    
#     # 建立該類型的 DataFrame 並帶上 Label
#     sub_df = df[related_cols + ['Target_Label']]
#     sub_df = sub_df.rename(columns={'Target_Label': 'y'})
    
#     # --- 關鍵修改：存到 splitDataPath ---
#     output_name = os.path.join(splitDataPath, f"{f_type}_train.csv")
#     sub_df.to_csv(output_name, index=False)
#     print(f" [OK] 已存入: {splitDataPath}/{f_type}_train.csv")

# print("\n特徵拆分完成！")

# comfirm the number of files
# if want to check number of files can only use following code
mlDataPath = "../data/mlData/"
splitDataPath = os.path.join(mlDataPath, "feature_types_split")
csv_files = [f for f in os.listdir(splitDataPath) if f.endswith('.csv')]
print(f"have number of {len(csv_files)} files")