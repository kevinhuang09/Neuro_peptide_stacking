import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import matthews_corrcoef

# 1. 設定讀取路徑與輸出路徑
data_file = "../data/mlData/train_F270.csv"
output_dir = "../data/mlScore/"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 2. 讀取並清理資料
df = pd.read_csv(data_file)

# 根據截圖，排除第一欄 ID 與標籤 y
y = df['y']
# .iloc[:, 1:] 跳過第一欄字串，drop('y') 移除標籤
X_all = df.drop(columns=['y']).iloc[:, 1:] 

# 3. 自動辨識特徵類別 (例如 z5, CTDD, PAAC)
all_features = X_all.columns.tolist()
feature_groups = {}
for col in all_features:
    # 取得底線前的名稱作為類型
    prefix = col.split('_')[0]
    if prefix not in feature_groups:
        feature_groups[prefix] = []
    feature_groups[prefix].append(col)

# 4. 循環測試每一組特徵的 MCC
results = []
scaler = StandardScaler()

print(f"開始評估 {len(feature_groups)} 種特徵類型...")

for f_type, cols in feature_groups.items():
    X_subset = X_all[cols]
    
    # 標準化並使用 5-Fold 交叉驗證
    X_scaled = scaler.fit_transform(X_subset)
    model = LogisticRegression(max_iter=1000)
    
    # 取得預測結果並計算 MCC
    y_pred = cross_val_predict(model, X_scaled, y, cv=5)
    mcc_score = matthews_corrcoef(y, y_pred)
    
    results.append({
        "Feature_Type": f_type,
        "MCC": mcc_score,
        "Feature_Count": len(cols)
    })

# 5. 整理結果並排序
mcc_df = pd.DataFrame(results).sort_values(by="MCC", ascending=False)

# 6. 繪圖 (使用橫向條形圖)
plt.figure(figsize=(12, 8))
sns.set_style("whitegrid")
ax = sns.barplot(x="MCC", y="Feature_Type", data=mcc_df, palette="magma")

# 標註具體數值與特徵數量
for i, row in enumerate(mcc_df.itertuples(index=False)):
    # row[1] 是 MCC, row[2] 是 Feature_Count
    ax.text(row[1] + 0.005, i, f"MCC: {row[1]:.3f} (n={row[2]})", va='center', fontsize=10)

plt.title("MCC Performance by Feature Type (5-Fold CV)", fontsize=16)
plt.xlabel("Matthews Correlation Coefficient (MCC)", fontsize=12)
plt.ylabel("Feature Groups", fontsize=12)
plt.tight_layout()

# 7. 儲存
plt.savefig(os.path.join(output_dir, "feature_type_mcc_plot.png"), dpi=300)
mcc_df.to_csv(os.path.join(output_dir, "feature_type_mcc_table.csv"), index=False)

print("\n分析完成！")
print(f"圖表儲存於: {output_dir}feature_type_mcc_plot.png")
print(f"表格儲存於: {output_dir}feature_type_mcc_table.csv")
print(mcc_df)