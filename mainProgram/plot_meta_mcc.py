# plot_meta_mcc.py
# 讀 meta_feature_gent.py 產生的 test_prob_*.csv，反推每個 feature_type × model 的 MCC，
# 畫出 4 張圖：(1) heatmap (2) MCC 分布直方圖 (3) 每個 feature type 最佳 model 排名長條圖
#             (4) 每個 feature type 的 MCC box plot（依中位數排序，1.5×IQR）
import os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import matthews_corrcoef
import config

plt.rcParams['axes.unicode_minus'] = False

# ========== 1. 定位 test_prob 檔案 ==========
dataName   = config.DATA_NAME
process1_t = config.process1_t
process2_t = config.process2_t
PROB_PATH  = f"../data/mlData/split_features_{dataName}_{process1_t}_{process2_t}/"
OUT_DIR    = "../data/mlScore/picture"
os.makedirs(OUT_DIR, exist_ok=True)

cands = glob.glob(os.path.join(PROB_PATH, "test_prob_*.csv"))
if not cands:
    raise FileNotFoundError(f"找不到 test_prob_*.csv，請確認路徑：{PROB_PATH}")
test_file = max(cands, key=os.path.getmtime)
print(f"讀取：{test_file}")

df = pd.read_csv(test_file, index_col=0)

# ========== 2. 分離 y 與機率欄位 ==========
if 'y' not in df.columns:
    raise KeyError("找不到標籤欄 'y'")
y_true = df['y'].astype(int)
prob_cols = [c for c in df.columns if c != 'y']

# ========== 3. 反推每個 feature_type × model 的 MCC ==========
records = []
for col in prob_cols:
    if '_' not in col:
        continue
    f_type, model = col.rsplit('_', 1)
    y_pred = (df[col].values >= 0.5).astype(int)
    mcc = matthews_corrcoef(y_true, y_pred)
    records.append({'feature_type': f_type, 'model': model, 'mcc': mcc})

mcc_long = pd.DataFrame(records)
mcc_long.to_csv(os.path.join(OUT_DIR, "meta_mcc_long.csv"), index=False)
print(f"共 {len(mcc_long)} 個 feature×model 組合，MCC 明細已存 meta_mcc_long.csv")

mcc_matrix = mcc_long.pivot(index='feature_type', columns='model', values='mcc')
mcc_matrix.to_csv(os.path.join(OUT_DIR, "meta_mcc_matrix.csv"))

# ========== 圖 1：Heatmap（不變） ==========
plt.figure(figsize=(max(10, 0.7*mcc_matrix.shape[1]),
                     max(8, 0.5*mcc_matrix.shape[0])))
sns.heatmap(mcc_matrix, annot=True, fmt=".2f", cmap="viridis",
            linewidths=.5, cbar_kws={'label': 'MCC'})
plt.title(f"MCC Heatmap (Feature Type x Model, Test set) - {dataName}", fontsize=14)
plt.xlabel("Model"); plt.ylabel("Feature Type")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "meta_mcc_heatmap.png"), dpi=300)
plt.close()

# ========== 圖 2：MCC 分布直方圖（不變） ==========
mcc_values = mcc_long['mcc'].values
mean_mcc   = mcc_values.mean()

lo = min(0.0, np.floor(mcc_values.min() * 10) / 10)
hi = max(1.0, np.ceil (mcc_values.max() * 10) / 10)
bins = np.arange(lo, hi + 0.1, 0.1)

plt.figure(figsize=(12, 7))
sns.set_style("whitegrid")
counts, edges, patches = plt.hist(
    mcc_values, bins=bins,
    color="skyblue", edgecolor="black", linewidth=1.2
)
for cnt, left, right in zip(counts, edges[:-1], edges[1:]):
    if cnt > 0:
        plt.text((left + right) / 2, cnt + max(counts) * 0.01,
                 f"{int(cnt)}", ha='center', va='bottom', fontsize=11)
plt.axvline(mean_mcc, color='red', linestyle='--', linewidth=2,
            label=f"Mean: {mean_mcc:.3f}")
plt.title("MCC Distribution of All Model Combinations (bin = 0.1)", fontsize=16)
plt.xlabel("Test MCC Bins", fontsize=13)
plt.ylabel("Number of Models", fontsize=13)
plt.xticks(bins)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "meta_mcc_histogram.png"), dpi=300)
plt.close()

# ========== 圖 3：每個 feature type 最佳 model MCC 排名（不變） ==========
best = (mcc_long.loc[mcc_long.groupby('feature_type')['mcc'].idxmax()]
                .sort_values('mcc', ascending=False))
plt.figure(figsize=(12, max(6, 0.5*len(best))))
ax = sns.barplot(data=best, x='mcc', y='feature_type', palette='magma')
for i, row in enumerate(best.itertuples(index=False)):
    ax.text(row.mcc + 0.005, i, f"{row.mcc:.3f} ({row.model})",
            va='center', fontsize=9)
plt.title(f"Best MCC per Feature Type (Test set) - {dataName}", fontsize=14)
plt.xlabel("MCC (best model)"); plt.ylabel("Feature Type")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "meta_mcc_best_bar.png"), dpi=300)
plt.close()

# ========== 圖 4：每個 feature type 的 MCC Box Plot（★新增） ==========
# 依「中位數 MCC」由高到低排序 feature type
order = (mcc_long.groupby('feature_type')['mcc']
                 .median()
                 .sort_values(ascending=False)
                 .index.tolist())

plt.figure(figsize=(max(12, 0.7*len(order)), 8))
sns.set_style("whitegrid")
ax = sns.boxplot(
    data=mcc_long, x='feature_type', y='mcc',
    order=order,
    whis=1.5,                 # ← 1.5 倍 IQR 畫鬍鬚（超過的點視為離群值）
    palette="Set2",
    showfliers=True,          # 顯示離群點
    fliersize=4,
    linewidth=1.2
)
# 疊上實際資料點，看每個 model 落在哪
sns.stripplot(
    data=mcc_long, x='feature_type', y='mcc',
    order=order, color='black', size=3, alpha=0.4, jitter=0.2, ax=ax
)
plt.title(f"MCC Distribution per Feature Type (Boxplot, sorted by median, whisker=1.5xIQR) - {dataName}",
          fontsize=14)
plt.xlabel("Feature Type", fontsize=13)
plt.ylabel("Test MCC", fontsize=13)
plt.xticks(rotation=45, ha='right')
plt.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.6)  # MCC=0 參考線
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "meta_mcc_boxplot.png"), dpi=300)
plt.close()

print("\n完成！四張圖已存到:", OUT_DIR)
print(" 1) meta_mcc_heatmap.png")
print(" 2) meta_mcc_histogram.png")
print(" 3) meta_mcc_best_bar.png")
print(" 4) meta_mcc_boxplot.png   ← 新增的 box plot")
