import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 1. 準備模擬數據
models = ['LightGBM', 'XGBoost', 'NB', 'ET', 'RF', 'QDA']
bins = ['0.0-0.2', '0.2-0.6', '0.6-1.0']

# 模擬 Sequence Number (柱狀圖數據)
data_num = {
    'LightGBM': [680, 80, 250],
    'XGBoost': [690, 70, 260],
    'NB': [750, 20, 280],
    'ET': [550, 220, 230],
    'RF': [450, 320, 240],
    'QDA': [700, 40, 300]
}

# 模擬 TPR (折線圖數據)
data_tpr = {
    'LightGBM': [0.1, 0.6, 1.8],
    'XGBoost': [0.15, 0.55, 1.6],
    'NB': [0.4, 0.3, 1.1],
    'ET': [0.05, 0.7, 2.4],
    'RF': [0.1, 0.8, 2.8],
    'QDA': [0.3, 0.3, 1.4]
}

# 2. 設定顏色 (參考原圖)
colors = ['#42d4f4', '#13f194', '#9b66ff', '#ff6699', '#ffcc00', '#ffa066']
line_colors = ['#f39c12', '#3498db', '#8e44ad', '#2ecc71', '#c0392b', '#ff0000']

# 3. 開始畫圖
fig, ax1 = plt.subplots(figsize=(10, 6))
ax2 = ax1.twinx()  # 建立第二個 Y 軸

x = np.arange(len(bins))
width = 0.12  # 柱子寬度

# 畫柱狀圖 (Sequence Number)
for i, model in enumerate(models):
    ax1.bar(x + (i - 2.5) * width, data_num[model], width, label=f'{model}_Num', color=colors[i])

# 畫折線圖 (TPR)
for i, model in enumerate(models):
    ax2.plot(x, data_tpr[model], marker='o', label=f'{model}_TPR', color=line_colors[i], linewidth=2)

# 4. 圖表細節設定
ax1.set_xlabel('Prediction Probability', fontsize=12)
ax1.set_ylabel('Sequence Number', fontsize=12)
ax2.set_ylabel('True Positive Rate', fontsize=12, rotation=270, labelpad=20)

# 設定 X 軸刻度
ax1.set_xticks(x)
ax1.set_xticklabels(bins)

# 設定 Y 軸範圍
ax1.set_ylim(0, 880)
ax1.set_yticks([0, 440, 880])
ax2.set_ylim(0, 3)
ax2.set_yticks([0, 1.5, 3])

# 合併圖例並放在上方
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center', 
           bbox_to_anchor=(0.5, 1.15), ncol=4, frameon=False, fontsize=9)

plt.tight_layout()
plt.show()