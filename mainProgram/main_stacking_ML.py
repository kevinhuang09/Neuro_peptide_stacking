# set path
import sys, os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent = os.path.join(current_dir, "..")
sys.path.append(parent)

# using package
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, matthews_corrcoef, confusion_matrix

# set data path
mlDataPath = "../data/mlData/"  
mlScorePath = "../data/mlScore/" 
finalModelPath = "../data/finalModel"
print(f"data path set OK")

# 1. 讀取與清理數據
dataTrainDf = pd.read_csv(mlDataPath + "train_F270.csv")

# 修正：排除第一欄 ID (字串) 與目標欄位 y
# 根據截圖，A 欄是字串 ID，我們從第二欄開始取特徵
X = dataTrainDf.drop(columns=['y']).iloc[:, 1:] 
y = dataTrainDf['y']

print(f"特徵數量確認: {X.shape[1]}")
print(f"前 3 個特徵名稱: {X.columns[:3].tolist()}")

# 2. 標準化 (Scaler)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 分割數據
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 3. 訓練模型
print("正在訓練 Logistic Regression...")
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)

print("正在訓練 MLP (Neural Network)...")
mlp = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', max_iter=500, random_state=42)
mlp.fit(X_train, y_train)

# 4. 取得預測值與計算分數
lr_preds = lr.predict(X_test)
mlp_preds = mlp.predict(X_test)
lr_probs = lr.predict_proba(X_test)[:, 1]
mlp_probs = mlp.predict_proba(X_test)[:, 1]

# 5. 整理結果表格 (包含 MCC)
results = {
    "Model": ["Logistic Regression", "MLP (Neural Network)"],
    "AUC": [roc_auc_score(y_test, lr_probs), roc_auc_score(y_test, mlp_probs)],
    "Accuracy": [accuracy_score(y_test, lr_preds), accuracy_score(y_test, mlp_preds)],
    "MCC": [matthews_corrcoef(y_test, lr_preds), matthews_corrcoef(y_test, mlp_preds)]
}

final_score_df = pd.DataFrame(results)
print("\n--- 訓練集驗證結果 ---")
print(final_score_df)

# 6. 儲存結果與模型
if not os.path.exists(mlScorePath): os.makedirs(mlScorePath)
final_score_df.to_csv(os.path.join(mlScorePath, "lr_mlp_comparison.csv"), index=False)

if not os.path.exists(finalModelPath): os.makedirs(finalModelPath)
joblib.dump(lr, os.path.join(finalModelPath, "lr_final.pkl"))
joblib.dump(mlp, os.path.join(finalModelPath, "mlp_final.pkl"))
joblib.dump(scaler, os.path.join(finalModelPath, "scaler.pkl"))
print(f"\n✅ 模型與 Scaler 已儲存至 {finalModelPath}")

# 7. 預測新資料 (test_F270.csv)
new_data_path = mlDataPath + "test_F270.csv"
if os.path.exists(new_data_path):
    new_df = pd.read_csv(new_data_path)
    
    # 修正：確保測試集的特徵與訓練集完全對齊 (排除 y 與 ID 欄位)
    if 'y' in new_df.columns:
        y_true = new_df['y']
        # 使用訓練時 X 的欄位清單來篩選，保證順序與數量一致
        X_new = new_df[X.columns] 
        
        X_new_scaled = scaler.transform(X_new) 
        
        new_lr_preds = lr.predict(X_new_scaled)
        new_mlp_preds = mlp.predict(X_new_scaled)
        
        print("\n--- 新資料測試結果 (New Samples) ---")
        print(f"LR MCC: {matthews_corrcoef(y_true, new_lr_preds):.4f}")
        print(f"MLP MCC: {matthews_corrcoef(y_true, new_mlp_preds):.4f}")
        print("LR Confusion Matrix:\n", confusion_matrix(y_true, new_lr_preds))
else:
    print(f"\n提示: 找不到 {new_data_path}，跳過新資料預測步驟。")