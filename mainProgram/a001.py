import ast

# 定義檔案路徑
file_path = 'a001.txt'

try:
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        
        # 使用 ast.literal_eval 安全地將字串轉為 list
        # 這會處理掉引號和逗號
        feature_list = ast.literal_eval(f"[{content}]") if not content.startswith('[') else ast.literal_eval(content)

    print(f"成功讀取！特徵總數：{len(feature_list)}")
    print(f"前五個特徵：{feature_list[:5]}")

except FileNotFoundError:
    print("找不到 a001.txt，請檢查檔案路徑是否正確。")
except Exception as e:
    print(f"讀取失敗：{e}")