def count_fasta_sequences(filename):
    """計算單一 FASTA 檔案中的序列數量"""
    count = 0
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                # 判斷行首是否為 '>'
                if line.startswith('>'):
                    count += 1
        return count
    except FileNotFoundError:
        return None  # 找不到檔案時返回 None，方便後面做邏輯判斷


# ==================== 設定你要讀取的四個檔案 ====================
file_list = [
    "neg2_train.fasta", 
    "pos_train.fasta", 
    "neg2_test.fasta", 
    "pos_test.fasta"
]
# ============================================================

total_sequences = 0

print("--- FASTA 序列計算結果 ---")
for file_path in file_list:
    result = count_fasta_sequences(file_path)
    
    if result is not None:
        print(f"📄 檔案 [{file_path}]: 共有 {result} 條序列")
        total_sequences += result
    else:
        print(f"❌ 檔案 [{file_path}]: 找不到檔案，請檢查路徑是否正確。")

print("------------------------")
print(f"📊 所有檔案總計共有: {total_sequences} 條序列")