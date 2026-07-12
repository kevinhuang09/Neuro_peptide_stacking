import os
import random

def read_fasta_to_list(file_path):
    """讀取 FASTA 檔案並將每筆資料(Header+Sequence)轉成 list 中的一個元素"""
    records = []
    if not os.path.exists(file_path):
        print(f"警告: 找不到檔案 {file_path}")
        return []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        # 以 '>' 分割，但保留 '>'
        entries = content.split('>')
        for entry in entries:
            if entry.strip():
                records.append('>' + entry.strip() + '\n')
    return records

def save_fasta(data_list, output_path):
    """將 list 寫回 FASTA 格式"""
    with open(output_path, 'w', encoding='utf-8') as f:
        for record in data_list:
            f.write(record)

def main():
    # --- 設定區 ---
    random_seed = 42
    ratio = 0.8  # 訓練集比例
    input_dir = "../cd_hit_2d/"  # 原始檔案目錄
    output_dir = "../data/" # 輸出檔案目錄
    
    # 定義原始檔案名稱
    pos_file = os.path.join(input_dir, "pos.fasta")
    neg_file = os.path.join(input_dir, "neg_1.fasta") # 建議用 neg1，確保 1:1 平衡
    
    # 讀取資料
    pos_records = read_fasta_to_list(pos_file)
    neg_records = read_fasta_to_list(neg_file)
    
    print(f"讀取完成: 正樣本 {len(pos_records)} 筆, 負樣本 {len(neg_records)} 筆")

    # 設定隨機種子
    random.seed(random_seed)
    
    # --- 處理正樣本 ---
    random.shuffle(pos_records)
    split_pos = int(len(pos_records) * ratio)
    pos_train = pos_records[:split_pos]
    pos_test = pos_records[split_pos:]
    
    # --- 處理負樣本 ---
    random.shuffle(neg_records)
    split_neg = int(len(neg_records) * ratio)
    neg_train = neg_records[:split_neg]
    neg_test = neg_records[split_neg:]
    
    # --- 儲存檔案 ---
    save_fasta(pos_train, os.path.join(output_dir, "pos_train.fasta"))
    save_fasta(pos_test, os.path.join(output_dir, "pos_test.fasta"))
    save_fasta(neg_train, os.path.join(output_dir, "neg_train.fasta"))
    save_fasta(neg_test, os.path.join(output_dir, "neg_test.fasta"))
    
    print("-" * 30)
    print("切割任務完成！詳細數量如下：")
    print(f"【訓練集 (80%)】: 正樣本 {len(pos_train)} 筆 / 負樣本 {len(neg_train)} 筆")
    print(f"【測試集 (20%)】: 正樣本 {len(pos_test)} 筆 / 負樣本 {len(neg_test)} 筆")
    print(f"產出檔案路徑: {output_dir}")
    print("-" * 30)

if __name__ == "__main__":
    main()