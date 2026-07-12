def filter_fasta(input_file, output_file, remove_ids):
    from Bio import SeqIO # 需安裝 biopython: pip install biopython
    
    # 您提供的需要移除的 Accession ID 列表
    ids_to_remove = set(remove_ids)
    
    filtered_records = []
    count = 0
    
    # 讀取原始 FASTA
    for record in SeqIO.parse(input_file, "fasta"):
        # 提取 ID (例如 sp|A8AHK2|... 中的 A8AHK2)
        accession = record.id.split('|')[1] if '|' in record.id else record.id
        
        if accession not in ids_to_remove:
            filtered_records.append(record)
        else:
            print(f"已移除洩漏序列: {record.id}")
            count += 1
            
    # 存回新的 FASTA
    SeqIO.write(filtered_records, output_file, "fasta")
    print(f"--- 處理完成，共移除 {count} 條序列 ---")

# 1. 定義要移除的 ID (根據您提供的資料)
target_ids = [
    "A8AHK2", "O34862", "P54264", "Q31VJ3", "Q04100", 
    "P25516", "P15751", "P59714", "Q2KMM2", "Q5XEC9", "A0A1X9ISP7"
]

# 2. 執行過濾 (請修改您的檔名)
filter_fasta("../data/neg_test.fasta", "../data/neg_test_cleaned.fasta", target_ids)