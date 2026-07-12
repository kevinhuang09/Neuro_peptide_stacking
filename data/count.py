def count_fasta_sequences(filename):
    count = 0
    try:
        with open(filename, 'r') as f:
            for line in f:
                # 判斷行首是否為 '>'
                if line.startswith('>'):
                    count += 1
        return count
    except FileNotFoundError:
        return "找不到檔案，請檢查路徑。"

# 使用範例
file_path = "neuro.fasta" # 這裡換成你的檔案名稱
print(f"共有 {count_fasta_sequences(file_path)} 條序列")