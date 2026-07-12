def merge_fasta_files(input_files, output_file):
    try:
        with open(output_file, 'w') as outfile:
            for file_name in input_files:
                with open(file_name, 'r') as infile:
                    # 讀取並寫入內容
                    outfile.write(infile.read())
                    # 確保檔案結尾有換行，避免兩條序列首尾相連
                    if not outfile.tell() == 0: # 如果不是空檔案
                        outfile.write('\n')
        print(f"成功！已將檔案合併至: {output_file}")
    except FileNotFoundError as e:
        print(f"錯誤：找不到檔案 - {e}")

# 設定檔案路徑
files_to_combine = ['pos_train.fasta', 'pos_test.fasta']
target_name = 'neuro.fasta'

merge_fasta_files(files_to_combine, target_name)