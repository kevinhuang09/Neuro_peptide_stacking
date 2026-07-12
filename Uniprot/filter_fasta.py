def filter_fasta(input_file, output_fasta, log_file):
    remove_count = 0
    total_count = 0
    with open(input_file, 'r') as f_in, \
        open(output_fasta, 'w') as f_out, \
        open(log_file, 'w') as f_log:
        f_log.write("be remove list : \n")
        f_log.write("-" * 30 + "\n")

        header = ""
        sequence = ""

        def process_entry(h, s):
            nonlocal remove_count, total_count
            if not h: return 
            
            # 修正：移除序列中的空格、換行以及常見的終止符 *
            clean_seq = s.replace(" ", "").replace("\n", "").replace("*", "")
            total_count += 1
            seq_len = len(clean_seq)

            if 5 <= seq_len <= 100:
                f_out.write(f"{h}\n{clean_seq}\n") # 寫入清理後的序列
            else:
                remove_count += 1
                f_log.write(f"ID: {h.strip()} | length : {seq_len}\n")

        for line in f_in:
            line = line.strip()
            if line.startswith(">"):
                process_entry(header, sequence)
                header = line
                sequence = ""
            else:
                sequence += line
        process_entry(header, sequence)

        f_log.write("-" * 30 + "\n")
        f_log.write(f"total read : {total_count} sequences\n")
        f_log.write(f"total remove : {remove_count} sequences\n")
        f_log.write(f"after filter save : {total_count - remove_count} sequences\n")
    print("process complete!")
    print(f"save sequence saved to {output_fasta}")
    print(f"remove sequence saved to {log_file}")
filename = "uniprot_neuropeptides_massive"
outputname = f"{filename}_filter.fasta"
remove_record = f"{filename}_remove_log.txt"
filter_fasta(f"{filename}.fasta", outputname, remove_record)