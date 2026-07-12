import torch
from transformers import pipeline

class Protgpt2:
    def __init__(self, origFile, outputPath, foldNum, modelPath, maxlenth, minlenth, datasetName, outputSuffix):
        self.origFile = origFile
        self.outputPath = outputPath
        self.foldNum = foldNum
        self.modelPath = modelPath
        self.maxlenth = maxlenth
        self.minlenth = minlenth
        self.datasetName = datasetName
        self.outputSuffix = outputSuffix
        
        # --- GPU 偵測部分 ---
        print("-" * 30)
        print("torch.__version__:", torch.__version__)
        print("torch.version.cuda:", torch.version.cuda)
        print("torch.cuda.is_available():", torch.cuda.is_available())
        print("GPU count:", torch.cuda.device_count())
        
        self.device = -1
        if torch.cuda.is_available():
            self.device = 0 # 使用第一張顯卡
            print("Current device:", torch.cuda.current_device())
            print("GPU name:", torch.cuda.get_device_name(0))
        else:
            print("警告: 未偵測到 GPU，將使用 CPU 進行運算，速度會非常慢。")
        print("-" * 30)

        # 1. 預先載入模型 (Pipeline 內部會處理 Tokenizer)
        print(f"正在載入模型: {self.modelPath} ...")
        self.generator = pipeline(
            'text-generation', 
            model=self.modelPath, 
            device=self.device
        )

    def listToFasta(self, lst, output_file, peptideNameNumStart):
        with open(output_file, 'w') as file:
            num = peptideNameNumStart + 1 if peptideNameNumStart is not None else 1
            for sequence in lst:
                file.write(f'>GPT{self.outputSuffix}{num}\n{sequence}\n')
                num += 1

    def is_valid_sequence(self, sequence, current_fold_list, master_set):
        """ 檢查序列是否合法、不重複、長度正確 """
        amino_acids = set("ACDEFGHIKLMNPQRSTVWY")
        
        # 1. 長度檢查
        if not (self.minlenth <= len(sequence) <= self.maxlenth):
            return False
        # 2. 重複檢查 (檢查本次生成與歷史紀錄)
        if sequence in current_fold_list or sequence in master_set:
            return False
        # 3. 氨基酸合法性檢查
        if not all(aa.upper() in amino_acids for aa in sequence):
            return False
            
        return True

    def gen(self, start_aa):
        # 這裡的參數可以根據需求微調
        res = self.generator(
            f"{start_aa}", 
            max_new_tokens=self.maxlenth, 
            min_new_tokens=self.minlenth, 
            do_sample=True, 
            top_k=950, 
            repetition_penalty=1.2,
            num_return_sequences=1, 
            eos_token_id=0
        )
        # 移除換行符與空格
        return res[0]["generated_text"].replace("\n", "").replace(" ", "")

    def readFastaToList(self, fileInput):
        orig_peps = []
        for line in fileInput:
            line = line.strip()
            if line and not line.startswith(">"):
                orig_peps.append(line)
        return orig_peps

    def main(self):
        with open(self.origFile, "r") as f:
            origPepList = self.readFastaToList(f)
        
        # master_set 用於快速檢查該序列是否在歷史中出現過
        master_set = set(origPepList)
        peptideNameNumStart = len(origPepList)

        for i in range(1, self.foldNum + 1):
            foldPepList = []
            print(f"\n開始生成第 {i} 輪 (Fold {i})...")
            
            for pepStr in origPepList:
                startAA = pepStr[0]
                isValid = False
                retry_count = 0
                
                while not isValid and retry_count < 10: # 避免死迴圈
                    genPepStr = self.gen(startAA)
                    if self.is_valid_sequence(genPepStr, foldPepList, master_set):
                        foldPepList.append(genPepStr)
                        master_set.add(genPepStr) # 加入 master_set 防止未來重複
                        isValid = True
                        print(f"[Success] {genPepStr}")
                    retry_count += 1
            
            # 存儲當前 Fold 的結果
            output_name = f"{self.outputPath}{self.datasetName}x{i}{self.outputSuffix}_gpt.fasta"
            self.listToFasta(foldPepList, output_name, peptideNameNumStart)
            
            # 更新下一輪的起始編號
            peptideNameNumStart += len(foldPepList)
            
            # 存儲累計結果
            accumulated_name = f"{self.outputPath}{self.datasetName}x{i}{self.outputSuffix}_accumulated_gpt.fasta"
            self.listToFasta(list(master_set), accumulated_name, None)

import os 
output_dir = "../data/neuro_output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# --- 執行 ---
protgpt2Obj = Protgpt2(
    origFile="../data/neuro.fasta",
    outputPath= output_dir,
    modelPath="nferruz/ProtGPT2",  # 修改這裡：使用官方預訓練模型名稱
    foldNum=10,
    maxlenth=100,
    minlenth=5,
    datasetName="neuro_ext",
    outputSuffix="v1"
)
protgpt2Obj.main()