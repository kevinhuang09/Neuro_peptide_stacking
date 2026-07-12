import requests

def download_uniprot_foolproof():
    # 1. 使用 UniProt 的 Stream API 接口
    url = "https://rest.uniprot.org/uniprotkb/stream"
    
    # 2. 設定參數 (讓 requests 自動處理編碼，解決 400 Error)
    # query: 使用最廣泛的關鍵字 "Neuropeptide" (包含 KW-0527 與文字描述)
    # 我們不加 reviewed:true，這樣可以抓到 TrEMBL (未審核) 的數據，保證數量 > 5000
    params = {
        "query": "keyword:KW-0527", 
        "format": "fasta",
        "includeIsoform": "true"  # 包含異構體，最大化數據量
    }
    
    print("📡 正在連接 UniProt (使用參數自動編碼)...")
    
    try:
        # 3. 發送請求 (stream=True 避免記憶體爆掉)
        # 注意：我們把 params 傳進去，requests 會自動把它轉成正確的網址格式
        with requests.get(url, params=params, stream=True) as response:
            response.raise_for_status() # 檢查是否有 400/500 錯誤
            
            output_file = "uniprot_neuropeptides_massive.fasta"
            
            print(f"📥 連接成功！正在下載至 {output_file}...")
            print("⏳ 數據量可能很大 (預計超過 10萬條)，請耐心等待...")
            
            with open(output_file, "wb") as f:
                # 每次讀取 10MB，加快寫入速度
                for chunk in response.iter_content(chunk_size=10 * 1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        
        # 4. 統計結果
        print("✅ 下載完成！正在計算數量...")
        with open(output_file, "r", encoding="utf-8") as f:
            # 計算有多少個 ">" 符號
            count = sum(1 for line in f if line.startswith(">"))
            
        print("-" * 30)
        print(f"📊 最終下載數量: {count} 條序列")
        
        if count > 5000:
            print("✨ 成功！數量遠超 5000 條。")
        else:
            print("⚠️ 數量仍然不足，請檢查網路或 API 限制。")
            
    except requests.exceptions.HTTPError as err:
        print(f"❌ 下載失敗 (HTTP Error): {err}")
        print("💡 原因可能是伺服器暫時繁忙，請過幾分鐘再試。")
    except Exception as e:
        print(f"❌ 發生未預期的錯誤: {e}")

if __name__ == "__main__":
    download_uniprot_foolproof()