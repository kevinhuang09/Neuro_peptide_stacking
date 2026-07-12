# Neuro_peptide_stacking

> 基於 **Stacking 集成學習（Stacking Ensemble Learning）** 的神經胜肽（Neuropeptide）序列預測系統

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-0.23.2-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-Academic-green.svg)](#授權-license)
[![Status](https://img.shields.io/badge/Status-研究專題-yellow.svg)](#專案狀態)

---

## 📖 目錄（Table of Contents）

- [專案簡介](#專案簡介-introduction)
- [研究背景與動機](#研究背景與動機-background)
- [核心方法](#核心方法-methodology)
- [系統架構](#系統架構-architecture)
- [專案目錄結構](#專案目錄結構-project-structure)
- [資料來源](#資料來源-data-sources)
- [特徵工程](#特徵工程-feature-engineering)
- [環境需求與安裝](#環境需求與安裝-installation)
- [使用方式](#使用方式-usage)
- [模型評估指標](#模型評估指標-evaluation-metrics)
- [實驗流程](#實驗流程-pipeline)
- [常見問題](#常見問題-faq)
- [未來工作](#未來工作-future-work)
- [參考文獻](#參考文獻-references)
- [作者與致謝](#作者與致謝-authors)
- [授權](#授權-license)

---

## 專案簡介 (Introduction)

**Neuro_peptide_stacking** 是一個利用機器學習方法，從蛋白質 / 胜肽的胺基酸序列中，**辨識該序列是否為神經胜肽（Neuropeptide, NP）** 的生物資訊學（Bioinformatics）研究專案。

本專案的核心特色在於採用 **多層堆疊集成（Multi-layer Stacking Ensemble）** 架構：先以多種基礎分類器（Base Learners）在不同的序列特徵編碼上進行訓練，再將這些模型的預測輸出作為「元特徵（Meta-features）」，交由上層的元學習器（Meta Learner）做最終決策，藉此提升整體預測的準確度與穩定度。

本專案為 **大專生研究專題（Undergraduate Research Project）**，涵蓋完整的機器學習研究流程：資料蒐集與清洗 → 特徵工程 → 模型訓練與選擇 → 集成堆疊 → 效能評估 → 序列預測。

---

## 研究背景與動機 (Background)

**神經胜肽（Neuropeptides）** 是由神經元合成與分泌的一類小分子胜肽，作為神經傳導物質或神經調節因子，廣泛參與生物體內的：

- 疼痛調節、情緒與壓力反應
- 攝食行為、能量代謝
- 學習與記憶、睡眠週期
- 生殖與內分泌調控

由於神經胜肽在生理調節與潛在藥物開發（如止痛藥、代謝疾病治療）上具有重要價值，**如何快速且準確地從大量蛋白質序列中辨識出神經胜肽**，是一項具有實務意義的課題。

傳統以濕實驗（wet-lab）鑑定神經胜肽的方式耗時且成本高昂。因此，本專案希望透過 **計算方法（in-silico prediction）**，建立一套自動化、可規模化的神經胜肽序列分類模型，作為實驗前的高通量篩選工具。

---

## 核心方法 (Methodology)

本專案的方法核心為 **Stacking（堆疊泛化，Stacked Generalization）**，其基本概念為「用一個模型來學習如何組合多個模型的預測」。

### 為什麼使用 Stacking？

單一模型往往只能捕捉資料中的部分模式。透過集成多個具有不同偏誤（bias）與變異（variance）特性的基礎模型，並讓元學習器學習如何最佳地融合它們的輸出，通常能得到比任何單一模型更好、更穩健的預測結果。

### 多層架構概念

```
序列資料 (Amino Acid Sequences)
        │
        ▼
[ 特徵編碼 Feature Encoding ]  ← 多種序列描述子
        │
        ▼
┌─────────────────────────────────────────┐
│  第一層 First Layer (Base Learners)        │
│  多個基礎分類器 × 多種特徵                    │
│  例：XGBoost / LightGBM / CatBoost /       │
│      RandomForest / SVM / ...             │
└─────────────────────────────────────────┘
        │  產生每個模型的預測機率
        ▼
[ 元特徵生成 Meta-feature Generation ]
        │
        ▼
┌─────────────────────────────────────────┐
│  第二 / 三 / 四層 (Meta Learners)           │
│  以元特徵訓練上層模型，逐層融合               │
└─────────────────────────────────────────┘
        │
        ▼
   最終預測 Final Prediction (NP / non-NP)
```

> 對應原始碼中的 `first_layer.py`、`second_layer.py`、`third_ML.py`、`fourth_layer.py` 以及 `meta_feature_gent.py` 等模組。

---

## 系統架構 (Architecture)

整體流程可分為五大階段，對應到 `mainProgram/` 與 `MLProcess/` 中的模組：

| 階段 | 說明 | 對應主要程式 |
|------|------|--------------|
| 1️⃣ 資料前處理 | 序列去冗餘（CD-HIT）、正負樣本切分 | `get_all_cd_hit.py`, `split_dataset_2.py`, `remove_train_data.py` |
| 2️⃣ 資料擴增 | 對不平衡資料進行擴增 | `main_DataAugmentation.py` |
| 3️⃣ 特徵工程 | 序列特徵編碼與特徵選擇 | `main_Feature_v2.py`, `feature_encode.py`, `feature_split.py`, `boruta_selecion.py` |
| 4️⃣ 模型訓練與堆疊 | 多層 Stacking 集成訓練 | `main_ML.py`, `main_ML_neuro.py`, `main_stacking_ML.py`, `first_layer.py` ~ `fourth_layer.py` |
| 5️⃣ 預測與評估 | 對新序列進行預測、繪製效能圖表 | `main_Predict.py`, `predict.py`, `draw_plot.py`, `draw_picture.py` |

`MLProcess/` 則封裝了可重複使用的機器學習流程元件：

| 模組 | 功能 |
|------|------|
| `PycaretWrapper.py` | 封裝 [PyCaret](https://pycaret.org/) 進行模型自動訓練與比較 |
| `Stacking.py` | Stacking 集成模型的建構 |
| `StackingModelSelector_HC.py` | 基礎模型的挑選（Model Selection） |
| `Voting.py` | Voting 投票集成 |
| `Scoring.py` | 模型評分與效能計算 |
| `DrawPlot.py` | 效能結果視覺化 |
| `Predict.py` | 預測執行流程 |
| `change_binary.py` | 標籤 / 資料二元化轉換 |

---

## 專案目錄結構 (Project Structure)

```
Neuro_peptide_stacking/
│
├── mainProgram/               # 主要執行程式（研究主流程）
│   ├── main_DataAugmentation.py   # 資料擴增
│   ├── main_Feature_v2.py         # 特徵工程主程式
│   ├── main_ML.py                 # 機器學習主流程
│   ├── main_ML_neuro.py           # 神經胜肽專用 ML 流程
│   ├── main_ML_neuro_multi.py     # 多分類 / 多模型流程
│   ├── main_stacking_ML.py        # Stacking 集成主程式
│   ├── main_Predict.py            # 預測主程式
│   ├── first_layer.py             # 第一層基礎模型
│   ├── second_layer.py            # 第二層元模型
│   ├── third_ML.py                # 第三層
│   ├── fourth_layer.py            # 第四層
│   ├── meta_feature_gent.py       # 元特徵生成
│   ├── meta_feature_gent_add_mcc.py
│   ├── feature_encode.py          # 特徵編碼
│   ├── feature_split.py           # 特徵切分
│   ├── boruta_selecion.py         # Boruta 特徵選擇
│   ├── featuretype_mcc.py         # 各特徵類型的 MCC 評估
│   ├── split_dataset_2.py         # 資料集切分
│   ├── remove_train_data.py       # 移除重複訓練資料
│   ├── draw_plot.py / draw_picture.py  # 繪圖
│   └── predict.py / check.py / a001.py
│
├── MLProcess/                 # 可重用之 ML 流程元件
│   ├── PycaretWrapper.py
│   ├── Stacking.py
│   ├── StackingModelSelector_HC.py
│   ├── Voting.py
│   ├── Scoring.py
│   ├── Predict.py
│   ├── DrawPlot.py
│   └── change_binary.py
│
├── data/                      # 資料集
├── Uniprot/                   # UniProt 下載之蛋白質序列
├── swiss_prot/                # Swiss-Prot 資料（負樣本來源）
├── TopicalPdb/                # PDB 相關資料
├── cd-hit/ , cd_hit_2d/       # CD-HIT 序列去冗餘工具與結果
├── protGPT2/                  # ProtGPT2 相關（序列生成 / 擴增）
├── devPackage/ , userPackage/ # 自定義套件
├── tool/                      # 工具腳本
├── test/                      # 測試
│
├── get_all_cd_hit.py          # 批次執行 CD-HIT
├── output_90 / output_90.clstr # CD-HIT 90% 相似度去冗餘結果
├── swiss_prot紀錄.xlsx         # Swiss-Prot 資料處理紀錄
├── 0310.pptx / 0407.pptx      # 專題進度報告簡報
├── requirements.txt           # Python 套件相依清單
└── README.md
```

---

## 資料來源 (Data Sources)

| 來源 | 用途 | 說明 |
|------|------|------|
| **UniProt** | 正 / 負樣本序列 | 收錄已註解之蛋白質與胜肽序列 |
| **Swiss-Prot** | 負樣本（non-neuropeptide） | 人工審閱之高品質蛋白質資料庫 |
| **PDB (TopicalPdb)** | 結構相關資訊 | 蛋白質三維結構資料 |
| **CD-HIT 去冗餘** | 資料清洗 | 以序列相似度（例如 90%）移除高度相似序列，避免資料洩漏（Data Leakage） |
| **ProtGPT2** | 資料擴增 | 以蛋白質語言模型生成序列，緩解類別不平衡 |

> **CD-HIT** 是生物資訊學中常用的序列聚類工具，`output_90` 與 `output_90.clstr` 即為以 90% 序列一致性閾值進行去冗餘後的結果。此步驟確保訓練集與測試集之間不存在高度相似的序列，是胜肽 / 蛋白質預測研究中避免結果高估的關鍵環節。

---

## 特徵工程 (Feature Engineering)

胺基酸序列本身無法直接輸入機器學習模型，需先轉換為數值特徵向量。本專案使用多種常見的序列描述子（Sequence Descriptors），可能包含：

- **AAC（Amino Acid Composition）**：胺基酸組成比例
- **DPC（Dipeptide Composition）**：雙胜肽組成
- **CTD（Composition/Transition/Distribution）**：理化性質分布
- **PAAC / QSO** 等：偽胺基酸組成、序列順序耦合
- **理化性質特徵**（透過 `modlamp` 等套件計算）

### 特徵選擇（Feature Selection）

- **Boruta**（`boruta_selecion.py`）：以隨機森林為基礎的全相關特徵選擇演算法，篩選出真正具有辨識力的特徵。
- **MCC 導向評估**（`featuretype_mcc.py`）：以 Matthews 相關係數評估各特徵類型的貢獻度。

---

## 環境需求與安裝 (Installation)

### 系統需求

- Python **3.7+**（建議 3.7 / 3.8，因部分套件版本較舊）
- 建議使用 **虛擬環境（venv / conda）**
- （選用）CD-HIT 命令列工具

### 步驟

```bash
# 1. 複製專案
git clone https://github.com/kevinhuang09/Neuro_peptide_stacking.git
cd Neuro_peptide_stacking

# 2. 建立虛擬環境
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# 3. 安裝相依套件
pip install -r requirements.txt
```

> ⚠️ **注意**：`requirements.txt` 內含完整開發環境的所有套件（如 `pycaret`, `xgboost`, `lightgbm`, `catboost`, `scikit-learn==0.23.2`, `Boruta`, `modlamp`, `imbalanced-learn` 等）。由於部分套件版本較舊，建議在乾淨的虛擬環境中安裝，以避免版本衝突。若只需執行部分流程，可挑選必要套件安裝。

### 核心套件一覽

| 類別 | 主要套件 |
|------|----------|
| 資料處理 | `numpy`, `pandas`, `openpyxl`, `xlrd` |
| 機器學習 | `scikit-learn`, `xgboost`, `lightgbm`, `catboost` |
| 集成 / 自動化 | `pycaret`, `mlxtend`, `imbalanced-learn` |
| 特徵選擇 | `Boruta` |
| 胜肽特徵 | `modlamp` |
| 最佳化 | `optuna`, `hyperopt`, `scikit-optimize` |
| 視覺化 | `matplotlib`, `seaborn`, `plotly`, `yellowbrick` |
| 可解釋性 | `shap`, `lime`, `interpret` |

---

## 使用方式 (Usage)

> 以下為建議的執行順序，實際參數請依各程式內設定調整。

### 1. 資料去冗餘（CD-HIT）

```bash
python get_all_cd_hit.py
```

### 2. 資料集切分

```bash
python mainProgram/split_dataset_2.py
```

### 3. 資料擴增（選用，處理類別不平衡）

```bash
python mainProgram/main_DataAugmentation.py
```

### 4. 特徵工程

```bash
python mainProgram/main_Feature_v2.py
```

### 5. 訓練 Stacking 集成模型

```bash
# 基礎模型訓練
python mainProgram/main_ML_neuro.py

# 生成元特徵
python mainProgram/meta_feature_gent.py

# Stacking 集成
python mainProgram/main_stacking_ML.py
```

### 6. 對新序列進行預測

```bash
python mainProgram/main_Predict.py
```

### 7. 繪製效能圖表

```bash
python mainProgram/draw_plot.py
```

---

## 模型評估指標 (Evaluation Metrics)

本專案在二元分類（Neuropeptide vs. Non-neuropeptide）任務中，採用以下標準指標評估模型效能：

| 指標 | 說明 |
|------|------|
| **Accuracy (ACC)** | 整體預測正確率 |
| **Sensitivity (SN / Recall)** | 真陽性率，正確辨識神經胜肽的能力 |
| **Specificity (SP)** | 真陰性率，正確辨識非神經胜肽的能力 |
| **MCC** | Matthews 相關係數，適合類別不平衡資料，範圍 [-1, 1] |
| **AUC / ROC** | ROC 曲線下面積，衡量整體判別能力 |
| **F1-score** | 精確率與召回率的調和平均 |

> 在生物序列預測研究中，**MCC** 與 **AUC** 常被視為最重要的綜合指標，因為它們對類別不平衡較不敏感。

---

## 實驗流程 (Pipeline)

```
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│ 資料蒐集       │──▶│ CD-HIT 去冗餘 │──▶│ 正負樣本切分   │
│ UniProt/Swiss │   │  (90% 相似度) │   │              │
└──────────────┘   └──────────────┘   └──────┬───────┘
                                              │
                    ┌─────────────────────────┘
                    ▼
            ┌──────────────┐   ┌──────────────┐
            │ 資料擴增       │──▶│ 特徵編碼       │
            │ (ProtGPT2)   │   │ + Boruta 選擇 │
            └──────────────┘   └──────┬───────┘
                                       │
                    ┌──────────────────┘
                    ▼
            ┌──────────────────────────────┐
            │ 多層 Stacking 集成訓練          │
            │ Layer 1 → Meta → Layer 2/3/4  │
            └──────────────┬───────────────┘
                           │
                           ▼
            ┌──────────────┐   ┌──────────────┐
            │ 效能評估       │──▶│ 新序列預測     │
            │ ACC/MCC/AUC  │   │              │
            └──────────────┘   └──────────────┘
```

---

## 常見問題 (FAQ)

**Q1. 安裝套件時發生版本衝突怎麼辦？**
A：`requirements.txt` 為完整開發環境快照。建議先建立乾淨虛擬環境，若仍衝突，可只安裝執行特定腳本所需的核心套件（如 `scikit-learn`, `xgboost`, `pandas`, `numpy`）。

**Q2. 什麼是「去冗餘（Redundancy Reduction）」，為何重要？**
A：若訓練集與測試集中存在高度相似的序列，模型可能「背答案」導致效能被高估。使用 CD-HIT 移除相似序列可讓評估結果更貼近真實泛化能力。

**Q3. Stacking 和一般 Voting 有何不同？**
A：Voting 是以固定規則（多數決或平均）組合模型；Stacking 則是**訓練一個元學習器來學習如何組合**，通常表現更佳但也更容易過擬合，需搭配交叉驗證。

**Q4. 資料在哪裡？**
A：原始 / 處理後資料位於 `data/`、`Uniprot/`、`swiss_prot/` 等資料夾，處理紀錄見 `swiss_prot紀錄.xlsx`。

---

## 未來工作 (Future Work)

- [ ] 補充完整的資料集下載連結與資料格式說明
- [ ] 提供已訓練好的模型權重（pre-trained models）
- [ ] 加入獨立測試集（Independent Test Set）驗證泛化能力
- [ ] 提供 Web 介面 / API 供線上預測（可用 Flask / FastAPI，已列於相依套件）
- [ ] 整合 SHAP 進行模型可解釋性分析，找出關鍵胺基酸模式
- [ ] 與現有神經胜肽預測工具（如 NeuroPP、PredNeuroP）進行效能比較

---

## 參考文獻 (References)

> 以下為與本主題相關之常見參考方向，實際引用請依專題報告補充。

1. Wolpert, D. H. (1992). *Stacked generalization*. Neural Networks, 5(2), 241–259.
2. Fu, L., Niu, B., Zhu, Z., Wu, S., & Li, W. (2012). *CD-HIT: accelerated for clustering the next-generation sequencing data*. Bioinformatics, 28(23), 3150–3152.
3. Ferruz, N., Schmidt, S., & Höcker, B. (2022). *ProtGPT2 is a deep unsupervised language model for protein design*. Nature Communications, 13, 4348.
4. The UniProt Consortium. (2023). *UniProt: the Universal Protein Knowledgebase*. Nucleic Acids Research.
5. Chou, K. C. (2001). *Prediction of protein cellular attributes using pseudo-amino acid composition*. Proteins, 43(3), 246–255.

---

## 作者與致謝 (Authors)

- **作者 / Author**：Kevin Huang（[@kevinhuang09](https://github.com/kevinhuang09)）
- **專案性質**：大專生研究專題（Undergraduate Research Project）
- **研究領域**：生物資訊學 × 機器學習（Bioinformatics × Machine Learning）

感謝指導老師與實驗室夥伴於資料蒐集、模型設計與實驗過程中的協助。

---

## 授權 (License)

本專案為**學術研究用途**。如需引用、重製或商業使用，請先聯繫作者取得授權。

---

> 📌 *本 README 依據專案現有目錄結構與程式檔案命名整理而成。部分模組的詳細參數與資料格式，建議搭配原始碼註解及專題報告（`0310.pptx` / `0407.pptx`）一併閱讀。*