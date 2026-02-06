# TTAS LLM 檢傷分類預測系統 (Triage Prediction System)

本目錄包含 TTAS（台灣檢傷分類標準）檢傷分類預測系統的程式碼。此系統利用量化的 Qwen3-4B-Instruct 模型（透過 `llama-cpp-python`）和檢索增強生成（RAG）技術（使用 ChromaDB），根據病患的主訴和生命徵象數據來協助判斷檢傷分級。

## 目錄結構

```
data_explore/
├── models/             # 存放模型檔案的目錄 (例如 .gguf 檔)
├── test_file/          # 包含測試資料檔案 (例如 test.json)
├── utilities/          # 輔助腳本
│   ├── chunk_list.py           # 用於 RAG 的檢傷標準文本塊
│   ├── get_patient_info_csv.py # 讀取並格式化病患資料的腳本
│   └── save2json.py            # 儲存結果的腳本
├── main.py             # 程式的主要進入點
├── model_init.py       # 模型與向量資料庫初始化
├── model_predict.py    # 使用 LLM 和 RAG 進行預測的邏輯
└── README.md           # 本說明文件
```

## 前置需求

請確保已安裝以下 Python 套件：

- `torch`
- `chromadb`
- `llama-cpp-python`
- `pandas`

(您可能需要根據硬體配置調整 `llama-cpp-python` 的安裝方式，例如為了支援 CUDA 加速)。

## 設定

1.  **模型準備：** 將您的 GGUF 模型檔案（例如 `Qwen3-4B-Instruct-2507-Q4_K_M.gguf`）放入 `models/` 目錄中。
2.  **資料準備：** 確保輸入資料（JSON 或 CSV）可被讀取。系統目前預設讀取 `test_file/test.json`，或可於 `utilities/get_patient_info_csv.py` 中設定特定的 CSV 路徑。

## 使用方式

執行 `main.py` 腳本來運行完整的流程：

```bash
python main.py
```

### 運作流程
1.  **初始化 (`model_init.py`)：** 載入 LLM 並建立包含檢傷標準文本塊的向量資料庫 (ChromaDB)。
2.  **資料讀取 (`get_patient_info_csv.py`)：** 從 JSON 或 CSV 讀取病患資訊（生命徵象、主訴等）並進行格式化。
3.  **預測 (`model_predict.py`)：**
    -   **RAG (檢索)：** 根據病患主訴檢索相關的檢傷標準。
    -   **LLM (生成)：** 結合檢索到的標準與病患資訊，生成檢傷分級與理由。
4.  **輸出 (`save2json.py`)：** 將最終判斷結果存為 JSON 檔案。

- **輸入輸出**
    - **輸入**：`test_file/test.json`
    - **輸出**：`test_file/out.json`

## 關鍵檔案說明

-   **`main.py`**：統籌初始化、資料讀取、預測與儲存結果的流程。
-   **`model_init.py`**：處理 GPU 偵測、透過 `llama_cpp` 載入模型，以及建立 ChromaDB 集合。
-   **`model_predict.py`**：包含執行向量搜尋的 `model_predict` 函數，以及提示 LLM 生成回應的 `generate_ttas_response` 函數。
-   **`utilities/get_patient_info_csv.py`**：負責解析不同來源（JSON/CSV）的病患資料，並格式化為 LLM 所需的標準字串格式。

## 備註

-   系統預設使用本地 GGUF 模型進行推論。
-   若遇到 VRAM 問題或希望調整 GPU 的負載比例，請調整 `model_init.py` 中的 `n_gpu_layers` 參數。
