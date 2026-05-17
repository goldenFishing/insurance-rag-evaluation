# 保險條款檢索增強生成 (RAG) 系統

## 目錄

* [1. 系統概述]
* [2. 文檔解析與 RAG 前處理模組]
* [3. 高級檢索生成管線與推論引擎]
* [4. 批次推論與 Ragas 評估資料集建構單元]
* [5. RAG 框架量化評估單元]

---

## 1. 系統概述

本專案實作一基於檢索增強生成 (Retrieval-Augmented Generation, RAG) 架構之問答系統，專為保險條款與理賠實務（涵蓋保單條款及民法繼承編等相關法規）設計。系統透過向量檢索技術提取關聯條文，並經由大型語言模型 (LLM) 生成具備事實基礎之推論與解答，旨在降低法規解釋的幻覺 (Hallucination) 並提高回答的精確度。

使用者可直接於 Google Colab 環境中載入並執行 `main.ipynb` 筆記本。於執行過程中，請注意以下硬體資源與輸入輸出 (I/O) 之配置調整：

* **記憶體溢出 (Out of Memory, OOM) 處置**：由於大型語言模型 (LLM) 與向量嵌入模型 (Embedding Model) 之推論需耗費大量顯存與記憶體，若於執行期間遭遇 OOM 認證或硬體資源限制導致程式中斷，請重啟 Google Colab 的工作階段 (Restart Session) 後再度依序執行。
* **I/O 檔案路徑配置**：本系統涉及外部文檔之讀取與評估結果之複寫儲存。在執行任何涉及讀取或儲存檔案的儲存格 (Cells) 之前，請務必依據您實際的雲端雲碟掛載狀態或本機路徑，修正程式碼中的變數路徑。

---

## 2. 文檔解析與 RAG 前處理模組

本章節實作保險條款 PDF 文檔之端到端前處理流程。系統透過自動化管線將非結構化的條款文本轉化為結構化、語義高度集中的文字與表格區塊（Chunks），以供後續向量嵌入與檢索使用。

### 核心功能與處理流程

* **文檔載入與噪訊過濾 (Text Extraction & Denoising)**：利用 `PyMuPDF (fitz)` 提取原始文本，並透過正則表達式自動辨識並剔除頁碼、頁首頁尾等頁面噪訊（Page Noise），同時進行 Unicode 字元正規化。
* **版面配置輪廓偵測 (Layout Profile Detection)**：根據文檔首段之關鍵字特徵（如是否包含「商品代號」、「共同條款」或特定標記），自動將文檔分類為「富邦條款（帶括號標題）」、「章節型條款」、「一般條款」等特定版面 Profile，以優化後續切塊精確度。
* **保險中繼資料推論 (Metadata Inference)**：自動從文本與檔名中啟發式推導保險公司名稱（Company Name）、商品名稱（Product Name）、主附約屬性（Document Type，如主約或附約）以及方案名稱（Plan Name），並將此中繼資料封裝至每個 Chunk 中，以便進行精準的篩選檢索。
* **條文語義切塊 (Semantic Article Chunking)**：以「條（Article）」為核心語義單位進行切塊。透過正則表達式精確定位「第Ｘ條」之邊界，並向上回溯或向下提取該條文之實體標題，確保每個區塊皆具備完整的上下文資訊。
* **表格偵測與結構扁平化 (Table Parsing & Flattening)**：呼叫進階表格查核機制，當辨識到附表（如失能等級表、給付比例表）時，自動提取表頭並將二維表格行列結構「扁平化（Flatten）」為鍵值對（Key-Value Pairs）形式的語義文本（Semantic Text），解決傳統 RAG 無法有效索引表格資料的痛點。
* **多格式產出與儲存 (Multi-Format Outputs)**：前處理完成後，系統將自動於 `./outputs` 目錄生成通用 CSV/JSONL 資料，並於 `./outputs/rag` 目錄產出專供檢索器（Retriever）直接匯入之 `policy_rag_text_chunks.jsonl`（純文字條文）與 `policy_rag_table_chunks.jsonl`（結構化表格）。

---

## 3. 高級檢索生成管線與推論引擎

本章節實作系統的核心推論與檢索模組，採用多階段路由、雙軌混合檢索、交叉重排以及複用型大型語言模型生成技術，構建高度防禦幻覺且顯存優化的端到端 RAG 管線。

### 核心模組與技術原理

* **階段一：結構化意圖解析與動態路由 (Intent Parsing & Routing)**
* **技術實作**：採用 `Qwen2.5-7B-Instruct` 進行 4-bit 量化 (`BitsAndBytesConfig / NF4`) 推論，將使用者輸入之自然語言查詢，精準解析為包含保險公司（`company_name`）等屬性之中繼資料過濾器（Metadata Filters）。
* **路由機制**：動態判定查詢意圖類型（`intent_type`），將問題分類為純文字概念（`text_qa`）、數值與圖表查核（`table_qa`）或混合型（`mixed`），依此判定後續向量檢索的通道。


* **階段二與三：雙軌混合檢索與動態上下文組裝 (Hybrid Search & Context Aggregation)**
* **密集與稀疏向量檢索 (Dense & Sparse Retrieval)**：利用 `BGE-M3` 嵌入模型，同步計算文本與表格區塊的語義向量（Dense）與詞彙權重（Sparse），並透過倒數排名融合演算法（Reciprocal Rank Fusion, RRF）加權整合，排除傳統單一檢索的盲點。
* **上下文動態壓縮與映射 (Context Compression & Expansion)**：
* *純文字*：採用父子文檔映射邏輯，檢索到子區塊後自動向外擴充相鄰條文，確保法規上下文的完整性。
* *表格數據*：引入列層級（Row-level）二次嵌入計算，當檢索到大型附表時，僅動態截取與查詢最相關的 Top-k 扁平化行數據，避免無關數據污染上下文。




* **階段四：原生交叉重排機制 (Cross-Encoder Reranking)**
* **技術實作**：使用 `bge-reranker-v2-m3` 模型進行精準的雙向注意力機制（Cross-Attention）計分，為所有召回的父文本區段與原始查詢進行相似度重排，並設立嚴格的分數閾值（Threshold Filter），僅保留高置信度的核心文獻。


* **階段五：顯存防禦型答案生成 (Memory-Defensive Generation)**
* **顯存配置優化 (VRAM Management)**：透過 XML 標籤（`<context>` 與 `<query>`）將隔離後的文獻注入千問模型（Qwen2.5）。在生成前與生成後，強制執行垃圾回收（`gc.collect()`）與顯存清空（`torch.cuda.empty_cache()`），並引入重複性懲罰（`repetition_penalty`）與 N-gram 禁制，徹底根除 Colab 環境下長文本推論常見的記憶體溢出（OOM）與循環覆寫問題。



---

## 4. 批次推論與 Ragas 評估資料集建構單元 (Batch Inference & Ragas Dataset Alignment)

本章節實作問答系統的生產環境批次推論管線，並負責將生成的預測結果與人工標註之真實標籤（Ground Truth）進行多維度對齊，以建構標準化之 RAG 自動化評估資料集。

### 本章節與上一章節之架構差異

* **核心定位之轉變**：上一章節聚焦於「單一元件的技術實作與管線串接原理」**（包含意圖解析、向量雙軌檢索、交叉重排及顯存防禦生成機制）；本章節則躍升至**「系統級的壓力測試、產出驗證與資料持久化」，針對大規模多元保險理賠邊緣案例（Edge Cases）進行批次化推論。
* **評估層級之擴充**：上一章節僅對查詢進行即時的主控台（Console）推論輸出；本章節引入了端到端 Ragas（Retrieval-Augmented Generation Assessment）評估框架之標準資料容器，將推論過程中涉及的「原始提問（`question`）」、「高分檢索上下文（`contexts`）」、「模型預測答案（`answer`）」與「標準答案（`ground_truth`）」封裝為具備四維對齊結構的 `Dataset`。
* **資料持久化機制**：新增外部 I/O 串接單元，透過掛載虛擬檔案系統將整合完成的 Ragas 評估陣列序列化為 JSON 格式，並寫入指定之雲端儲存路徑（如 Google Drive），為後續計算忠實度（Faithfulness）、答案相關性（Answer Relevance）等量化指標奠定數據基礎。

---

## 5. RAG 框架量化評估單元

本章節實作問答系統的量化評估管線。系統導入 `Ragas` 評估框架，多維度量化檢索品質與生成文本的置信度。為因應雲端 API 速率限制（Rate Limits）與不穩定性，本單元採取「流式批次迴圈」與「單題中繼除錯」之雙軌容錯架構。

### 1. 評估指標與基底模型配置

評估管線採用裁判模型（LLM-as-a-Judge）機制，底層核心配置如下：

* **裁判 LLM**：使用 `gemini-3.1-flash-lite`（經由 `LangchainLLMWrapper` 封裝），提供高速且具成本效益的語義審查。
* **評估嵌入模型**：使用 `BAAI/bge-m3`（經由 `LangchainEmbeddingsWrapper` 封裝），維持與檢索端一致的向量空間。
* **核心量化指標 (Metrics)**：
* **忠實度 (Faithfulness)**：評估生成答案是否完全基於檢索上下文，用以偵測幻覺。
* **答案相關性 (Answer Relevancy)**：評估輸出內容是否切中使用者提問核心。
* **檢索精確度 (Context Precision)**：評估召回的高分文本片段是否與問題密切相關。



### 2. 流式批次評估機制與速率防禦

為了防止大量批次呼叫時觸發遠端 API 的 Rate Limit 或連線中斷，主評估管線採用以下防禦策略：

* **單題獨立迭代**：將上一章節持久化儲存之四維對齊資料集（`ragas_temp_data.json`）載入後，不直接進行整包評估，而是透過 `for` 迴圈逐題分離出單一子集（`single_item`）送入 `evaluate` 引擎。
* **時間延遲防禦**：每輪迭代強制執行定時休眠 `time.sleep(20)`，平滑化請求頻率，確保整體評估腳本的強健性（Robustness）。

### 3. 異常容錯與動態單題除錯機制 (Fault Tolerance & Single-Index Debugging)

若於自動化批次評估期間，因網路波動、API 憑證失效或文本長度觸發遠端限制而導致部分題目評估失敗（或產出 `NaN` 異常值），系統內建動態診斷機制：

* **容錯隔離**：設定 `raise_exceptions=False`，確保單一題目失敗時不中斷整條流水線，並持續累加成功題目的觀測值。
* **獨立單題診斷單元 (`evaluate_single_index`)**：當特定題目 API 請求失敗或分數異常時，使用者可直接移步執行「單題測試函數定義」儲存格（Cell）。只要傳入指定索引值（`target_index`），該單元即會繞過批次管線，完全展開「原始問題」、「檢索條文 (Contexts) 段數」、「LLM 實際回答」與「人工標準答案 (Ground Truth)」進行肉眼對齊比對，並單獨對該題發起 Ragas 評估請求，以利進行精準排錯。
