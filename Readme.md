# 個人 AI 助理專案總結

**專案目標：**

* 為使用者量身打造一個個人 AI 助理。
* 具備更好的使用介面（網頁/手機）。
* 能夠管理使用者個人的知識筆記。
* 能夠獲取和提供外部（網路）資訊。
* 具備語音互動能力。
* 符合 Industry Standard 的程式設計實踐。

**核心已實現功能：**

1.  **基於大型語言模型 (LLM) 的智能對話：** 使用 Gemini 模型進行自然語言聊天互動。
2.  **個人知識庫問答 (RAG)：**
    * 載入和分割使用者提供的文件（如 `.txt`, `.pdf`）。
    * 將文件片段轉換為向量 (Embedding)。
    * 將向量和文件片段儲存到**持久化的向量資料庫 (Chroma)**。
    * 根據使用者問題，在向量資料庫中檢索最相關的個人筆記片段。
    * **混合檢索 (Hybrid Search)：** 結合向量檢索和**基於關鍵詞的 BM25 檢索**來提高檢索相關性。
3.  **網路資訊獲取 (RAG)：**
    * 基於 AI 的判斷，決定使用者問題是否需要進行網路搜尋。
    * 呼叫 Google Custom Search API 執行網路搜尋。
    * 將網路搜尋結果作為上下文提供給 AI。
4.  **多來源上下文整合：** 將**個人知識庫檢索結果**和**網路搜尋結果**（如果執行）與**使用者原始問題**結合，構造增強型 Prompt 發送給 AI，生成基於多個資訊來源的回答。
5.  **AI 意圖判斷：** AI 助理能夠自主判斷使用者輸入的**意圖**（例如：通用聊天 `chat`、解釋概念 `explain`、討論 `discuss`），並根據不同意圖調整回答風格和檢索策略。
6.  **對話歷史記錄：** 將使用者和 AI 的對話訊息**持久化儲存到 SQLite 資料庫**中。
7.  **文件上傳與管理：** 通過 API 端點接收使用者上傳的文件，自動處理並添加到個人知識庫向量資料庫。
8.  **語音輸入與處理：**
    * 通過 API 端點接收音頻文件或錄音數據。
    * 利用 OpenAI Whisper API 進行**語音轉文字 (STT)**。
    * 將轉錄的文字發送給 AI 進行**自動摘要**。
    * 將摘要結果**保存到個人知識庫**。
9.  **後端 API 服務：** 基於 **FastAPI** 構建後端，提供結構化的 API 端點供前端或其他應用呼叫（`/chat`, `/upload_document`, `/upload_audio_for_summary`, `/history`, `/latest_newsletter`）。
10. **網頁前端介面：** 一個基於 HTML, CSS, JavaScript 的響應式網頁，提供了文字聊天、歷史記錄顯示、文件上傳、音頻文件上傳和即時語音錄製等使用者介面功能。可以通過瀏覽器訪問。
11. **Newsletter 功能：** 後端邏輯可以根據預設興趣**生成 AI 新聞 Newsletter**（基於網路搜尋和 AI 總結），並將其保存到資料庫。前端可以獲取並顯示最新一份 Newsletter（包含原始來源連結列表）。

**使用的主要技術和工具：**

* **程式語言：** Python, JavaScript, HTML, CSS
* **Python 函式庫/框架：**
    * **後端框架：** FastAPI
    * **ASGI 服務器：** Uvicorn
    * **AI 模型互動：** `google-generativeai` (Gemini API), `openai` (Whisper API)
    * **RAG 框架/工具：** LangChain (文件載入、文字分割、Google 嵌入模型封裝), Chroma (向量資料庫), `rank_bm25` (BM25 檢索)
    * **資料庫/ORM：** SQLModel (ORM, Pydantic/SQLAlchemy 結合), SQLAlchemy (資料庫核心)
    * **Google API 客戶端：** `google-api-python-client` (用於 Google Search API)
    * **文件處理：** `unstructured`, `pypdf` (LangChain 文件載入器依賴)
    * **其他標準庫：** `os`, `sys`, `logging`, `json`, `textwrap`, `typing`, `datetime`, `shutil`, `tempfile`
* **前端函式庫：** Marked.js (Markdown 渲染)
* **資料庫：** SQLite (`chat_history.db`)
* **開發工具：** Git (版本控制), 命令列終端機, 瀏覽器開發者工具

**應用的核心概念和技術：**

* **RAG (Retrieval Augmented Generation):** 從外部知識源檢索信息來增強 LLM 的生成能力。
* **Vector Embeddings：** 將文字轉換為數字向量，用於語義相似度計算。
* **Vector Database：** 專門儲存和檢索向量數據的資料庫。
* **Hybrid Search：** 結合向量檢索和關鍵詞檢索來提高檢索的精準度和召回率。
* **Prompt Engineering：** 設計有效的指令來引導 LLM 執行特定任務、扮演角色、利用上下文等。
* **AI 意圖識別：** 利用 AI 模型分析使用者輸入，判斷其背後的主要意圖。
* **API Design (RESTful 原則):** 設計結構化的後端接口。
* **客戶端-伺服器架構：** 將應用分為獨立的前端和後端服務。
* **響應式網頁設計：** 使用 CSS 和 HTML 使網頁在不同設備上都能良好顯示。
* **資料持久化：** 將數據（對話歷史、Newsletter、向量數據庫）儲存到磁碟，以便程式重啟後數據不丟失。
* **ORM (Object-Relational Mapping):** 使用 Python 對象操作關係型資料庫。
* **Pydantic：** 資料驗證和序列化。
* **SQLModel：** Pydantic 和 SQLAlchemy 的結合，簡化模型定義。
* **依賴注入 (Dependency Injection)：** 在 FastAPI 中管理依賴關係（如資料庫 Session）。
* **環境變數：** 安全地管理 API 金鑰等敏感資訊。
* **日誌記錄：** 記錄應用程式的運行狀態和錯誤。
* **瀏覽器 API：** Fetch API (網絡請求), MediaRecorder API (語音錄製), Web Streams API (處理數據流)。
* **CORS：** 處理跨來源請求問題。

**專案檔案結構 (示例):**