# ai_assistant_core.py (建議將原文件改名，以區別核心邏輯和 API/UI 入口)
# 個人 AI 助理核心邏輯函式庫

import google.generativeai as genai
import os
import textwrap
import logging
import sys
import googleapiclient.discovery
import json
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from typing import Optional # 用於類型提示 Optional
import openai

# --- 設定日誌記錄 (Industry Standard) ---
# 配置 logging 模組，通常在應用入口點配置一次即可
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    stream=sys.stdout)

# --- 全局變數 (在應用啟動時賦值) ---
# 這些變數將在應用啟動時初始化，並被核心函式使用
# 外部模組可以通過函式參數傳遞或直接訪問 (如果定義在類中)
# 在這裡先保持為全局，方便函式訪問
# model: Optional[genai.GenerativeModel] = None # 在初始化函式中賦值
# vectorstore: Optional[Chroma] = None # 在初始化函式中賦值
# GOOGLE_API_KEY: Optional[str] = None # 在載入函式中賦值
# SEARCH_API_KEY: Optional[str] = None # 在載入函式中賦值
# SEARCH_ENGINE_ID: Optional[str] = None # 在載入函式中賦值

# --- 定義向量資料庫的儲存路徑 ---
PERSIST_DIRECTORY = "vector_db"
DEFAULT_GEMINI_MODEL_NAME = 'gemini-1.5-flash-latest'
DEFAULT_EMBEDDING_MODEL_NAME = 'models/text-embedding-004'
DEFAULT_WHISPER_MODEL_NAME = 'whisper-1'

# --- 將環境變數載入封裝成函式 ---
def load_env_variables() -> dict: # 返回一個字典
    """從環境變數載入所有 API 金鑰和 IDs，並返回一個設定字典。"""
    config = {}
    config['GOOGLE_API_KEY'] = os.environ.get('GOOGLE_API_KEY')
    if not config['GOOGLE_API_KEY']:
        logging.error("錯誤：GOOGLE_API_KEY 環境變數未設定。")
        logging.info("請設定環境變數以啟用 Gemini 模型。")

    config['SEARCH_API_KEY'] = os.environ.get('Google_Search_API_KEY')
    config['SEARCH_ENGINE_ID'] = os.environ.get('Google_Search_ENGINE_ID')

    if not config['SEARCH_API_KEY'] or not config['SEARCH_ENGINE_ID']:
        logging.warning("警告：Google Search_API_KEY 或 Google Search_ENGINE_ID 環境變數未設定。資訊搜集 (網路搜尋) 功能將無法使用。")

    config['OPENAI_API_KEY'] = os.environ.get('OPENAI_API_KEY')
    if not config['OPENAI_API_KEY']:
        logging.warning("警告：OPENAI_API_KEY 環境變數未設定。語音轉文字 (STT) 功能將無法使用。")

    logging.info("環境變數載入完成。")
    return config # 返回設定字典


# --- 將 Gemini 模型初始化封裝成函式 ---
def initialize_gemini_model(api_key: str, model_name: str = DEFAULT_GEMINI_MODEL_NAME) -> Optional[genai.GenerativeModel]:
    """初始化 Gemini 模型。"""
    if not api_key:
        logging.error("缺少 API 金鑰，無法初始化 Gemini 模型。")
        return None

    logging.info(f"嘗試初始化 Gemini 模型：{model_name}")
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        logging.info(f"成功初始化 Gemini 模型：{model_name}")
        return model
    except Exception as e:
        logging.error(f"配置 Gemini API 金鑰或初始化模型 {model_name} 時發生錯誤: {e}", exc_info=True)
        return None # 初始化失敗返回 None


# --- 建立執行網路搜尋的函式 ---
def perform_web_search(query, api_key, cx_id, num_results=5):
    """
    使用 Google Custom Search Engine API 執行網路搜尋。

    Args:
        query (str): 搜尋查詢字串。
        api_key (str): Google Custom Search API 金鑰。
        cx_id (str): Custom Search Engine 的 ID。
        num_results (int): 希望獲取的搜尋結果數量 (最多 10 個)。

    Returns:
        list: 搜尋結果列表 (字典形式)，每個字典包含 'title', 'link', 'snippet' 等資訊。
              如果搜尋失敗或無結果，則返回空列表 []。
    """
    # 在呼叫函式時再次檢查金鑰和 ID 是否存在
    if not api_key or not cx_id:
        # logging.warning("perform_web_search 函式被呼叫，但缺少搜尋 API 金鑰或 CX ID。") # 避免重複警告
        return [] # 返回空列表表示無結果，避免後續處理出錯

    try:
        # 創建 CSE 服務對象
        service = googleapiclient.discovery.build(
            "customsearch", "v1", developerKey=api_key
        )

        logging.info(f"開始執行網路搜尋：'{query}' (獲取 {num_results} 個結果)")

        # 執行搜尋請求
        search_results = service.cse().list(
            q=query,
            cx=cx_id,
            num=num_results, # 獲取指定數量的結果 (num 參數最大為 10)
            # 添加更多參數以精煉搜尋結果 (可選)
            # lr='lang_zh-TW', # 可選：限定搜尋語言為繁體中文
            # gl='tw' # 可選：限定搜尋地區為台灣
            # siteSearch='example.com' # 可選：只搜尋特定網站
        ).execute()

        # 檢查是否有搜尋結果
        items = search_results.get('items', [])
        logging.info(f"網路搜尋完成，找到 {len(items)} 個結果。")

        # 返回搜尋結果列表
        return items

    except Exception as e:
        # 捕獲搜尋過程中可能發生的錯誤 (例如 API 呼叫失敗, 金鑰無效, 超出配額等)
        logging.error(f"執行網路搜尋時發生錯誤: {e}", exc_info=True)
        return [] # 搜尋失敗，返回空列表


# --- 新增：建立 AI 判斷使用者意圖及關鍵詞的函式 ---
# 這個函式負責呼叫 AI 判斷意圖
def decide_intent_and_query(user_input: str, model_instance: Optional[genai.GenerativeModel]) -> dict:
    """
    發送 Prompt 給 AI 模型，請它判斷使用者問題的意圖和相關關鍵詞。

    Args:
        user_input (str): 使用者的原始輸入。
        model_instance: 已初始化好的 Gemini 模型實例，可能為 None。

    Returns:
        dict: 包含意圖 ('chat', 'explain', 'discuss') 和相關關鍵詞 ('term') 的字典。
              如果判斷失敗或模型不可用，返回默認意圖 {'intent': 'chat', 'term': user_input}。
    """

     # 定義在各種錯誤情況下返回的預設字典
    default_response = {
        'intent': 'chat',
        'term': user_input,
        'web_search_needed': False, # 預設為 False，除非 AI 明確指示
        'web_search_query': ""    # 預設為空
    }

    if model_instance is None:
        logging.error("AI 模型未初始化，無法判斷使用者意圖。")
        return default_response # 返回默認意圖

    # 設計一個 Prompt，要求 AI 分析問題並以特定 JSON 格式回應意圖和關鍵詞 (Industry Standard - 結構化輸出)
    # 這個 Prompt 的設計非常重要，需要清晰地指示 AI 的任務和期望的輸出格式
    decision_prompt = textwrap.dedent(f"""
    分析以下使用者問題，判斷其主要意圖屬於以下哪種類型：'chat' (普通聊天/通用問題), 'explain' (解釋一個具體的概念/技術名詞), 或 'discuss' (討論某個開放性主題/觀點/筆記內容)。
    同時，提取問題中最核心的、用於表示主題或概念的關鍵詞或短語。

    請以嚴格的 JSON 格式回應，只包含 JSON 內容，不要有額外的文字說明或符號 (例如 ```json)。
    JSON 應包含以下兩個鍵：
    "intent": 字串 ('chat', 'explain', 或 'discuss')
    "term": 字串 (問題中最重要的關鍵詞或概念名稱，用於後續檢索或作為討論主題)
    "web_search_needed": 布林值 (如果意圖是 'chat' 且問題暗示需要外部、即時或廣泛的資訊，則為 true；對於 'explain' 和 'discuss' 通常為 false，除非判斷現有知識庫極度缺乏該主題信息，則可考慮為 true)
    "web_search_query": 字串 (如果 web_search_needed 為 true，則提供一個簡潔有效的網路搜尋查詢詞；否則可以為空字串或相關關鍵詞)

    使用者問題：{user_input}

    JSON 回應：
    """)

    logging.info("發送意圖判斷 Prompt 給 AI 模型...")
    # print(f"DEBUG: Intent Decision Prompt:\n{decision_prompt}") # 調試用

    try:
        # 呼叫 Gemini 模型獲取判斷結果
        response = model_instance.generate_content(decision_prompt)
        logging.info("成功接收 AI 模型意圖判斷回應。")

        if response.text:
            json_string = response.text.strip()
            # 清理可能的 JSON 邊界符號
            if json_string.startswith("```json"):
                 json_string = json_string[7:].strip()
            if json_string.endswith("```"):
                 json_string = json_string[:-3].strip()

            try:
                decision_data = json.loads(json_string) # 解析 JSON 字串

                # 驗證 AI 返回的 JSON 是否符合預期結構
                required_keys = ["intent", "term", "web_search_needed", "web_search_query"]
                if not all(key in decision_data for key in required_keys):
                    logging.warning(f"AI 返回的 JSON 缺少必要的鍵: {response.text}。返回預設值。")
                    return default_response

                # 驗證 intent 的值是否在允許的範圍內
                valid_intents = ['chat', 'explain', 'discuss']
                if decision_data.get("intent") not in valid_intents:
                    logging.warning(f"AI 返回的 JSON 中 'intent' 值無效: {decision_data.get('intent')}。返回預設值。")
                    return default_response
                
                # 驗證 web_search_needed 是否為布林值
                if not isinstance(decision_data.get("web_search_needed"), bool):
                    logging.warning(f"AI 返回的 JSON 中 'web_search_needed' 值不是布林值: {decision_data.get('web_search_needed')}。返回預設值。")
                    # 你也可以嘗試轉換，例如 'true' -> True，但嚴格檢查更安全
                    return default_response

                # 確保 term 和 web_search_query 是字串
                if not isinstance(decision_data.get("term"), str) or \
                    not isinstance(decision_data.get("web_search_query"), str):
                    logging.warning(f"AI 返回的 JSON 中 'term' 或 'web_search_query' 不是字串。返回預設值。")
                    return default_response

                # 如果一切正常，返回 AI 的判斷結果
                # 確保返回的字典包含所有四個鍵，即使 AI 少給了某個（雖然上面的檢查已經涵蓋了）
                final_decision = {
                    'intent': decision_data.get('intent', default_response['intent']),
                    'term': decision_data.get('term', user_input), # 如果 AI 沒給 term，用 user_input
                    'web_search_needed': decision_data.get('web_search_needed', default_response['web_search_needed']),
                    'web_search_query': decision_data.get('web_search_query', default_response['web_search_query'])
                }
                # 如果 web_search_needed 為 False，但 web_search_query 不是空，可以選擇清空它
                if not final_decision['web_search_needed']:
                    final_decision['web_search_query'] = ""

                logging.info(f"AI 判斷結果：{final_decision}")
                return final_decision

            except json.JSONDecodeError:
                logging.error(f"無法解析 AI 返回的內容為 JSON: {response.text}。返回預設值。", exc_info=True)
                return default_response
        else:
            logging.warning("AI 模型未返回文字內容以進行意圖判斷。返回預設值。")
            return default_response

    except Exception as e:
        # 捕獲呼叫 model_instance.generate_content 時可能發生的任何其他錯誤
        logging.error(f"與 AI 模型互動進行意圖判斷時發生意外錯誤: {e}。返回預設值。", exc_info=True)
        return default_response

def load_and_chunk_document(file_path: str) -> list[Document]:
    """
    載入指定路徑的文件，並將其分割成較小的文字片段 (chunks)。

    Args:
        file_path (str): 文件的完整路徑。

    Returns:
        list[Document]: 分割後的文字片段列表，每個片段是一個 Document 對象。
                        如果文件類型不受支持或載入失敗，返回空列表。
    """
    if not os.path.exists(file_path):
        logging.error(f"文件不存在：{file_path}")
        return []

    # 根據文件副檔名選擇合適的載入器
    _, file_extension = os.path.splitext(file_path)
    loader = None
    if file_extension.lower() == '.txt':
        loader = TextLoader(file_path, encoding='utf-8')
        logging.info(f"使用 TextLoader 載入文件：{file_path}")
    elif file_extension.lower() == '.pdf':
        loader = PyPDFLoader(file_path)
        logging.info(f"使用 PyPDFLoader 載入文件：{file_path}")
    # 您可以添加更多文件類型的支持
    else:
        logging.warning(f"不支援的文件類型：{file_extension} (文件路徑: {file_path})")
        return []

    # 載入文件內容
    try:
        documents = loader.load()
        logging.info(f"成功載入文件：{file_path}，共 {len(documents)} 頁/部分。")
    except Exception as e:
        logging.error(f"載入文件 {file_path} 時發生錯誤: {e}", exc_info=True)
        return []

    # 初始化文字分割器
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # 每個片段的最大字元數
        chunk_overlap=200, # 相鄰片段之間重疊的字元數
        length_function=len,
        add_start_index=True
    )

    # 分割文件內容
    logging.info(f"開始分割文件內容...")
    try:
        chunks = text_splitter.split_documents(documents)
        logging.info(f"文件分割完成，共生成 {len(chunks)} 個片段 (chunks)。")
    except Exception as e:
        logging.error(f"分割文件 {file_path} 時發生錯誤: {e}", exc_info=True)
        return []

    return chunks


# --- 將向量資料庫初始化/載入封裝成函式 (處理持久化) ---
def initialize_vector_store(persist_directory: str, document_paths: list[str], google_api_key: str, embedding_model_name: str = DEFAULT_EMBEDDING_MODEL_NAME) -> Optional[Chroma]:
    """
    初始化或載入向量資料庫。如果持久化目錄存在則載入，否則處理文件並創建。

    Args:
        persist_directory (str): 向量資料庫儲存的目錄。
        document_paths (list[str]): 需要處理的文件路徑列表。
        google_api_key (str): Google API 金鑰，用於嵌入模型。
        embedding_model_name (str): 用於生成嵌入的模型名稱。

    Returns:
        Optional[Chroma]: 初始化或載入成功的 Chroma 向量資料庫實例，失敗則返回 None。
    """
    logging.info(f"嘗試初始化向量資料庫。持久化目錄: {persist_directory}")

    # 確保有 API 金鑰來創建嵌入對象
    if not google_api_key:
         logging.error("缺少 Google API 金鑰，無法創建嵌入模型或向量資料庫。")
         return None

    try:
         # 準備 LangChain 嵌入模型對象
         logging.info(f"準備使用嵌入模型 '{embedding_model_name}' 創建 LangChain Embedding 對象...")
         langchain_gemini_embeddings = GoogleGenerativeAIEmbeddings(
             model=embedding_model_name,
             google_api_key=google_api_key # 傳遞 API 金鑰
         )
         logging.info("成功創建 LangChain 的 GoogleGenerativeAIEmbeddings 對象。")

         # --- 判斷是創建新的資料庫還是載入現有的 ---
         if os.path.exists(persist_directory):
             # 如果儲存目錄已存在，則載入現有的向量資料庫
             logging.info(f"載入現有的 Chroma 向量資料庫從目錄: {persist_directory}")
             # 載入時也需要提供相同的嵌入對象，確保兼容性
             vectorstore = Chroma(persist_directory=persist_directory, embedding_function=langchain_gemini_embeddings)
             logging.info("成功載入現有的 Chroma 向量資料庫。")
             return vectorstore

         else:
             # 如果儲存目錄不存在，則需要處理文件並創建新的資料庫
             logging.warning(f"儲存目錄 '{persist_directory}' 不存在。開始處理文件並創建新的資料庫。")

             all_chunks = []
             for doc_path in document_paths:
                 if os.path.exists(doc_path):
                     logging.info(f"處理文件: {doc_path}")
                     chunks = load_and_chunk_document(doc_path) # 使用之前的 load_and_chunk_document 函式
                     all_chunks.extend(chunks)
                     logging.info(f"文件 {doc_path} 處理完成，生成 {len(chunks)} 個片段。總片段數: {len(all_chunks)}")
                 else:
                     logging.warning(f"文件不存在，跳過處理：{doc_path}")


             if all_chunks:
                 # 建立並填充向量資料庫並儲存到磁碟
                 logging.info(f"開始創建並填充 Chroma 向量資料庫到目錄: {persist_directory}，共 {len(all_chunks)} 個片段...")
                 vectorstore = Chroma.from_documents(
                     documents=all_chunks, # 所有文件片段列表
                     embedding=langchain_gemini_embeddings, # 嵌入對象
                     persist_directory=persist_directory # 指定儲存目錄
                 )
                 logging.info(f"成功創建並填充 Chroma 向量資料庫到目錄: {persist_directory}。")

                 # 持久化儲存 (確保資料寫入磁碟)
                 vectorstore.persist()
                 logging.info("向量資料庫已持久化儲存。")
                 return vectorstore

             else:
                 logging.warning("沒有找到有效的文檔可供處理，未能創建向量資料庫。")
                 logging.warning("\n沒有找到有效的文檔，未能創建向量資料庫。請確認文件路徑是否正確。")
                 return None # 沒有片段，無法創建資料庫


    except ImportError:
         logging.error("需要安裝 'langchain-google-genai' 和 'chromadb' 函式庫才能初始化向量資料庫。")
         logging.error("請運行指令：python -m pip install langchain-google-genai chromadb")
         return None
    except Exception as e:
        logging.error(f"初始化向量資料庫時發生錯誤: {e}", exc_info=True)
        return None # 初始化失敗返回 None


# --- 建立處理單一文件並加入現有向量資料庫的函式 ---
def process_and_add_to_vector_store(file_path: str, vectorstore: Chroma, embedding_model_name: str, google_api_key: str) -> bool:
    """
    載入、分割指定文件，並將其內容添加到現有的向量資料庫中。

    Args:
        file_path (str): 需要處理的文件的完整路徑。
        vectorstore (Chroma): 已初始化好的 Chroma 向量資料庫實例。
        embedding_model_name (str): 用於生成嵌入的模型名稱。
        google_api_key (str): Google API 金鑰，用於嵌入模型。

    Returns:
        bool: 如果成功處理並添加到資料庫，返回 True，否則返回 False。
    """
    # 函數內部首先進行基本檢查
    if vectorstore is None:
        logging.error("向量資料庫未初始化，無法添加文件。")
        return False
    if not os.path.exists(file_path):
        logging.error(f"文件不存在：{file_path}，無法添加。")
        return False
    if not google_api_key:
         logging.error("缺少 Google API 金鑰，無法創建嵌入模型或添加到向量資料庫。")
         return False

    logging.info(f"開始處理文件 '{file_path}' 並添加到向量資料庫...")

    try:
        # 1. 載入並分割文件 (使用之前定義的函式)
        # 這個函式內部已經處理了支援的文件類型，不支援的會返回空列表
        document_chunks = load_and_chunk_document(file_path)

        if not document_chunks:
            logging.warning(f"文件 '{file_path}' 未能生成任何文件片段，跳過添加。請確認文件內容及類型是否正確。")
            return False # 沒有片段，視為失敗

        # 2. 準備嵌入模型對象 (與初始化向量庫時使用相同的模型和金鑰)
        logging.info(f"準備使用嵌入模型 '{embedding_model_name}' 創建 LangChain Embedding 對象用於添加新文檔...")
        langchain_gemini_embeddings = GoogleGenerativeAIEmbeddings(
            model=embedding_model_name,
            google_api_key=google_api_key
        )
        logging.info("成功創建 LangChain 的 GoogleGenerativeAIEmbeddings 對象用於添加。")

        # 3. 將文件片段添加到現有的向量資料庫
        # vectorstore.add_documents 方法可以將新的 Document 對象列表添加到現有資料庫
        # 它會自動為新文檔生成嵌入，並儲存到 Chroma 中
        logging.info(f"開始添加 {len(document_chunks)} 個文件片段到向量資料庫...")
        vectorstore.add_documents(
            documents=document_chunks,
            embedding=langchain_gemini_embeddings # 提供嵌入對象，它會自動為新文檔生成嵌入
        )
        logging.info(f"成功添加文件片段到向量資料庫。")

        # 4. 持久化儲存變更 (非常重要！)
        # 如果資料庫是持久化的，添加文檔後需要呼叫 persist() 方法來保存變更
        # 檢查 vectorstore 是否有 persist 方法 (內存資料庫可能沒有)
        if hasattr(vectorstore, 'persist') and callable(vectorstore.persist):
            vectorstore.persist()
            logging.info("向量資料庫已持久化變更到磁碟。")
        else:
            logging.info("向量資料庫不支援持久化或未配置持久化路徑，變更僅在運行期間有效。")


        return True # 成功處理並添加

    except Exception as e:
        logging.error(f"處理文件 '{file_path}' 並添加到向量資料庫時發生錯誤: {e}", exc_info=True)
        return False # 處理失敗


# --- 新增：建立語音轉文字 (STT) 函式 ---
def transcribe_audio(audio_file_path: str, openai_client: Optional[openai.OpenAI]) -> Optional[str]: # 接收 client 實例
    """
    使用 OpenAI Whisper 模型將音頻檔案轉錄為文字。

    Args:
        audio_file_path (str): 音頻檔案的完整路徑。
        openai_api_key (Optional[str]): OpenAI API 金鑰。

    Returns:
        Optional[str]: 轉錄的文字，如果失敗或無效則返回 None。
    """
    if not openai_client:
        logging.error("OpenAI API 金鑰未提供，無法執行語音轉文字。")
        return None
    if not os.path.exists(audio_file_path):
        logging.error(f"音頻檔案不存在：{audio_file_path}")
        return None

    logging.info(f"開始將音頻檔案 '{os.path.basename(audio_file_path)}' 轉錄為文字...")

    try:
        # 打開音頻檔案以二進制讀取模式 ('rb')
        with open(audio_file_path, "rb") as audio_file:
            # 呼叫 OpenAI 音頻轉錄 API
            transcription = openai_client.audio.transcriptions.create(
              model=DEFAULT_WHISPER_MODEL_NAME, # 使用 Whisper 模型
              file=audio_file,
              # response_format="text" # 可選：指定返回純文本
              language="zh" # 可選：指定語言，有助於提高轉錄中文的準確性
            )

        # 轉錄結果通常在 transcription.text 中
        transcribed_text = transcription.text
        logging.info(f"語音轉文字成功。轉錄文本 (前 100 字元): {transcribed_text[:100]}...")
        return transcribed_text

    except openai.APIError as e:
        logging.error(f"OpenAI API 錯誤 (STT): {e}", exc_info=True)
        # 在核心模組避免直接 print 給終端機使用者或 API 調用者
        return None # 返回 None 表示失敗
    except FileNotFoundError:
        logging.error(f"音頻檔案未找到或無法打開：{audio_file_path}")
        return None
    except Exception as e:
        logging.error(f"執行語音轉文字時發生意外錯誤: {e}", exc_info=True)
        return None # 返回 None 表示失敗

# --- 將核心 RAG 回答邏輯封裝成函式 ---
def get_ai_response(user_input: str, model: genai.GenerativeModel, vectorstore: Optional[Chroma], search_api_key: Optional[str], search_engine_id: Optional[str]) -> str:
    """
    根據使用者輸入，結合個人知識庫和網路搜尋，生成 AI 的回答。

    Args:
        user_input (str): 使用者的問題或指令。
        model: 已初始化好的 Gemini 模型實例。
        vectorstore (Optional[Chroma]): 已初始化好的個人知識庫向量資料庫實例，可能為 None。
        search_api_key (Optional[str]): Google Search API 金鑰，可能為 None。
        search_engine_id (Optional[str]): Google Search Engine ID，可能為 None。

    Returns:
        str: AI 生成的回答文字。
    """
    if model is None:
        return "錯誤：AI 模型未初始化，無法生成回答。" # 安全檢查

    logging.info(f"處理使用者輸入：{user_input}")

    # 直接呼叫新的意圖判斷函式
    intent_data = decide_intent_and_query(user_input, model) # <--- 呼叫新的函式
    primary_intent = intent_data.get('intent', 'chat') # 獲取判斷的意圖，默認為 chat
    relevant_term = intent_data.get('term', user_input) # 獲取判斷的關鍵詞，默認為原始輸入
    search_needed = intent_data.get('web_search_needed', False) # 從 AI 判斷結果獲取
    web_search_query = intent_data.get('web_search_query', relevant_term or user_input) # 從 AI 判斷結果獲取

    # --- 個人知識庫檢索 ---
    personal_context = None
    retrieved_personal_docs = []

    # 只有當 vectorstore 成功創建或載入時才執行檢索
    if vectorstore is not None:
        # 根據判斷的意圖和關鍵詞調整檢索查詢詞
        # 如果是 explain 或 discuss，優先使用 AI 提取的關鍵詞作為檢索詞
        # 如果是 chat，使用原始使用者輸入作為檢索詞
        # 如果 AI 判斷的關鍵詞是空字串，回退到使用原始輸入
        retrieval_query = relevant_term if relevant_term and primary_intent in ['explain', 'discuss'] else user_input
        # 確保檢索查詢詞不是空的
        if not retrieval_query.strip():
             retrieval_query = user_input # 避免空查詢

        logging.info(f"從個人知識庫檢索與查詢 '{retrieval_query}' (意圖: {primary_intent}) 相關的片段...")
        try:
            # 在向量資料庫中進行相似度搜尋以檢索相關的個人知識片段
            # 檢索更多相關片段，以獲取更全面的概念信息
            retrieved_personal_docs = vectorstore.similarity_search(retrieval_query, k=5) # 檢索最相似的 5 個片段
            if retrieved_personal_docs:
                logging.info(f"從個人知識庫檢索到 {len(retrieved_personal_docs)} 個相關片段。")
                personal_context_parts = [doc.page_content for doc in retrieved_personal_docs]
                personal_context = "\n---\n".join(personal_context_parts) # 使用分隔符連接
                logging.info("已將個人知識片段格式化為上下文。")
            else:
                logging.info("從個人知識庫未能檢索到相關片段。")

        except Exception as e:
             logging.error(f"執行個人知識庫檢索時發生錯誤: {e}", exc_info=True)
             #print("AI 助理：執行個人知識庫檢索時發生錯誤。") # 在 API 模式下避免直接 print 給使用者
             personal_context = None # 確保在發生錯誤時上下文為 None


    else:
        logging.warning("向量資料庫未初始化，跳過個人知識庫檢索。")


    # --- 使用 AI 判斷是否需要網路搜尋及獲取查詢詞 ---
    
    web_search_context = None # 初始化網路搜尋結果的上下文

    if search_needed: # 直接使用 AI 的判斷
        logging.info(f"根據 AI 判斷，執行網路搜尋，查詢詞：'{web_search_query}'")
        # 確保 SEARCH_API_KEY 和 SEARCH_ENGINE_ID 已傳入 get_ai_response 或可從 config 獲取
        if not search_api_key or not search_engine_id: # 假設這些是 get_ai_response 的參數
            logging.warning("無法執行網路搜尋，因為缺少搜尋 API 金鑰或 CX ID。")
        else:
            search_results = perform_web_search(web_search_query, search_api_key, search_engine_id, num_results=3)
            if search_results:
                logging.info(f"搜尋到 {len(search_results)} 個結果。")
                context_parts = []
                for i, item in enumerate(search_results):
                    title = item.get('title', 'N/A')
                    link = item.get('link', 'N/A')
                    snippet = item.get('snippet', 'N/A')
                    context_parts.append(textwrap.dedent(f"""
                    Web Source {i+1}:
                    Title: {title}
                    URL: {link}
                    Snippet: {snippet}
                    """).strip())
                web_search_context = "\n---\n".join(context_parts)
                logging.info("已將網路搜尋結果格式化為上下文。")
            else:
                logging.info("網路搜尋未找到相關結果。")
    else:
        logging.info("根據 AI 判斷，不需要執行網路搜尋。")


    # --- 構造最終發送給 AI 的 Prompt (整合 個人知識庫 + 網路搜尋 + 原始問題 + 模式指令) ---
    final_prompt_parts = []

    # 提供核心指令 - 指導 AI 如何使用上下文並設定角色 (根據主要意圖調整指令)
    if primary_intent == 'explain':
        # 在解釋概念模式下使用特定的 Prompt 指令
        # 使用 AI 判斷的關鍵詞作為概念名稱
        final_prompt_parts.append(textwrap.dedent(f"""
        請扮演一位資深的 AI 軟體工程師和專業的面試準備輔導員。
        你的任務是根據提供的參考資訊（如果有的話），以**清晰、有條理、針對面試考點的方式**，深入解釋概念「{relevant_term}」。
        主要基於「個人知識庫」中的內容進行解釋。如果個人知識庫資訊不足，可參考「網路搜尋結果」。如果所有參考資訊都不足以解釋，請用通用知識。
        回答應該包含概念的定義、原理、關鍵要素、優缺點等，適合面試回答的風格，精簡且精確。使用流暢的中文。

        以下是參考資訊：
        """).strip())
        logging.info("構造解釋概念 Prompt。")

    elif primary_intent == 'discuss':
        # 在討論模式下使用特定的 Prompt 指令
        # 討論主題可以基於 AI 判斷的關鍵詞或原始輸入
        final_prompt_parts.append(textwrap.dedent(f"""
        請扮演一個擁有使用者所有知識概念的 AI 討論夥伴。
        你的任務是根據提供的「個人知識庫參考」內容，與使用者就相關主題進行**深入的、發人深省的互動和討論**。相關主題可能是關於「{relevant_term}」或使用者問題「{user_input}」。

        回答時請：
        1. **優先且重點參考「個人知識庫」**中的內容。分析其中的觀點、論證和可能的關聯。
        2. **以鼓勵討論的方式回應**。可以總結使用者的觀點，指出其中的關鍵點。
        3. **提出相關的、開放性的問題**來引導使用者更深入地思考或從新的角度看待問題。這些問題應該基於個人知識庫的內容或與之相關。
        4. 如果個人知識庫資訊不足，可以參考「網路搜尋結果」（如果有的話）補充背景知識。
        5. 避免給出終結性的答案，而是保持討論的流動性。
        6. 使用流暢的中文。

        以下是參考資訊：
        """).strip())
        logging.info("構造討論夥伴 Prompt。")

    else: # 默認為 'chat' 意圖
        # 在標準聊天模式下使用之前的 Prompt 指令
        final_prompt_parts.append(textwrap.dedent("""
        請扮演一個經驗豐富的 AI 軟體工程師，同時也是一位樂於助人的面試準備輔導員。
        你的任務是根據提供的參考資訊（如果有的話），精準、清晰、有條理、針對面試考點地回答或解釋使用者關於 AI、機器學習、深度學習、演算法、資料結構、系統設計、個人專案等方面的問題。

        回答時請嚴格遵循以下優先順序和原則：
        1.  最優先且深度參考「個人知識庫」。... (同之前標準 Prompt 的第 1-5 點) ...
        以下是參考資訊：
        """).strip())
        logging.info("構造標準聊天 Prompt。")


    # 2. 加入個人知識庫上下文 (如果檢索到了或狀態提示) (這部分代碼保持原樣)
    if personal_context:
        final_prompt_parts.append(textwrap.dedent(f"""
        --- 個人知識庫參考 ---
        {personal_context}
        """).strip())
    elif vectorstore is not None and not retrieved_personal_docs:
        final_prompt_parts.append(textwrap.dedent("""
        --- 個人知識庫參考 ---
        (未從個人知識庫檢索到與問題相關的筆記)
        """).strip())
    elif vectorstore is None:
         final_prompt_parts.append(textwrap.dedent("""
        --- 個人知識庫參考 ---
        (個人知識庫功能未啟用或未成功載入)
        """).strip())

    # 2. 加入個人知識庫上下文 (如果檢索到了或狀態提示) (這部分代碼保持原樣)
    if personal_context:
        final_prompt_parts.append(textwrap.dedent(f"""
        --- 個人知識庫參考 ---
        {personal_context}
        """).strip())
    elif vectorstore is not None and not retrieved_personal_docs:
        final_prompt_parts.append(textwrap.dedent("""
        --- 個人知識庫參考 ---
        (未從個人知識庫檢索到與問題相關的筆記)
        """).strip())
    elif vectorstore is None:
         final_prompt_parts.append(textwrap.dedent("""
        --- 個人知識庫參考 ---
        (個人知識庫功能未啟用或未成功載入)
        """).strip())

    # 3. 加入網路搜尋上下文 (如果網路搜尋被觸發且有結果或狀態提示)
    # 只有在主要意圖是 'chat' 且 AI 判斷需要網路搜尋時，web_search_context 才會有內容
    if web_search_context:
        final_prompt_parts.append(textwrap.dedent(f"""
        --- 網路搜尋結果參考 ---
        {web_search_context}
        """).strip())
    # 這裡不再需要像之前那樣根據 search_needed 和 !web_search_context 添加無結果提示
    # 因為只有在 chat 模式下才會觸發 web search，而 chat 模式使用標準 Prompt，
    # 標準 Prompt 已經包含了無結果回退到通用知識的指令。


    # 4. 加入使用者原始問題
    final_prompt_parts.append(textwrap.dedent(f"""
        --- 使用者問題 ---
        {user_input}
        """).strip())

    # 將所有部分連接起來形成最終 Prompt
    final_prompt = "\n\n".join(final_prompt_parts)

    # print(f"DEBUG: Final Prompt:\n{final_prompt}") # 調試用

    # --- 發送最終 Prompt 給 AI 模型並處理回應 ---
    # ... (The rest of the get_ai_response function code remains unchanged) ...
    if model is not None and final_prompt.strip():
        try:
            logging.info("發送最終 Prompt 給 AI 模型 (串流模式)...")
            # 當 stream=True, response 是一個可迭代物件
            response_iterable = model.generate_content(final_prompt, stream=True)

            collected_text_parts = []
            for chunk in response_iterable:
                # 每個 chunk 通常是一個 GenerateContentResponse 物件，代表數據流的一部分
                # 它可以包含文本、候選內容等。chunk.text 是獲取該塊文本的便捷方式。
                try:
                    if chunk.text:
                        collected_text_parts.append(chunk.text)
                except ValueError as ve: # 有時 chunk.text 可能因內容過濾而引發 ValueError
                    logging.warning(f"處理串流中的一個區塊時，訪問 .text 出錯: {ve}")
                    # 你可能需要檢查 chunk.parts 和 chunk.candidates[0].finish_reason
                    # 如果是因安全原因被阻止，finish_reason 可能是 "SAFETY"
                    if chunk.candidates and chunk.candidates[0].finish_reason == "SAFETY":
                        logging.error("串流因安全原因被終止。")
                        # 你可能需要從 chunk.candidates[0].safety_ratings 獲取更詳細的原因
                        # 這裡我們直接返回一個通用錯誤訊息
                        # 安全回饋通常在整個 response_iterable.prompt_feedback 中，或在最後一個 chunk 中
                        safety_feedback_message = "由於內容安全政策，請求的部分內容無法顯示。"
                        try:
                            if response_iterable.prompt_feedback and response_iterable.prompt_feedback.block_reason:
                                safety_feedback_message = response_iterable.prompt_feedback.block_reason_message
                        except Exception:
                            pass # 獲取 feedback 失敗，使用通用訊息
                        return f"抱歉，{safety_feedback_message}"
                    continue # 繼續處理下一個 chunk，或者你可以選擇在此處中斷

            final_response_text = "".join(collected_text_parts)
            logging.info("AI 模型串流回應已收集完成。")

            if final_response_text:
                return final_response_text
            else:
                # 如果收集到的文本為空，檢查是否有 prompt_feedback
                # prompt_feedback 可能在迭代完成後的 response_iterable 物件上
                block_reason_message = "AI 未返回任何內容。"
                try:
                    if response_iterable.prompt_feedback and response_iterable.prompt_feedback.block_reason:
                        block_reason_message = response_iterable.prompt_feedback.block_reason_message
                        logging.warning(f"AI 回應（串流）被阻止，原因：{block_reason_message}")
                        return f"抱歉，由於內容安全政策，您的請求無法完成 ({block_reason_message})。"
                except AttributeError: # response_iterable 可能沒有 prompt_feedback 屬性直到迭代完成或特定情況
                    logging.warning("無法在串流回應迭代器上立即獲取 prompt_feedback。")
                except Exception as e_feedback:
                    logging.warning(f"獲取串流回應的 prompt_feedback 時出錯: {e_feedback}")
                
                logging.error(f"AI 模型串流處理後未生成有效文字內容。({block_reason_message})")
                return f"抱歉，未能獲得有效回應。({block_reason_message})"
        
        except genai.types.BlockedPromptException as bpe: # 捕獲 Prompt 被阻止的特定異常
            logging.error(f"由於 Prompt 被阻止，無法生成回答 (串流模式): {bpe}", exc_info=True)
            # response.prompt_feedback 可能包含更多信息
            block_reason_msg = "您的請求因安全原因被阻止"
            try:
                if bpe.response and bpe.response.prompt_feedback: # BlockedPromptException 可能帶有 response 物件
                    block_reason_msg = bpe.response.prompt_feedback.block_reason_message or block_reason_msg
            except AttributeError:
                pass
            return f"抱歉，{block_reason_msg}，無法處理。"
        except Exception as e:
            logging.error(f"與 AI 模型互動生成最終回答時發生錯誤 (串流模式): {e}", exc_info=True)
            return f"抱歉，生成回答時發生內部錯誤：{e}"

    elif model is None:
        logging.error("AI 模型未初始化，無法生成最終回答。")
        return "抱歉，AI 模型未準備好，無法生成回答。"
    else: # Prompt is empty or only whitespace
         logging.warning("構造的最終 Prompt 為空，跳過發送給 AI。")
         return "" # 返回空字串或特定提示


# --- 主要應用程式入口點 --- (保留這個區塊，但修改其內容)
if __name__ == "__main__":

    # --- 載入環境變數 ---
    config = load_env_variables() # 接收設定字典

    # --- 初始化核心組件 (AI 模型 和 向量資料庫) ---
    print("--- 應用程式啟動初始化 ---")
    # 使用 config 中的金鑰初始化模型
    model = None
    if config.get('GOOGLE_API_KEY'):
        model = initialize_gemini_model(config['GOOGLE_API_KEY'])
    else:
        logging.error("無法初始化 Gemini 模型，因為 GOOGLE_API_KEY 未設定。")
        # 根據你的應用程式邏輯，這裡可以決定是否 sys.exit

    # 初始化或載入向量資料庫
    # 需要提供文件路徑列表。這裡硬編碼一個測試文件路徑。
    # 未來可以擴展從配置文件讀取文件列表或支持文件上傳後動態更新
    document_paths_to_process = ["my_notes.txt"] # <--- 設定您個人文件的路徑列表

    if config.get('GOOGLE_API_KEY'):
        vectorstore = initialize_vector_store(
            PERSIST_DIRECTORY,
            document_paths_to_process,
            config['GOOGLE_API_KEY'] # 傳遞金鑰
        )
    else:
        logging.error("無法初始化向量資料庫，因為 GOOGLE_API_KEY 未設定。")
        
    # 確保核心組件初始化成功
    if model is None:
        # 讓主程式決定是否因模型初始化失敗而退出
        # sys.exit("應用程式啟動失敗：AI 模型未成功初始化。")
        logging.critical("AI 模型未成功初始化，某些功能可能無法使用或應用程式可能無法正常運行。")

    # 向量資料庫可能為 None，如果初始化失敗或沒有文件
    print("--- 應用程式啟動初始化完成 ---")
    print("-" * 40)

    # --- 新增：語音轉文字 (STT) 功能測試區塊 ---
    print("\n--- 運行語音轉文字測試 ---")
    # 請準備一個小型音頻檔案 (.mp3, .wav 等)，放在與 ai_assistant_core.py 同一個目錄下
    # 並修改這裡的路徑指向您的測試檔案
    test_audio_file = "test_audio_2.mp3" # <--- 修改為您的測試音頻檔案名稱和副檔名，例如 "my_voice_record.wav"

    # 只有當 OpenAI API 金鑰已設定且測試檔案存在時才運行測試
    # OPENAI_API_KEY 已經在 load_env_variables 中載入
    openai_client = None # 初始化為 None
    if config.get('OPENAI_API_KEY'):
        try:
            openai_client = openai.OpenAI(api_key=config.get('OPENAI_API_KEY')) # 初始化一次
            logging.info("OpenAI 客戶端初始化成功。")
        except Exception as e:
            logging.error(f"初始化 OpenAI 客戶端失敗: {e}")
    else:
        logging.warning("跳過語音轉文字測試：缺少 OPENAI_API_KEY。")

    if openai_client:
        if not os.path.exists(test_audio_file):
            logging.warning(f"跳過語音轉文字測試：測試音頻檔案不存在：{test_audio_file}")
            print(f"跳過語音轉文字測試，因為測試音頻檔案 '{test_audio_file}' 不存在。")
            print("請準備一個音頻檔案並修改 test_audio_file 變數指向它。")
        else:
            transcribed_text = transcribe_audio(test_audio_file, openai_client) # 傳遞 client 實例
            if transcribed_text:
                print(f"\n語音轉文字測試成功。轉錄結果：")
                print(transcribed_text)
            else:
                print("\n語音轉文字測試失敗。請查看上面的日誌獲取詳細錯誤。")
    else:
        logging.warning("跳過語音轉文字測試：OpenAI 客戶端未初始化 (可能缺少 API 金鑰或初始化失敗)。")
    
    logging.info("--- 語音轉文字測試結束 ---")
    print("-" * 40)

    # --- 主要對話迴圈 (CLI 模式) ---
    # 我們現在將 get_ai_response 函式應用到 CLI 對話中
    print("--- 個人 AI 助理 (CLI 模式，整合個人知識庫與網路搜尋 RAG) ---")
    print("輸入您的問題或指令，輸入 'exit', 'quit' 結束。")

    # 根據 vectorstore 是否可用，提示個人知識庫狀態
    if vectorstore is not None:
        try:
            count = vectorstore._collection.count()
            print(f"個人知識庫已載入/創建，包含約 {count} 個片段。")
        except Exception:
             print("個人知識庫已載入/創建。")
    else:
        print("個人知識庫功能未啟用，因為向量資料庫未成功創建或載入。")

    print("-" * 40)

    while True:
        try:
            user_input = input("您：")
            logging.info(f"使用者輸入：{user_input}")

        except (EOFError, KeyboardInterrupt):
            print("\nAI 助理：接收到結束訊號或中斷指令，再見！")
            logging.info("接收到終端機中斷訊號，程式結束。")
            break
        except Exception as e:
             logging.error(f"獲取使用者輸入時發生意外錯誤: {e}", exc_info=True)
             print("AI 助理：獲取您的輸入時發生問題，請重試。")
             continue

        # 檢查是否退出指令
        if user_input.lower() in ['exit', 'quit']:
            print("AI 助理：再見！")
            logging.info("使用者要求退出，程式正常結束。")
            break

        # 忽略空白輸入
        if not user_input.strip():
            continue

        # --- 呼叫核心 RAG 回答函式 ---
        # 將使用者輸入、模型、向量庫、搜尋金鑰都傳遞給函式
        # 函式內部處理檢索、搜尋判斷、Prompt 構造和 AI 呼叫
        ai_response_text = get_ai_response(
            user_input,
            model, # 傳遞已初始化的模型,
            vectorstore,
            config.get('SEARCH_API_KEY'), # 從 config 獲取
            config.get('SEARCH_ENGINE_ID') # 從 config 獲取
            # 注意：decide_intent_and_query 和 perform_web_search 也需要 API keys
            # get_ai_response 內部呼叫這些函式時，也需要將相關的 key 傳遞下去，或將 model/config 作為參數傳遞
        )
        # 顯示 AI 的回應 (直接使用 get_ai_response 返回的文本)
        if ai_response_text:
             # 在 CLI 模式下，格式化輸出
             formatted_text = textwrap.fill(ai_response_text, width=80, replace_whitespace=False)
             print("AI 助理：")
             print(formatted_text)
        # else: get_ai_response 在失敗時已經返回錯誤訊息，成功但無文本的情況也處理了
        print("-" * 40) # 對話分隔線

    logging.info("應用程式已結束。")