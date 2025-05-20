# ai_assistant_core.py (建議將原文件改名，以區別核心邏輯和 API/UI 入口)
# 個人 AI 助理核心邏輯函式庫

import asyncio
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
# from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import PGVector  # <--- 新增 PGVector 導入
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from typing import AsyncIterable, Optional # 用於類型提示 Optional
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
DEFAULT_GEMINI_MODEL_NAME = 'gemini-1.5-flash-latest'
DEFAULT_EMBEDDING_MODEL_NAME = 'models/text-embedding-004'
DEFAULT_WHISPER_MODEL_NAME = 'whisper-1'
PGVECTOR_COLLECTION_NAME = "ai_assistant_document_vectors" # 給你的向量集合取個名字


# --- 使用者興趣的關鍵詞列表 --- 
USER_AI_INTERESTS = ["AI", "LLM", "ChatGPT", "Gemini", "Google AI"] 

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

    config['DATABASE_URL'] = os.environ.get('DATABASE_URL')
    if not config['DATABASE_URL']:
        logging.warning("DATABASE_URL 環境變數未設定。資料庫功能將無法使用。")
    
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
async def decide_intent_and_query(user_input: str, model_instance: Optional[genai.GenerativeModel]) -> dict:
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
        response = await model_instance.generate_content_async(decision_prompt)
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
        else: # response.text 為空
            # 檢查是否有因安全原因被阻止
            if response.prompt_feedback and response.prompt_feedback.block_reason:
                logging.warning(f"意圖判斷 Prompt 被阻止，原因: {response.prompt_feedback.block_reason_message}")
            else:
                logging.warning("AI 模型未返回文字內容以進行意圖判斷。返回預設值。")
            return default_response

    except genai.types.BlockedPromptException as bpe:
        logging.error(f"意圖判斷 Prompt 被阻止 (BlockedPromptException): {bpe}", exc_info=False)
        return default_response # 或者返回一個包含錯誤指示的特定回應
    except Exception as e:
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
def initialize_vector_store(db_connection_string: str, document_paths: list[str], google_api_key: str, embedding_model_name: str = DEFAULT_EMBEDDING_MODEL_NAME) -> Optional[PGVector]:
    """
    初始化 PGVector 向量資料庫。如果提供了文件路徑，則處理並添加文件。

    Args:
        db_connection_string (str): 向量資料庫儲存的目錄。
        document_paths (list[str]): 需要處理的文件路徑列表。
        google_api_key (str): Google API 金鑰，用於嵌入模型。
        embedding_model_name (str): 用於生成嵌入的模型名稱。
    """
    logging.info(f"嘗試初始化 PGVector 向量資料庫。集合名稱: {PGVECTOR_COLLECTION_NAME}")


    # 確保有 API 金鑰來創建嵌入對象
    if not google_api_key:
        logging.error("缺少 Google API 金鑰，無法創建嵌入模型或向量資料庫。")
        return None
    
    if not db_connection_string:
        logging.error("缺少資料庫連接字串，無法初始化 PGVector。")
        return None
    
    try:
        # 準備 LangChain 嵌入模型對象
        logging.info(f"準備使用嵌入模型 '{embedding_model_name}' 創建 LangChain Embedding 對象...")
        langchain_gemini_embeddings = GoogleGenerativeAIEmbeddings(
            model=embedding_model_name,
            google_api_key=google_api_key # 傳遞 API 金鑰
        )
        logging.info("成功創建 LangChain 的 GoogleGenerativeAIEmbeddings 對象。")

        # 初始化 PGVector store
        # 它會連接到你的 PostgreSQL 資料庫，並使用指定的集合名稱（類似於表名）
        # 如果集合不存在，它通常會自動創建
        vectorstore = PGVector(
            connection_string=db_connection_string,
            embedding_function=langchain_gemini_embeddings,
            collection_name=PGVECTOR_COLLECTION_NAME,
            # distance_strategy=DistanceStrategy.COSINE, # 可以指定距離策略，預設是 COSINE
        )
        logging.info(f"PGVector 實例已初始化。連接到資料庫並使用集合 '{PGVECTOR_COLLECTION_NAME}'。")
        
        # 可選：檢查集合是否真的存在或是否需要手動調用 create_collection
        # 通常，add_documents 會觸發集合的創建（如果不存在）
        # vectorstore.create_collection() # 如果需要顯式創建
        
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
            logging.info(f"開始添加 {len(all_chunks)} 個初始文件片段到 PGVector 集合 '{PGVECTOR_COLLECTION_NAME}'...")
            vectorstore.add_documents(all_chunks)
            logging.info("成功添加初始文件片段到 PGVector。")

        else:
            logging.info("沒有提供初始文件路徑，或文件未找到/為空，跳過初始文件添加。")

        return vectorstore

    except ImportError:
         logging.error("需要安裝 'langchain-google-genai' 和 'pgvector' 函式庫才能初始化向量資料庫。")
         logging.error("請運行指令：python -m pip install langchain-google-genai pgvector")
         return None
    except Exception as e:
        logging.error(f"初始化 PGVector 向量資料庫時發生錯誤: {e}", exc_info=True)
        return None # 初始化失敗返回 None


# --- 建立處理單一文件並加入現有向量資料庫的函式 ---
def process_and_add_to_vector_store(file_path: str, vectorstore: PGVector) -> bool:
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

    logging.info(f"開始處理文件 '{os.path.basename(file_path)}' 並添加到 PGVector...")

    try:
        document_chunks = load_and_chunk_document(file_path) # load_and_chunk_document 保持不變

        if not document_chunks:
            logging.warning(f"文件 '{os.path.basename(file_path)}' 未能生成任何文件片段，跳過添加。")
            return False

        logging.info(f"開始添加 {len(document_chunks)} 個文件片段到 PGVector...")
        vectorstore.add_documents(documents=document_chunks) # 使用 PGVector 的 add_documents
        logging.info(f"成功添加文件片段到 PGVector。PGVector 會自動處理持久化到 PostgreSQL。")
        return True

    except Exception as e:
        logging.error(f"處理文件 '{os.path.basename(file_path)}' 並添加到 PGVector 時發生錯誤: {e}", exc_info=True)
        return False


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


# --- 新增：建立生成 AI 新聞 Newsletter 的函式 ---
async def generate_ai_newsletter(interests: list[str], model: Optional[genai.GenerativeModel], search_api_key: Optional[str], search_engine_id: Optional[str]) -> tuple[Optional[str], list]:
    """
    根據使用者興趣，獲取近期 AI 新聞，並使用 AI 總結成 Newsletter 格式。

    Args:
        interests (list[str]): 使用者感興趣的 AI 主題關鍵詞列表。
        model (Optional[genai.GenerativeModel]): 已初始化好的 Gemini 模型實例，用於總結。
        search_api_key (Optional[str]): Google Search API 金鑰。
        search_engine_id (Optional[str]): Google Search Engine ID。

    Returns:
        Optional[str]: AI 生成的 Newsletter 文字，如果失敗或無內容則返回 None。
                       如果缺少必要金鑰或無新聞，返回描述性字符串。
    """
    # 使用連結作為唯一標識進行去重
    unique_search_results = []
    # 檢查必要條件
    if model is None:
        logging.error("AI 模型未初始化，無法生成 Newsletter 摘要。")
        return ("錯誤：AI 模型未準備好，無法生成 Newsletter。",unique_search_results) # 返回錯誤字符串給可能的調用者
    if not search_api_key or not search_engine_id:
        logging.warning("缺少搜尋 API 金鑰或 CX ID，無法獲取 AI 新聞。")
        return ("警告：缺少搜尋 API 金鑰或 CX ID，無法獲取 AI 新聞。",unique_search_results) # 返回警告字符串

    logging.info("開始生成 AI 新聞 Newsletter...")
    all_search_results = []

    # 獲取每個興趣的近期新聞
    for interest in interests:
        # 構造搜尋查詢：例如 "最新LLM進展 上週新聞" 或 "AI在醫療應用 recent news"
        # 結合時間詞 ("上週", "近期", "最新", "過去七天") 有助於獲取時效性內容
        # 實際查詢詞可以根據經驗或測試調整
        query = f"{interest} AI news last week" # <-- 設置搜尋查詢格式 (使用英文可能搜尋結果更廣)
        logging.info(f"搜尋 '{interest}' 相關新聞：'{query}'")

        # 使用之前的 perform_web_search 函式獲取搜尋結果
        # 可以考慮獲取更多結果，例如 num_results=5 或 10，並在後續處理時篩選
        results = perform_web_search(query, search_api_key, search_engine_id, num_results=5) # 為每個興趣獲取前 5 個結果

        if results:
            logging.info(f"找到 {len(results)} 個關於 '{interest}' 的新聞結果。")
            all_search_results.extend(results) # 將結果添加到總列表
        else:
            logging.info(f"未能找到關於 '{interest}' 的新聞結果。")

    # 移除重複的搜尋結果 (可能不同興趣搜到同一篇)
    seen_links = set()
    for result in all_search_results:
        link = result.get('link')
        # 確保 link 存在且是字串，避免因錯誤數據導致崩潰
        if link and isinstance(link, str) and link not in seen_links:
            unique_search_results.append(result)
            seen_links.add(link)
    logging.info(f"總共找到 {len(unique_search_results)} 個不重複的新聞結果。")


    if not unique_search_results:
        logging.info("未找到任何相關的近期 AI 新聞。")
        return "未找到任何相關的近期 AI 新聞。" # 返回提示信息


    # --- 使用 AI 模型總結並格式化 Newsletter ---
    logging.info("使用 AI 模型總結並格式化 Newsletter...")

    # 構造 Prompt，指示 AI 扮演新聞編輯並總結
    # 這個 Prompt 需要清晰指示 AI 的任務和期望的輸出格式
    # 提取搜尋結果的標題、URL 和摘要作為 AI 的上下文
    context_text = "\n---\n".join([
         f"Title: {item.get('title', 'N/A')}\nURL: {item.get('link', 'N/A')}\nSnippet: {item.get('snippet', 'N/A')}"
         for item in unique_search_results
    ])

    newsletter_prompt = textwrap.dedent(f"""
    你是一個 AI 新聞編輯助理，負責根據提供的搜尋結果，為使用者編寫一份關於**過去一週 AI 領域主要進展**的個性化 Newsletter。
    使用者的興趣主題包括：{", ".join(interests)}。

    請閱讀以下搜尋結果，識別最重要的、與使用者興趣相關的、發生在過去一週的 AI 新聞和發展。
    從中提煉核心要點，並以簡潔、有條理的方式進行總結和呈現，像一份 Newsletter。

    Newsletter 應包含：
    - 一個吸引人的標題 (例如：『本週 AI 新聞摘要』)。
    - 根據主題或重要性進行分組的總結，使用條列或簡短段落。
    - 每條新聞提及關鍵要點。
    - （可選）在總結末尾提及總結來源是基於提供的資訊。
    - 使用流暢的中文。
    - 請確保內容是基於搜尋結果中的**近期**資訊。

    搜尋結果：
    ---
    {context_text}
    ---

    Newsletter 內容：
    """)

    # print(f"DEBUG: Newsletter Prompt:\n{newsletter_prompt}") # 調試用

    try:
        # 呼叫 Gemini 模型生成 Newsletter 內容
        response = await model.generate_content_async(newsletter_prompt)
        newsletter_content = response.text.strip() if response.text else None

        if newsletter_content:
            return (newsletter_content, unique_search_results)
        else:
            # 檢查安全回饋
            if response.prompt_feedback and response.prompt_feedback.block_reason:
                 logging.warning(f"Newsletter 生成 Prompt 被阻止，原因: {response.prompt_feedback.block_reason_message}")
                 return (f"抱歉，因內容政策，Newsletter 未能生成 ({response.prompt_feedback.block_reason_message})。", unique_search_results)
            return ("抱歉，AI 未能成功生成 Newsletter 內容。", unique_search_results)
            
    except genai.types.BlockedPromptException as bpe:
        logging.error(f"Newsletter 生成 Prompt 被阻止 (BlockedPromptException): {bpe}", exc_info=False)
        return (f"抱歉，因內容政策，Newsletter 生成請求被阻止。", unique_search_results)
    except Exception as e:
        return (f"抱歉，生成 Newsletter 時發生內部錯誤：{e}", unique_search_results)



# --- 將核心 RAG 回答邏輯封裝成函式 ---
async def get_ai_response(user_input: str, model: genai.GenerativeModel, vectorstore: Optional[PGVector], search_api_key: Optional[str], search_engine_id: Optional[str]) -> str:
    """
    根據使用者輸入，結合個人知識庫和網路搜尋，生成 AI 的回答。

    Args:
        user_input (str): 使用者的問題或指令。
        model: 已初始化好的 Gemini 模型實例。
        vectorstore (Optional[PGVector]): 已初始化好的個人知識庫向量資料庫實例，可能為 None。
        search_api_key (Optional[str]): Google Search API 金鑰，可能為 None。
        search_engine_id (Optional[str]): Google Search Engine ID，可能為 None。

    Returns:
        str: AI 生成的回答文字。
    """
    if model is None:
        return "錯誤：AI 模型未初始化，無法生成回答。" # 安全檢查

    logging.info(f"處理使用者輸入：{user_input}")

    # 直接呼叫新的意圖判斷函式
    intent_data = await decide_intent_and_query(user_input, model) # <--- 呼叫新的函式
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

async def get_ai_response_stream(
    user_input: str,
    model: genai.GenerativeModel, # 確保傳入的是初始化的 genai.GenerativeModel 實例
    vectorstore: Optional[PGVector], # 假設你已遷移到 PGVector
    search_api_key: Optional[str],
    search_engine_id: Optional[str]
    # session_id: Optional[str] = None # 考慮傳入 session_id 以便後續記錄完整對話
) -> AsyncIterable[str]: # 返回一個非同步可迭代的字串 (文本片段)
    """
    根據使用者輸入，結合 RAG，以非同步串流方式生成 AI 的回答。
    逐塊 yield 文本。
    """
    if model is None:
        yield "錯誤：AI 模型未初始化，無法生成回答。"
        return # 結束生成器

    logging.info(f"串流處理使用者輸入：{user_input}")

    # 1. 意圖判斷 (這部分目前是同步的，如果 model.generate_content 是瓶頸，未來也可考慮優化)
    #    注意：decide_intent_and_query 內部也呼叫了 model.generate_content。
    #    為了串流，理想情況下所有對 LLM 的呼叫都應該是非同步的。
    #    暫時我們先假設 decide_intent_and_query 執行相對較快，或者其結果對於串流生成是必要的。
    #    如果 decide_intent_and_query 本身也需要變成 async，那麼這裡需要 await。
    #    為了簡化第一步，我們先保持 decide_intent_and_query 同步，但要注意這點。
    try:
        intent_data = await decide_intent_and_query(user_input, model) # 假設 model 實例適用於同步和異步
    except Exception as e_intent:
        logging.error(f"意圖判斷時發生錯誤: {e_intent}", exc_info=True)
        yield f"抱歉，在理解您的問題時發生錯誤。"
        return

    primary_intent = intent_data.get('intent', 'chat')
    relevant_term = intent_data.get('term', user_input)
    search_needed = intent_data.get('web_search_needed', False)
    web_search_query = intent_data.get('web_search_query', relevant_term or user_input)
     
    # 為了收集 RAG 的上下文，我們可以定義一個內部異步函數或者直接 await
    personal_context_str = None
    web_search_context_str = None

    # 2. RAG - 個人知識庫檢索 (異步)
     # --- RAG 操作可以考慮並行執行 ---
    rag_tasks = []
    personal_context = None
    if vectorstore is not None:
        retrieval_query = relevant_term if relevant_term and primary_intent in ['explain', 'discuss'] else user_input
        if not retrieval_query.strip():
            retrieval_query = user_input

        logging.info(f"從 PGVector 檢索與查詢 '{retrieval_query}' (意圖: {primary_intent}) 相關的片段...")
        async def retrieve_personal_context():
            nonlocal personal_context_str # 允許修改外部作用域的變數
            logging.info(f"從 PGVector 非同步檢索與查詢 '{retrieval_query}' 相關片段...")
            try:
                # 優先使用非同步方法 asimilarity_search
                retrieved_docs = await vectorstore.asimilarity_search(retrieval_query, k=3)
                if retrieved_docs:
                    personal_context_parts = [doc.page_content for doc in retrieved_docs]
                    personal_context_str = "\n---\n".join(personal_context_parts)
                    logging.info(f"PGVector 非同步檢索到 {len(retrieved_docs)} 個片段。")
                else:
                    logging.info("PGVector 非同步檢索未能找到相關片段。")
            except Exception as e_rag:
                logging.error(f"執行 PGVector 非同步檢索時發生錯誤: {e_rag}", exc_info=True)
        
        rag_tasks.append(retrieve_personal_context())

    # 3. RAG - 網路搜尋 (同步)
    web_search_context = None
    if search_needed:
        if not search_api_key or not search_engine_id:
            logging.warning("無法執行網路搜尋，缺少 API 金鑰或 CX ID。")
        else:
            async def retrieve_web_context():
                nonlocal web_search_context_str
                logging.info(f"使用 asyncio.to_thread 執行網路搜尋，查詢詞：'{web_search_query}'")
                try:
                    # asyncio.to_thread 將同步函式放到執行緒池中執行
                    search_results_items = await asyncio.to_thread(
                        perform_web_search, web_search_query, search_api_key, search_engine_id, 3
                    )
                    if search_results_items:
                        context_parts = []
                        for i, item in enumerate(search_results_items):
                            title = item.get('title', 'N/A')
                            link = item.get('link', 'N/A')
                            snippet = item.get('snippet', 'N/A')
                            context_parts.append(textwrap.dedent(f"Web Source {i+1}:\nTitle: {title}\nURL: {link}\nSnippet: {snippet}\n").strip())
                        web_search_context_str = "\n---\n".join(context_parts)
                        logging.info(f"網路搜尋（執行緒）找到 {len(search_results_items)} 個結果。")
                    else:
                        logging.info("網路搜尋（執行緒）未找到相關結果。")
                except Exception as e_web_search:
                    logging.error(f"執行網路搜尋（執行緒）時發生錯誤: {e_web_search}", exc_info=True)

            rag_tasks.append(retrieve_web_context())

    # 並行執行所有 RAG 任務 (如果有的話)
    if rag_tasks:
        logging.info(f"開始並行執行 {len(rag_tasks)} 個 RAG 任務...")
        await asyncio.gather(*rag_tasks)
        logging.info("所有 RAG 任務執行完畢。")
        
    # 4. 構造最終 Prompt (與 get_ai_response 中類似)
    final_prompt_parts = []
    if primary_intent == 'explain':
        final_prompt_parts.append(textwrap.dedent(f"""
        請扮演一位資深的 AI 軟體工程師和專業的面試準備輔導員。
        你的任務是根據提供的參考資訊（如果有的話），以**清晰、有條理、針對面試考點的方式**，深入解釋概念「{relevant_term}」。
        主要基於「個人知識庫」中的內容進行解釋。如果個人知識庫資訊不足，可參考「網路搜尋結果」。如果所有參考資訊都不足以解釋，請用通用知識。
        回答應該包含概念的定義、原理、關鍵要素、優缺點等，適合面試回答的風格，精簡且精確。使用流暢的中文。
        以下是參考資訊：
        """).strip())
    else: # 'chat'
        final_prompt_parts.append(textwrap.dedent("""
        請扮演一個經驗豐富的 AI 軟體工程師，同時也是一位樂於助人的面試準備輔導員。
        你的任務是根據提供的參考資訊（如果有的話），精準、清晰、有條理、針對面試考點地回答或解釋使用者關於 AI、機器學習、深度學習、演算法、資料結構、系統設計、個人專案等方面的問題。
        回答時請嚴格遵循以下優先順序和原則：
        1.  最優先且深度參考「個人知識庫」。... (同之前標準 Prompt 的第 1-5 點) ...
        以下是參考資訊：
        """).strip())

    if personal_context:
        final_prompt_parts.append(textwrap.dedent(f"---\n個人知識庫參考:\n{personal_context}").strip())
    # ... (處理 vectorstore is None 或未檢索到內容的提示) ...
    elif vectorstore is not None:
         final_prompt_parts.append(textwrap.dedent("---\n個人知識庫參考:\n(未從個人知識庫檢索到與問題相關的筆記)").strip())
    else:
         final_prompt_parts.append(textwrap.dedent("---\n個人知識庫參考:\n(個人知識庫功能未啟用或未成功載入)").strip())


    if web_search_context:
        final_prompt_parts.append(textwrap.dedent(f"---\n網路搜尋結果參考:\n{web_search_context}").strip())
    
    final_prompt_parts.append(textwrap.dedent(f"---\n使用者問題:\n{user_input}").strip())
    final_prompt = "\n\n".join(final_prompt_parts)

    # 5. 執行 AI 模型生成內容 (串流模式)
    logging.info("開始從 AI 模型獲取串流回應...")
    try:
        async_response_iterable = await model.generate_content_async(final_prompt, stream=True)
        async for chunk in async_response_iterable:
            # ... (與你之前版本相同的串流處理和安全檢查邏輯) ...
            # 例如:
            # if chunk.candidates and any(c.finish_reason == genai.types.Candidate.FinishReason.SAFETY for c in chunk.candidates):
            #     logging.warning("串流內容被安全策略阻止。")
            #     yield "[內容因安全原因被過濾]" # 或者不 yield 任何東西，或者 yield 一個特定的錯誤事件
            #     break 
            if chunk.text:
                yield chunk.text
        
        # 檢查迭代完成後的 prompt_feedback (如果適用且重要)
        if hasattr(async_response_iterable, 'prompt_feedback') and \
           async_response_iterable.prompt_feedback and \
           async_response_iterable.prompt_feedback.block_reason:
            block_reason_message = async_response_iterable.prompt_feedback.block_reason_message
            logging.warning(f"完整提示在串流後被阻止，原因：{block_reason_message}")
            # 注意：此時串流可能已經結束，前端可能已經收到了部分內容（如果有的話）
            # 在 SSE 中，可以考慮發送一個特殊的 "error" 或 "blocked" 事件
            # 這裡我們可能無法再 yield，因為迭代器已耗盡
            # 更好的做法是在 FastAPI 端點捕獲這個狀態，或者在前端有結束標記時檢查

    except genai.types.BlockedPromptException as bpe:
        logging.error(f"由於 Prompt 被阻止，無法生成串流回答: {bpe}", exc_info=False)
        block_reason_msg = "您的請求因內容政策被阻止"
        yield f"抱歉，{block_reason_msg}，無法處理。"
    except Exception as e:
        logging.error(f"與 AI 模型互動生成串流回答時發生錯誤: {e}", exc_info=True)
        yield f"抱歉，生成回答時發生內部錯誤。"
        # 完整的錯誤是 str(e)，但可能太長不適合直接 yield 給前端

async def run_cli_tests_and_loop(): # <--- 新的 async 函式包裹 __main__ 的邏輯
    config = load_env_variables()
    
    GOOGLE_API_KEY_LOCAL = config.get('GOOGLE_API_KEY')
    DATABASE_URL_LOCAL = config.get('DATABASE_URL')
    SEARCH_API_KEY_LOCAL = config.get('SEARCH_API_KEY')
    SEARCH_ENGINE_ID_LOCAL = config.get('SEARCH_ENGINE_ID')
    OPENAI_API_KEY_LOCAL = config.get('OPENAI_API_KEY')

    model = None
    if GOOGLE_API_KEY_LOCAL:
        model = initialize_gemini_model(GOOGLE_API_KEY_LOCAL) # initialize_gemini_model 保持同步
    else:
        logging.error("本地測試：無法初始化 Gemini 模型...")

    vectorstore = None
    if GOOGLE_API_KEY_LOCAL and DATABASE_URL_LOCAL:
        document_paths_to_process = ["my_notes.txt"]
        # initialize_vector_store 保持同步，它內部不直接 await IO 密集操作
        # 而是設定 PGVector，實際的 IO 操作 (add_documents, similarity_search) 才需要非同步
        vectorstore = initialize_vector_store( 
            document_paths_to_process,
            GOOGLE_API_KEY_LOCAL,
            DATABASE_URL_LOCAL
        )

    if model is None:
        logging.critical("本地測試：AI 模型未成功初始化...")
    
    print("--- 應用程式啟動初始化完成 (CLI 模式) ---")
    print("-" * 40)

    # --- Newsletter 測試 ---
    USER_AI_INTERESTS = ["大型語言模型", "AI倫理"]
    # print("\n--- 運行 AI 新聞 Newsletter 生成測試 ---")
    # if model and SEARCH_API_KEY_LOCAL and SEARCH_ENGINE_ID_LOCAL:
    #     logging.info(f"開始非同步生成 Newsletter...")
    #     # 使用 await 呼叫非同步的 generate_ai_newsletter
    #     newsletter_text, newsletter_sources = await generate_ai_newsletter( # <--- await
    #         USER_AI_INTERESTS, 
    #         model, 
    #         SEARCH_API_KEY_LOCAL, 
    #         SEARCH_ENGINE_ID_LOCAL
    #     )
    #     # ... (處理 newsletter_text 和 newsletter_sources) ...
    #     if newsletter_text and not newsletter_text.startswith("抱歉"):
    #         print("\n--- 生成的 AI 新聞 Newsletter ---\n", newsletter_text)
    #         if newsletter_sources: print("--- Newsletter 來源 ---\n", newsletter_sources)
    #     else:
    #         print(f"\n未能成功生成 Newsletter: {newsletter_text}")

    # else:
    #     logging.warning("跳過 Newsletter 生成測試...")
    print("-" * 40)
    
    # --- STT 測試 ---
    # 如果 openai_client.audio.transcriptions.create 有 async 版本，也可以優化) ...
    # 為了簡化，我們先保持 STT 測試部分的同步性，或者你可以用 await asyncio.to_thread()
    print("\n--- 運行語音轉文字測試 ---")

    # --- 主要對話迴圈 ---
    if model:
        print("--- 個人 AI 助理 (CLI 模式) ---")
    
        print("-" * 40)
        while True:
            try:
                user_input = await asyncio.to_thread(input, "您：") # <--- 使 input 非阻塞 (可選優化)
                # 或者保持同步 input: user_input = input("您：")
                logging.info(f"使用者輸入：{user_input}")
            except (EOFError, KeyboardInterrupt):
                # ...
                break
            # ... (處理退出和空輸入) ...
            if user_input.lower() in ['exit', 'quit']: break
            if not user_input.strip(): continue

            # ---- 測試選項 ----
            test_mode = await asyncio.to_thread(input, "選擇測試模式 (1: 非串流完整回應, 2: 串流回應, 直接按 Enter 預設為串流): ")
            
            if test_mode == '1':
                print("\n--- 測試非串流 get_ai_response ---")
                ai_response_text_full = await get_ai_response( # 假設 get_ai_response 也是 async def
                    user_input,
                    model,
                    vectorstore, 
                    SEARCH_API_KEY_LOCAL, 
                    SEARCH_ENGINE_ID_LOCAL
                )
                if ai_response_text_full:
                     print("AI 助理 (完整回應)：\n", textwrap.fill(ai_response_text_full, width=80))
                else:
                    print("AI 助理 (完整回應)：未能獲取回應。")
            else: # 預設或選擇 '2'
                print("\n--- 測試串流 get_ai_response_stream ---")
                print("AI 助理 (串流回應)：", end="", flush=True)
                try:
                    async for chunk in get_ai_response_stream(
                        user_input,
                        model,
                        vectorstore, 
                        SEARCH_API_KEY_LOCAL, 
                        SEARCH_ENGINE_ID_LOCAL
                    ):
                        print(chunk, end="", flush=True)
                    print() # 串流結束後換行
                except Exception as stream_err:
                    print(f"\n處理回應串流時發生錯誤: {stream_err}")
                    logging.error(f"回應串流錯誤: {stream_err}", exc_info=True)
            print("-" * 40)
    else:
        print("AI 模型未能初始化，無法啟動對話迴圈。")

    logging.info("應用程式已結束 (CLI 模式)。")

# --- 主要應用程式入口點 --- (保留這個區塊，但修改其內容)
if __name__ == "__main__":
    try:
        asyncio.run(run_cli_tests_and_loop())
    except KeyboardInterrupt:
        print("\n程式被使用者中斷。")
    except Exception as e:
        print(f"執行主程式時發生未預期的錯誤: {e}")
        logging.error("主程式執行錯誤", exc_info=True)