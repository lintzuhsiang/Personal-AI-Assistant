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



# --- 將核心 RAG 回答邏輯封裝成函式 ---
def get_ai_response(user_input: str, model: genai.GenerativeModel, search_api_key: Optional[str], search_engine_id: Optional[str]) -> str:
    """
    根據使用者輸入，結合個人知識庫和網路搜尋，生成 AI 的回答。

    Args:
        user_input (str): 使用者的問題或指令。
        model: 已初始化好的 Gemini 模型實例。
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
    
    # 確保核心組件初始化成功
    if model is None:
        # 讓主程式決定是否因模型初始化失敗而退出
        # sys.exit("應用程式啟動失敗：AI 模型未成功初始化。")
        logging.critical("AI 模型未成功初始化，某些功能可能無法使用或應用程式可能無法正常運行。")

    # 向量資料庫可能為 None，如果初始化失敗或沒有文件

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

        # 檢查是否退出指令vectorstore, # 傳遞已初始化的向量資料庫
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
            model, # 傳遞已初始化的模型
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