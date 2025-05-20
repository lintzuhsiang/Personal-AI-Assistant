# 導入 FastAPI 相關模組
import uuid
from fastapi import FastAPI, UploadFile, File, Depends
from fastapi.responses import StreamingResponse # <--- 新增 StreamingResponse
from pydantic import BaseModel # 用於定義請求和回應的資料模型
import uvicorn # 用於運行 ASGI 服務器
from typing import Annotated, Optional
import tempfile
import textwrap
import shutil
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware # 導入 CORS 中介層
import openai
from fastapi.concurrency import run_in_threadpool
from fastapi import BackgroundTasks, HTTPException
from typing import List, Dict
# 導入資料庫模組和相關類別
from database import engine, create_db_tables, get_db, Message, Newsletter, save_newsletter # 導入 engine 和模型創建函數等
from sqlmodel import Session  # 導入 Session 類型用於依賴注入
from sqlalchemy.orm import Session as SQLASession # 避免與 sqlmodel.Session 衝突，重命名
import json
# 導入 SQLAlchemy 的 desc 函式用於排序
from sqlalchemy import desc # <--- 新增：導入 desc

# 導入我們剛才重構的核心邏輯模組
# 假設您已將原文件改名為 ai_assistant_core.py
from ai_assistant_core import (
    load_env_variables,
    initialize_gemini_model,
    initialize_vector_store,
    get_ai_response,
    get_ai_response_stream,
    process_and_add_to_vector_store,
    generate_ai_newsletter,
    transcribe_audio,
    USER_AI_INTERESTS
    # 如果您將 GOOGLE_API_KEY, SEARCH_API_KEY, SEARCH_ENGINE_ID 也定義在 core 模組中並需要直接訪問，
    # 您可能需要從 core 模組導入或通過 core 模組的函數獲取
)

import logging # 導入 logging，與核心模組共享日誌配置
import sys # 導入 sys，可能用於退出
import os 
# 確保 logging 在應用啟動前配置
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    stream=sys.stdout)


# --- 全局變數：儲存應用啟動時初始化的核心組件 ---
# 這些變數將在 FastAPI 應用啟動時被賦值，並在 API 請求處理時使用
# 使用 Optional 並初始化為 None，表示可能未成功初始化
model = None
vectorstore = None
# 如果核心模組將 API 金鑰保留為全局變數，這裡不需要重複聲明
# 否則，需要通過 load_env_variables 獲取並儲存金鑰
GOOGLE_API_KEY = None
SEARCH_API_KEY = None
SEARCH_ENGINE_ID = None
OPENAI_API_KEY = None
openai_client = None # 新增：全局 OpenAI Client 實例
DEFAULT_EMBEDDING_MODEL_NAME = 'models/text-embedding-004' # 或者從設定檔讀取

# --- 定義請求和回應的資料模型 (使用 Pydantic) --- (Industry Standard)
# 定義使用者發送聊天訊息時的請求體格式
class ChatRequest(BaseModel):
    message: str # 期望接收一個名為 message 的字串
    session_id: Optional[str] = None # 改為可選，前端可以不傳，由後端生成


# 定義伺服器返回 AI 回應時的回應體格式
class ChatResponse(BaseModel):
    reply: str # 期望返回一個名為 reply 的字串
    session_id: str # 後端總是返回 session_id，無論是接收到的還是新生成的
    # 您可以添加更多字段，例如 status: str, source_docs: list = [] 等

# --- 新增：定義獲取最新 Newsletter 的回應模型 ---
# 這個模型用於包裝 Newsletter 內容，並提供錯誤訊息字段
class LatestNewsletterResponse(BaseModel):
    # 這裡使用 Newsletter SQLModel 作為嵌套模型
    # 但是由於 SQLModel 類別同時是 Pydantic 模型，可以直接使用它作為類型提示
    # 所以直接使用 Optional[Newsletter] 是可以的
    # 但為了提供額外的 message 字段，我們需要一個新的 Response 模型來包裝
    newsletter: Optional[Newsletter] = None # 可能包含 Newsletter 對象
    message: str = "" # 可能包含錯誤或提示訊息


# --- 初始化 FastAPI 應用程式 ---
app = FastAPI()

# --- 添加 CORS 中介層 (Industry Standard) ---
# 允許所有來源 (為了本地開發和測試方便)
# 在生產環境中，您應該將 "*" 替換為您前端應用的具體來源 URL
# 如果您是直接打開本地 HTML 文件 (file://)，某些瀏覽器可能會有 CORS 問題，
# 允許 "*" 是最寬鬆的，適合本地開發調試。
# 更嚴謹的做法是通過一個簡單的本地 HTTP 服務器來運行前端。
origins = [
    "*" # 允許所有來源進行跨來源請求
    # 如果您知道前端的精確來源，可以這樣設定 (例如):
    # "http://localhost",
    # "http://localhost:8080",
    # "http://127.0.0.1:8000", # 允許來自後端本身的請求 (如果前端和後端在同一個服務器)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, # 允許的來源列表
    allow_credentials=True, # 允許跨來源請求包含憑證 (cookies, 授權標頭等)，如果需要的話
    allow_methods=["*"], # 允許所有 HTTP 方法 (POST, GET, OPTIONS 等)，或者只列出需要的 ["POST", "GET"]
    allow_headers=["*"], # 允許所有 HTTP 標頭，或者只列出需要的 ["Content-Type", "Authorization"]
)
# --- 應用啟動時的事件處理器 ---
# @app.on_event("startup") 裝飾器確保這個函數在應用程式啟動時運行一次
@app.on_event("startup")
async def startup_event():
    """在應用啟動時載入環境變數並初始化 AI 模型和向量資料庫。"""
    logging.info("應用程式啟動中...")

    global model, GOOGLE_API_KEY, SEARCH_API_KEY, SEARCH_ENGINE_ID, OPENAI_API_KEY, DATABASE_URL # 聲明將要修改全局變數

    # 1. 載入環境變數
    config_from_core = load_env_variables() # 假設這個函式返回一個包含所有金鑰的字典
    GOOGLE_API_KEY = config_from_core.get('GOOGLE_API_KEY')
    SEARCH_API_KEY = config_from_core.get('SEARCH_API_KEY')
    SEARCH_ENGINE_ID = config_from_core.get('SEARCH_ENGINE_ID')
    OPENAI_API_KEY = config_from_core.get('OPENAI_API_KEY')
    DATABASE_URL = config_from_core.get("DATABASE_URL") # 從環境變數讀取

    if not GOOGLE_API_KEY:
         logging.error("缺少 GOOGLE_API_KEY 環境變數，無法初始化模型。")
         # 不 sys.exit，讓應用啟動但聊天功能會報錯
         # 可以設置一個狀態標誌，表示初始化失敗

    if not SEARCH_API_KEY or not SEARCH_ENGINE_ID:
         logging.warning("缺少搜尋 API 金鑰或 ID 環境變數，網路搜尋功能將無法使用。")

    if OPENAI_API_KEY:
        try:
            openai_client_instance = openai.OpenAI(api_key=OPENAI_API_KEY) # 初始化一次
            # 將實例賦給全局變數，或者如果使用 config 字典，存儲在 config 中
            # 例如，如果你有一個全局的 app_config 字典:
            # app_config['openai_client'] = openai_client_instance
            # 或者，如果你在 api_app.py 中也使用全局變數：
            global openai_client # 聲明在 startup_event 內部要修改的全局 openai_client
            openai_client = openai_client_instance
            logging.info("OpenAI 客戶端初始化成功。")
        except Exception as e:
            logging.error(f"初始化 OpenAI 客戶端失敗: {e}")
    else:
        logging.warning("缺少 OPENAI_API_KEY，語音轉文字功能可能無法使用。")
    
    if not DATABASE_URL:
        logging.error("DATABASE_URL 環境變數未設定，向量資料庫可能無法初始化。")
        # 根據你的邏輯，這裡可能需要更嚴格的處理，例如不啟動向量儲存

    # 1. 初始化 AI 模型
    model = initialize_gemini_model(GOOGLE_API_KEY) # 使用讀取的金鑰初始化

    # 2. 初始化向量資料庫
    # 假設我們處理單個 my_notes.txt 文件
    document_paths_to_process = ["my_notes.txt"] # <--- 設定您個人文件的路徑列表

    if GOOGLE_API_KEY and DATABASE_URL: # 確保金鑰和DB URL都存在
        global vectorstore # 聲明修改全局 vectorstore
        vectorstore = initialize_vector_store(
            document_paths_to_process, 
            GOOGLE_API_KEY,
            DATABASE_URL # 傳遞資料庫連接字串
            # embedding_model_name 可以使用 initialize_vector_store 中的預設值
        )
    else:
        logging.warning("由於缺少 GOOGLE_API_KEY 或 DATABASE_URL，跳過向量資料庫初始化。")

    if vectorstore:
        logging.info("PGVector 向量資料庫初始化完成。")
    else:
        logging.warning("PGVector 向量資料庫初始化失敗或未啟用。")

    # --- 新增：創建資料庫表 ---
    logging.info("檢查並創建資料庫表...")
    create_db_tables() # 呼叫 database.py 中定義的函數來創建表
    logging.info("資料庫表檢查與創建完成。")

    if model:
        logging.info("AI 模型初始化完成。")
    else:
        logging.error("AI 模型初始化失敗。")

    if vectorstore:
         logging.info("向量資料庫初始化完成。")
    else:
         logging.warning("向量資料庫初始化失敗或未啟用。")

    logging.info("應用程式啟動完成。")


# --- 定義 API 端點 ---
# @app.post("/chat") 定義一個處理 POST 請求的端點，路徑是 /chat
# response_model=ChatResponse 指定了回應的資料模型
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, db: Session = Depends(get_db)):
    """
    處理聊天請求，接收使用者訊息並返回 AI 的回答。
    """
    logging.info(f"接收到聊天請求：'{request.message}'")

       # 決定 current_session_id
    if request.session_id:
        current_session_id = request.session_id
        logging.info(f"接收到聊天請求：'{request.message}' (Session ID: {current_session_id})")
    else:
        current_session_id = str(uuid.uuid4()) # 生成一個新的 UUID 作為 session ID
        logging.info(f"未提供 Session ID，已為新對話生成 Session ID：{current_session_id}")
        logging.info(f"新對話的第一條訊息：'{request.message}'")
        
    # 檢查 AI 模型是否成功初始化
    if model is None:
        logging.error("接收到請求，但 AI 模型未初始化。")
        # 返回一個包含錯誤訊息的回應
        return ChatResponse(reply="抱歉，AI 助理目前無法使用，因為模型未初始化。")
    

    user_message = Message(session_id=current_session_id, sender="user", text=request.message)
    db.add(user_message) # 將使用者訊息添加到數據庫 Session
    db.commit() # 提交 Session，將數據寫入資料庫
    db.refresh(user_message) # 刷新對象，獲取資料庫自動生成的 ID 和時間戳

    # 呼叫核心邏輯函式來獲取 AI 的回答
    # 將使用者輸入、模型、向量庫、搜尋金鑰都傳遞給函式
    # 核心函式內部處理檢索、搜尋判斷、Prompt 構造和 AI 呼叫
    # 注意：這裡需要將搜索 API 金鑰和 ID 傳遞給 get_ai_response
    ai_response_text = await run_in_threadpool(
        get_ai_response, # 要運行的同步函式
        user_input=request.message, # 傳遞給函式的參數
        model=model,
        vectorstore=vectorstore,
        search_api_key=SEARCH_API_KEY,
        search_engine_id=SEARCH_ENGINE_ID
    )

    logging.info(f"生成 AI 回應：'{ai_response_text[:100]}...'") # 記錄回應前 100 字元

    # --- 新增：保存 AI 回應到資料庫 ---
    # 只有當 AI 成功返回回應文本時才保存
    if ai_response_text and ai_response_text != "抱歉，未能獲得有效回應。": # 避免保存通用的錯誤提示
         ai_message = Message(session_id=current_session_id, sender="ai", text=ai_response_text)
         db.add(ai_message) # 將 AI 回應添加到數據庫 Session
         db.commit() # 提交 Session
         db.refresh(ai_message) # 刷新對象

    # 返回 AI 的回答，包裝在 ChatResponse 資料模型中
    return ChatResponse(reply=ai_response_text, session_id=current_session_id)

# --- 新增：串流聊天 API 端點 ---
@app.post("/chat_stream")
async def chat_stream_endpoint(request: ChatRequest, db: Session = Depends(get_db)):
    logging.info(f"接收到串流聊天請求：'{request.message}', Session ID: {request.session_id}")

    if model is None:
        logging.error("串流請求：AI 模型未初始化。")
        raise HTTPException(status_code=503, detail="AI 服務未初始化 (模型)")

    current_session_id = request.session_id
    if not current_session_id:
        current_session_id = str(uuid.uuid4())
        logging.info(f"串流請求：未提供 Session ID，已生成新的 Session ID：{current_session_id}")

    # 1. 保存使用者訊息到資料庫
    try:
        user_message_db = Message(session_id=current_session_id, sender="user", text=request.message)
        db.add(user_message_db)
        db.commit()
        db.refresh(user_message_db)
        logging.info(f"串流請求：使用者訊息已保存 (Session ID: {current_session_id})")
    except Exception as e_db_user:
        logging.error(f"串流請求：保存使用者訊息失敗: {e_db_user}", exc_info=True)
        raise HTTPException(status_code=500, detail="伺服器內部錯誤，無法保存使用者訊息。")

    async def event_generator():
        full_ai_response_chunks = []
        ai_response_saved = False # 標記 AI 回應是否已嘗試儲存
        try:
            async for chunk in get_ai_response_stream(
                user_input=request.message,
                model=model, # 全局 model
                vectorstore=vectorstore, # 全局 vectorstore
                search_api_key=SEARCH_API_KEY,
                search_engine_id=SEARCH_ENGINE_ID
            ):
                if chunk: # 確保 chunk 不是空的
                    # 為 SSE 格式化數據: data: {json_payload}\n\n
                    # 我們傳送 JSON 物件，方便前端處理，例如 {"text": "片段內容"}
                    # 或者 {"event": "error", "data": "錯誤訊息"}
                    # 或者 {"event": "end", "data": "串流結束"}
                    payload = {"text": chunk}
                    yield f"data: {json.dumps(payload)}\n\n"
                    full_ai_response_chunks.append(chunk)
            
            # 串流正常結束後，發送一個結束事件 (可選，但對前端有益)
            yield f"event: end\ndata: Stream ended\n\n"

        except Exception as e_stream_gen:
            logging.error(f"串流生成過程中發生錯誤 (Session ID: {current_session_id}): {e_stream_gen}", exc_info=True)
            error_payload = json.dumps({"error": "串流內容生成時發生錯誤。", "detail": str(e_stream_gen)})
            yield f"event: error\ndata: {error_payload}\n\n" # 發送錯誤事件
        finally:
            # 2. 將完整的 AI 回應儲存到資料庫
            if full_ai_response_chunks:
                complete_ai_response = "".join(full_ai_response_chunks)
                # 避免儲存僅包含錯誤提示或非常短的無意義回應
                if complete_ai_response.strip() and \
                   not complete_ai_response.startswith("抱歉，") and \
                   not complete_ai_response.startswith("錯誤：") and \
                   len(complete_ai_response.strip()) > 10: # 隨意設定一個最小長度
                    try:
                        ai_message_db = Message(session_id=current_session_id, sender="ai", text=complete_ai_response)
                        db.add(ai_message_db)
                        db.commit()
                        db.refresh(ai_message_db)
                        logging.info(f"串流請求：完整 AI 回應已保存 (Session ID: {current_session_id})")
                        ai_response_saved = True
                    except Exception as e_db_ai:
                        logging.error(f"串流請求：保存 AI 回應失敗: {e_db_ai}", exc_info=True)
            
            if not ai_response_saved and not full_ai_response_chunks:
                 logging.info(f"串流請求：未收集到 AI 回應片段，無內容保存 (Session ID: {current_session_id})")
            elif not ai_response_saved and full_ai_response_chunks: # 有片段但未保存 (例如是錯誤訊息)
                 logging.info(f"串流請求：AI 回應為空或錯誤，未保存 (Session ID: {current_session_id})")

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# --- 背景任務處理函式 ---
def process_document_in_background(temp_file_path: str, original_filename: str):
    """
    在背景處理上傳的文件並將其添加到向量資料庫，然後清理臨時文件。
    """
    global vectorstore, GOOGLE_API_KEY # 假設這些是 api_app.py 中的全局變數

    logging.info(f"[背景任務] 開始處理文件 '{original_filename}' (路徑: {temp_file_path})")
    try:
        if vectorstore is None:
            logging.error(f"[背景任務] 向量資料庫未初始化，無法處理文件 '{original_filename}'。")
            return # 提前退出

        if not GOOGLE_API_KEY:
            logging.error(f"[背景任務] GOOGLE_API_KEY 未設定，無法處理文件 '{original_filename}'。")
            return # 提前退出

        success = process_and_add_to_vector_store(
            temp_file_path,
            vectorstore,
            DEFAULT_EMBEDDING_MODEL_NAME, # 使用定義好的常數
            GOOGLE_API_KEY
        )

        if success:
            logging.info(f"[背景任務] 文件 '{original_filename}' 成功處理並添加到向量資料庫。")
        else:
            # process_and_add_to_vector_store 內部應該已經記錄了詳細錯誤
            logging.error(f"[背景任務] 文件 '{original_filename}' 處理或添加到向量資料庫失敗。")

    except Exception as e:
        logging.error(f"[背景任務] 處理文件 '{original_filename}' 時發生意外錯誤: {e}", exc_info=True)
    finally:
        # 無論成功與否，都嘗試刪除臨時文件
        if os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                logging.info(f"[背景任務] 臨時文件 '{temp_file_path}' 已成功刪除。")
            except Exception as e_remove:
                logging.error(f"[背景任務] 刪除臨時文件 '{temp_file_path}' 時發生錯誤: {e_remove}", exc_info=True)

# --- 定義文件上傳 API 端點 ---
@app.post("/upload_document")
async def upload_document(
    file: Annotated[UploadFile, File(...)],
    background_tasks: BackgroundTasks # 添加 BackgroundTasks 依賴
):
    """
    處理文件上傳，將文件內容異步添加到個人知識庫向量資料庫。
    立即返回處理中狀態，實際處理在背景執行。
    """
    logging.info(f"接收到文件上傳請求：文件名 '{file.filename}'，內容類型 '{file.content_type}'")

    # 檢查向量資料庫是否成功初始化
    if vectorstore is None:
        logging.error("接收到文件上傳請求，但向量資料庫未初始化。")
        # 返回 HTTP 錯誤狀態碼和詳細訊息
        raise HTTPException(status_code=503, detail="個人知識庫向量資料庫未準備好。")

    # 檢查必要的 API 金鑰是否可用
    if not GOOGLE_API_KEY:
         logging.error("缺少 GOOGLE_API_KEY 環境變數，無法處理文件並添加到向量資料庫。")
         raise HTTPException(status_code=503, detail="處理文件所需服務未配置 (缺少 API 金鑰)。")

    temp_file_path = None # 初始化
    try:
        # 創建一個具名臨時文件，設置 delete=False 使其在關閉後不被立即刪除
        # 我們需要手動管理其刪除，由背景任務負責
        # 添加後綴以盡可能保留原始副檔名，雖然對於 NamedTemporaryFile 不是嚴格必要
        # 但有助於識別，並確保 file.filename 是安全的
        safe_suffix = "".join(c if c.isalnum() or c in ['.', '_'] else '_' for c in file.filename)

        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{safe_suffix}", mode="wb") as tmp_buffer:
            shutil.copyfileobj(file.file, tmp_buffer) # 將上傳文件的內容複製到臨時文件
            temp_file_path = tmp_buffer.name # 獲取臨時文件的完整路徑

        logging.info(f"文件 '{file.filename}' 已暫存到 '{temp_file_path}'。準備安排背景處理。")

        # 安排背景任務來處理文件
        # 注意：傳遞給 add_task 的參數必須是 background_tasks 實例本身能訪問到的
        # temp_file_path 和 file.filename 是局部變數，所以直接傳遞它們的值
        background_tasks.add_task(
            process_document_in_background, # 要執行的背景函式
            temp_file_path,                 # 傳遞給背景函式的參數
            file.filename                   # 傳遞給背景函式的參數
        )

        logging.info(f"文件 '{file.filename}' 的背景處理任務已成功安排。")

        # 立即返回一個表示請求已被接受並正在處理的回應
        return {"status": "processing", "message": f"文件 '{file.filename}' 已接收並正在背景處理中。最終結果請查看後台日誌。"}

    except Exception as e:
        # 捕獲在文件保存或安排背景任務過程中的任何錯誤
        logging.error(f"處理文件 '{file.filename}' 的上傳或安排背景任務時發生錯誤: {e}", exc_info=True)
        
        # 如果臨時文件已創建但背景任務未能成功安排，我們應該嘗試刪除它
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                logging.info(f"因錯誤未能安排背景任務，已清理臨時文件 '{temp_file_path}'。")
            except Exception as e_remove_outer:
                logging.error(f"因錯誤未能安排背景任務，清理臨時文件 '{temp_file_path}' 時也發生錯誤: {e_remove_outer}", exc_info=True)
        
        raise HTTPException(status_code=500, detail=f"處理文件 '{file.filename}' 時發生內部伺服器錯誤。")
    

# --- 定義語音輸入處理 API 端點 ---
# 接收音頻檔案上傳
@app.post("/upload_audio_for_summary", response_model=ChatResponse) # 可以返回摘要文字，使用 ChatResponse
async def upload_audio_for_summary(audio_file: Annotated[UploadFile, File(...)]): # 接收音頻檔案
    """
    接收音頻檔案，進行語音轉文字，AI 摘要，並將摘要保存到個人知識庫。
    """
    logging.info(f"接收到音頻文件上傳請求：文件名 '{audio_file.filename}'，內容類型 '{audio_file.content_type}'")

    # 檢查必要的 API 金鑰是否可用
    if not OPENAI_API_KEY: # 檢查 OpenAI 金鑰 (轉錄需要)
         logging.error("接收到音頻上傳請求，但缺少 OPENAI_API_KEY 環境變數。")
         return ChatResponse(reply="抱歉，語音轉文字功能未啟用，因為缺少 OpenAI API 金鑰。")

    if not GOOGLE_API_KEY: # 檢查 Google 金鑰 (摘要和保存需要)
         logging.error("接收到音頻上傳請求，但缺少 GOOGLE_API_KEY 環境變數。")
         return ChatResponse(reply="抱歉，AI 摘要和保存功能未啟用，因為缺少 Google API 金鑰。")

    if not openai_client: # 檢查 client 實例是否存在
        logging.error("接收到音頻上傳請求，但 OpenAI client 未初始化。")
        return ChatResponse(reply="抱歉，語音轉文字功能未啟用，因為 OpenAI 客戶端未初始化。")

    # 檢查 AI 模型是否成功初始化 (摘要需要)
    if model is None:
        logging.error("接收到音頻上傳請求，但 AI 模型未初始化。")
        return ChatResponse(reply="抱歉，AI 助理目前無法使用，因為模型未初始化。")

     # 檢查向量資料庫是否成功初始化 (保存摘要需要)
     # 如果向量庫未初始化，可以選擇不保存，但仍然返回摘要
    if vectorstore is None:
        logging.warning("接收到音頻上傳請求，但向量資料庫未初始化，摘要結果將無法保存。")
        # 不返回錯誤，繼續流程，只進行轉錄和摘要但不保存


    # 將上傳的音頻檔案暫時儲存到本地磁碟
    temp_audio_path = None
    transcribed_text = None
    summary_text = None # 初始化摘要文本變數
    try:
        # 創建一個臨時目錄來存放上傳的檔案
        with tempfile.TemporaryDirectory() as tmpdir:
            # 在臨時目錄中保存音頻檔案
            temp_audio_path = os.path.join(tmpdir, audio_file.filename)
            logging.info(f"將音頻檔案暫存到：{temp_audio_path}")

            with open(temp_audio_path, "wb") as buffer:
                shutil.copyfileobj(audio_file.file, buffer)

            logging.info(f"音頻檔案 '{audio_file.filename}' 暫存完成。開始轉錄。")

            # --- 1. 呼叫核心邏輯進行語音轉文字 ---
            # 使用全局的 OPENAI_API_KEY 進行轉錄
            transcribed_text = await run_in_threadpool(transcribe_audio,temp_audio_path, OPENAI_API_KEY)

            if not transcribed_text:
                logging.error("語音轉文字失敗。")
                # transcribe_audio 內部會記錄詳細錯誤
                return ChatResponse(reply="抱歉，語音轉文字失敗，請檢查音頻檔案或後台日誌。")

            logging.info(f"語音轉文字成功。轉錄文本 (前 100 字元): {transcribed_text[:100]}...")


            # --- 2. 呼叫核心邏輯使用 AI 進行摘要 ---
            # 使用已初始化的 AI 模型進行摘要
            logging.info("開始使用 AI 模型進行文本摘要...")
            # 構造摘要 Prompt，指示 AI 進行總結
            summarization_prompt = textwrap.dedent(f"""
            請將以下口語化的轉錄內容進行精煉和總結，提取核心信息。
            摘要應該簡潔、有條理，使用流暢的中文書面語。

            轉錄文字內容：
            ---
            {transcribed_text}
            ---

            精煉總結：
            """)

            try:
                # 呼叫 Gemini 模型進行摘要
                summary_response = await run_in_threadpool(model.generate_content,summarization_prompt)
                summary_text = summary_response.text.strip() if summary_response.text else None

                if not summary_text:
                     logging.warning("AI 模型未能生成摘要文本。")
                     # 返回轉錄文本，並告知未能生成摘要
                     return ChatResponse(reply=f"已轉錄：\n{transcribed_text}\n\n注意：AI 未能成功生成摘要。")

                logging.info(f"AI 摘要成功。摘要文本 (前 100 字元): {summary_text[:100]}...")

                # --- 3. 將摘要保存到個人知識庫 ---
                # 只有當向量資料庫成功初始化且 GOOGLE_API_KEY 可用時才保存
                if vectorstore is not None and GOOGLE_API_KEY:
                     logging.info("開始將摘要保存到個人知識庫...")
                     # process_and_add_to_vector_store 需要文件路徑
                     # 我們需要將摘要文本寫入一個臨時文件再傳遞給它
                     temp_summary_file_path = os.path.join(tmpdir, "summary_" + os.path.basename(audio_file.filename) + ".txt")
                     with open(temp_summary_file_path, "w", encoding="utf-8") as f:
                          # 可以添加一些元數據到摘要文件開頭，例如日期時間或來源
                          summary_header = f"## 語音筆記摘要 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n來源: {audio_file.filename}\n\n" # 添加來源文件名
                          f.write(summary_header)
                          f.write(summary_text)

                     logging.info(f"摘要文本已寫入臨時文件：{temp_summary_file_path}")

                     # 呼叫核心處理函式添加到向量資料庫
                     # 需要將 GOOGLE_API_KEY 傳遞給核心處理函式用於嵌入

                     success = process_and_add_to_vector_store(
                         temp_summary_file_path,
                         vectorstore, # 將全局的 vectorstore 對象傳遞進去
                         DEFAULT_EMBEDDING_MODEL_NAME,
                         GOOGLE_API_KEY
                     )

                     if success:
                         logging.info("摘要成功保存到個人知識庫。")
                         # 返回摘要結果，並告知已保存
                         return ChatResponse(reply=f"摘要成功並已保存到知識庫：\n{summary_text}")
                     else:
                         logging.error("將摘要保存到個人知識庫失敗。")
                         # 返回摘要結果，並告知保存失敗
                         # process_and_add_to_vector_store 內部會記錄詳細錯誤
                         return ChatResponse(reply=f"摘要成功：\n{summary_text}\n\n注意：保存到知識庫失敗，請查看後台日誌。")


                else:
                    logging.warning("向量資料庫未初始化，跳過保存摘要到個人知識庫。")
                    # 返回摘要結果，並告知未保存
                    return ChatResponse(reply=f"摘要成功：\n{summary_text}\n\n注意：個人知識庫未準備好，未能保存摘要。")


            except Exception as e:
                 # 捕獲 AI 摘要或保存時的錯誤
                 logging.error(f"執行 AI 摘要或保存時發生意外錯誤: {e}", exc_info=True)
                 # 即使摘要或保存失敗，也嘗試返回轉錄文本
                 return ChatResponse(reply=f"已轉錄：\n{transcribed_text}\n\n抱歉，生成摘要或保存時發生內部錯誤：{e}")


    except Exception as e:
        # 捕獲文件上傳或轉錄時的錯誤
        logging.error(f"處理音頻上傳和轉錄時發生意外錯誤: {e}", exc_info=True)
        return ChatResponse(reply=f"抱歉，處理音頻文件時發生內部錯誤：{e}")

    # 臨時文件和目錄在 with tempfile.TemporaryDirectory() 區塊結束時會被自動清理

# --- 定義獲取歷史記錄的 API 端點 ---
@app.get("/history", response_model=list[Message]) # 回應模型是一個 Message 對象的列表
async def get_history(session_id: str = "test_session_123", db: Session = Depends(get_db)): # <--- 添加資料庫 Session 依賴
    """
    獲取指定對話會話的歷史記錄。
    Args:
        session_id (str): 需要獲取的對話會話 ID。
        db (Session): 資料庫 Session 依賴。
    Returns:
        list[Message]: 對話訊息列表。
    """
    logging.info(f"接收到獲取對話歷史請求，session_id: {session_id}")

    # 使用資料庫 Session 執行查詢
    # db.query(Message) 選擇 Message 模型對應的表
    # .filter(Message.session_id == session_id) 篩選指定 session ID 的訊息
    # .order_by(Message.timestamp) 按時間戳排序
    # .all() 執行查詢並返回所有結果作為列表

    if not session_id: # 理論上 FastAPI 的路徑/查詢參數強制性會處理，但多一層保險
        raise HTTPException(status_code=400, detail="Session ID 是必需的參數。")

    messages = db.query(Message).filter(Message.session_id == session_id).order_by(Message.timestamp).all()

    logging.info(f"為 Session ID '{session_id}' 獲取到 {len(messages)} 條歷史記錄。")

    # 直接返回 SQLAlchemy 模型對象列表，Pydantic-SQLAlchemy 會自動將其轉換為 Pydantic 模型列表
    return messages

# --- 新增：定義獲取最新 Newsletter 的 API 端點 ---
# 回應模型使用 LatestNewsletterResponse SQLModel (因為它是 SQLModel，FastAPI 會自動轉 JSON)
@app.get("/latest_newsletter", response_model=LatestNewsletterResponse)
async def get_latest_newsletter(db: Session = Depends(get_db)): # <--- 添加資料庫 Session 依賴 (使用 sqlmodel.Session)
    """
    獲取資料庫中最新生成的一份 AI 新聞 Newsletter。

    Args:
        db (Session): 資料庫 Session 依賴。

    Returns:
        Optional[Newsletter]: 最新的一份 Newsletter 實例，如果沒有則返回 None (對應 JSON 中的 null)。
    """
    logging.info("接收到獲取最新 Newsletter 請求。")

    try:
        # 使用資料庫 Session 執行查詢
        # db.query(Newsletter) 選擇 Newsletter 模型對應的表
        # .order_by(desc(Newsletter.timestamp)) 按 timestamp 倒序排序 (最新的在前面)
        # .limit(1) 只獲取一個結果
        # .first() 執行查詢並返回第一個結果 (如果有的話)
        latest_newsletter = db.query(Newsletter).order_by(desc(Newsletter.timestamp)).limit(1).first()

        if latest_newsletter:
            logging.info(f"成功獲取到最新 Newsletter (ID: {latest_newsletter.id}, 時間: {latest_newsletter.timestamp})。")
             # 從資料庫對象中提取數據，包括 source_results_json
            # 將 source_results_json 從 JSON 字串轉換為 Python 列表/字典結構
            source_results_list = []
            if latest_newsletter.source_results_json:
                 try:
                      source_results_list = json.loads(latest_newsletter.source_results_json)
                      if not isinstance(source_results_list, list): # 確保解析結果是列表
                          logging.warning("從資料庫讀取的 source_results_json 解析結果不是列表。")
                          source_results_list = []
                 except json.JSONDecodeError:
                      logging.error("從資料庫讀取的 source_results_json 無法解析為 JSON。")
                      source_results_list = []
            # 返回包含 Newsletter 對象和原始搜尋結果列表的 Response
            # 注意：這裡將 Newsletter 對象賦給 newsletter 字段，將解析後的列表賦給 source_results 字段
            return LatestNewsletterResponse(
                newsletter=latest_newsletter, # 返回 SQLModel 對象
                source_results=source_results_list # 返回解析後的列表
            )
        else:
            logging.info("資料庫中沒有找到任何 Newsletter 記錄。返回 None。")
              # 確保 AI 模型和搜尋金鑰可用於生成 (這些是全局變數，在 startup_event 中初始化)
            if model is None:
                logging.error("AI 模型未初始化，無法生成 Newsletter。")
                return LatestNewsletterResponse(message="錯誤：AI 模型未準備好，無法生成 Newsletter。")
            if not SEARCH_API_KEY or not SEARCH_ENGINE_ID:
                logging.warning("缺少搜尋 API 金鑰或 CX ID，無法獲取 AI 新聞用於生成 Newsletter。")
                return LatestNewsletterResponse(message="警告：缺少搜尋 API 金鑰或 CX ID，無法獲取 AI 新聞用於生成 Newsletter。", source_results=None)
            # 呼叫核心邏輯生成 Newsletter 內容
            # 將用戶興趣列表、AI 模型、搜尋金鑰/ID 傳遞給生成函式
            generated_content, unique_search_results = generate_ai_newsletter(USER_AI_INTERESTS, model, SEARCH_API_KEY, SEARCH_ENGINE_ID)

            # 4. 處理生成結果
            if generated_content and not (isinstance(generated_content, str) and generated_content.startswith(("錯誤：", "警告：", "未找到"))):
                # 成功生成了 Newsletter 內容 (且內容不是表示錯誤/警告的字符串)
                logging.info("成功生成 Newsletter 內容，準備保存到資料庫。")
                # 創建 Newsletter SQLModel 實例 (timestamp 會在保存時自動生成)
                new_newsletter = Newsletter(content=generated_content)

                # 呼叫 save_newsletter 函式保存到資料庫
                saved_newsletter = save_newsletter(new_newsletter, unique_search_results, db)

                if saved_newsletter:
                    logging.info("新生成的 Newsletter 保存成功。")
                    # 返回新保存的 Newsletter
                    return LatestNewsletterResponse(newsletter=saved_newsletter, message="成功生成並保存新的 Newsletter。", source_results=None)
                else:
                    logging.error("新生成的 Newsletter 保存失敗。")
                    # 返回生成內容，並提示保存失敗
                    return LatestNewsletterResponse(message=f"成功生成 Newsletter 內容，但保存失敗。\n\n{generated_content}", source_results=None)

            elif isinstance(generated_content, str) and generated_content.startswith(("錯誤：", "警告：", "未找到")):
                 # generate_ai_newsletter 函式返回了表示警告或錯誤的字符串
                 logging.warning(f"Newsletter 生成函式返回警告或錯誤: {generated_content}")
                 return LatestNewsletterResponse(message=f"生成 Newsletter 失敗：{generated_content}", source_results=None)

            else:
                # generate_ai_newsletter 未返回有效內容或錯誤字符串
                 logging.error("generate_ai_newsletter 未返回有效內容或錯誤。")
                 return LatestNewsletterResponse(message="生成 Newsletter 失敗：未能獲取內容。", source_results=None)

    except Exception as e:
        logging.error(f"獲取最新 Newsletter 時發生錯誤: {e}", exc_info=True)
        # 在 API 模式下，返回 None 或者更詳細的錯誤信息（取決於需求）
        # 返回一個包含錯誤訊息的 LatestNewsletterResponse
        return LatestNewsletterResponse(message=f"處理請求時發生內部錯誤：{e}", source_results=None) 

# --- 運行 FastAPI 應用程式 --- (Industry Standard)
# 這個區塊讓您可以直接運行這個檔案來啟動 FastAPI 服務
if __name__ == "__main__":
    # 使用 uvicorn 運行 app 實例
    # host="0.0.0.0" 表示服務器將在所有可用的網絡接口上監聽
    # port=8000 是默認的 HTTP 端口
    # reload=True 使得在代碼修改時服務器自動重載 (僅用於開發階段)
    uvicorn.run(app, host="0.0.0.0", port=8000)