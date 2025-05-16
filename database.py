# database.py

import logging
import os
from sqlmodel import create_engine, Field, SQLModel, Session
from sqlalchemy.sql import func # 導入 func 來獲取當前時間
from typing import Optional # Import Optional for type hints
from datetime import datetime
import json

# --- 資料庫配置 ---
# 定義 SQLite 資料庫的文件路徑
# __file__ 指向當前文件 (database.py) 的路徑
# os.path.abspath(__file__) 獲取絕對路徑
# os.path.join 合併路徑
# os.path.dirname 獲取目錄路徑
# final_db_path 就是您的專案目錄下的 chat_history.db 文件
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_NAME = "chat_history.db"
FINAL_DATABASE_PATH = os.path.join(BASE_DIR, DATABASE_NAME)

# 定義資料庫 URL
# "sqlite:///./chat_history.db" 表示在程式運行目錄下創建 chat_history.db
# 使用絕對路徑更穩健
SQLALCHEMY_DATABASE_URL = f"sqlite:///{FINAL_DATABASE_PATH}"

# 創建 SQLAlchemy Engine
# connect_args={"check_same_thread": False} 僅對於 SQLite 是必需的，
# 允許在同一個線程中處理多個請求，這在 FastAPI 中常用。
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)


# --- 定義資料庫模型 (使用 SQLModel) ---
# SQLModel 類別同時是 SQLAlchemy 模型和 Pydantic 模型
# 它繼承自 SQLModel，並使用 Field 來定義列屬性
class Message(SQLModel, table=True): # table=True 表示這是一個對應資料表的模型
    # id 作為主鍵，自動增長。default=None 和 Optional[int] 是 SQLModel 處理自動增長的標準方式
    id: Optional[int] = Field(default=None, primary_key=True, index=True)

    # 其他字段使用 Pydantic 類型提示，Field 用於添加資料庫特定的屬性 (如 index)
    session_id: str = Field(index=True) # 對話會話 ID
    sender: str = Field(index=True) # 發送者 ('user' 或 'ai')
    text: str # 訊息內容 (對應 TEXT 類型)

    # 訊息時間戳，使用 Python datetime 在應用層生成默認值並確保不能為空
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, # 使用 UTC 時間作為默認值生成函數
        nullable=False # 資料庫中該列不能為 NULL
    )

    # 注意：SQLModel 會自動根據類別名稱（小寫加 s）生成 __tablename__，例如 "message" -> "messages"

# --- 新增：定義 Newsletter 的資料庫模型 (使用 SQLModel) ---
# 這個模型用於儲存生成的 AI 新聞 Newsletter
class Newsletter(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True, index=True) # 主鍵，自動增長

    # Newsletter 生成的時間戳
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, # 使用 UTC 時間作為默認值
        nullable=False # 資料庫中不能為 NULL
    )

    # Newsletter 的內容，使用 Text 類型支持較長文本
    content: str = Field(index=False) # Newsletter 內容不需要索引，但需要是字串

    # 將搜尋結果字典列表轉換為 JSON 字串儲存
    source_results_json: str = Field(default="[]", index=False) # 默認值為空 JSON 列表，確保始終是有效的 JSON 字串


# --- 資料庫工具函數 ---
# 創建資料庫表 (使用 SQLModel 的 metadata)
def create_db_tables():
    """創建資料庫中定義的所有表。"""
    # SQLModel.metadata 包含了所有繼承自 SQLModel 並設置 table=True 的模型信息
    SQLModel.metadata.create_all(bind=engine)

# 獲取資料庫 Session 的依賴函數 (FastAPI Dependency)
def get_db():
    """獲取一個資料庫 Session 實例，使用完畢後自動關閉。"""
    # Session 類型現在可以直接從 sqlmodel 導入
    # 使用 with 語句管理 session，它會在 yield 後自動關閉並回滾/提交事務 (根據有無異常)
    with Session(engine) as session:
        yield session # 將 session 提供給 FastAPI 請求處理函數

# --- 新增：保存 Newsletter 到資料庫的函式 ---
def save_newsletter(newsletter: Newsletter, source_results: list, db: Session) -> Optional[Newsletter]:
    """
    保存一個 Newsletter 實例到資料庫。

    Args:
        newsletter (Newsletter): 需要保存的 Newsletter 對象。
        db (Session): 資料庫 Session。

    Returns:
        Optional[Newsletter]: 保存後的 Newsletter 實例 (包含 ID 和時間戳)，如果失敗返回 None。
    """
    logging.info("保存 Newsletter 到資料庫...")
    try:
        # 將原始搜尋結果列表轉換為 JSON 字串並賦值給 Newsletter 對象的字段
        # 確保 source_results 是一個列表，並且可以被 json.dumps 序列化
        if isinstance(source_results, list):
            # 將 Python 列表/字典結構轉換為 JSON 字串
            newsletter.source_results_json = json.dumps(source_results)
        else:
            logging.warning(f"save_newsletter 接收到的 source_results 不是列表，將保存為空 JSON 列表。類型：{type(source_results)}")
            newsletter.source_results_json = "[]" # 如果不是列表，保存一個空 JSON 列表

        # 將 Newsletter 對象添加到 Session
        db.add(newsletter)
        # 提交 Session，將數據寫入資料庫
        db.commit()
        # 刷新對象以獲取資料庫自動生成的 ID 和時間戳
        db.refresh(newsletter)
        logging.info(f"Newsletter 保存成功 (ID: {newsletter.id})。")
        return newsletter
    except Exception as e:
        logging.error(f"保存 Newsletter 到資料庫時發生錯誤: {e}", exc_info=True)
        # 如果發生錯誤，回滾事務以保持數據庫一致性
        db.rollback()
        return None
