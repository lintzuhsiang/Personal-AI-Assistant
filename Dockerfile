# Dockerfile (Clean Version)

# 1. 使用官方的 Python 映像檔作為基礎
FROM python:3.11-slim

# 2. 設定容器內的工作目錄
WORKDIR /app

# 3. 安裝基本的編譯工具 (以防某些核心依賴需要編譯)
#    build-essential 包含 gcc, g++, make 等
#    pkg-config 幫助找到已安裝的函式庫
#    這比預先安裝大量特定的 -dev 函式庫更為精簡
RUN apt-get update && apt-get install -y \
    build-essential \
    pkg-config \
    && rm -rf /var/lib/apt/lists/* # 清理 apt 快取

# 4. 複製 requirements.txt 到容器中
#    (這裡假設你已經有了一個精簡後的 requirements.txt)
COPY requirements.txt requirements.txt

# 5. 安裝 Python 依賴
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# 6. 複製應用程式的其餘所有程式碼到容器中
COPY . .

# 7. 設定容器啟動時執行的命令
#    確保你的 FastAPI 主檔案是 api_app.py，且 FastAPI 實例名為 app
#    Cloud Run 通常會注入 PORT 環境變數，預設為 8080
CMD ["uvicorn", "api_app:app", "--host", "0.0.0.0", "--port", "8080"]
