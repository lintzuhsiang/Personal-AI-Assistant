// frontend/script.js (Modified to load history)

// 獲取 DOM 元素
const chatBox = document.getElementById('chat-box');
const userInput = document.getElementById('user-input');
const sendButton = document.getElementById('send-button');

// --- 新增的文件上傳 DOM 元素 ---
const documentUploadInput = document.getElementById('document-upload');
const uploadButton = document.getElementById('upload-button');
const uploadStatusDiv = document.getElementById('upload-status');

// --- 新增的音頻上傳 DOM 元素 ---
const audioUploadInput = document.getElementById('audio-upload'); // <--- 新增：獲取音頻文件選擇 input
const uploadAudioButton = document.getElementById('upload-audio-button'); // <--- 新增：獲取音頻上傳按鈕

// --- 新增的語音錄製 DOM 元素 ---
const startRecordingButton = document.getElementById('start-recording-button'); // <--- 新增：獲取開始錄音按鈕
const stopRecordingButton = document.getElementById('stop-recording-button'); // <--- 新增：獲取停止錄音按鈕

// --- 新增的 Newsletter DOM 元素 ---
const showNewsletterButton = document.getElementById('show-newsletter-button');
const newsletterContentDiv = document.getElementById('newsletter-content');

// FastAPI 後端 API 的端點 URL
const IS_DEVELOPMENT = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1';
const API_BASE_URL = IS_DEVELOPMENT ? 'http://localhost:8000' : 'https://personal-ai-assistant-471912625165.us-east1.run.app';
// const API_BASE_URL = 'https://personal-ai-assistant-471912625165.us-east1.run.app';
const API_CHAT_URL = `${API_BASE_URL}/chat`; // 聊天端點 URL
const API_CHAT_STREAM_URL = `${API_BASE_URL}/chat_stream`; // 新的串流端點 (假設後端已改為 GET)
const API_HISTORY_URL = `${API_BASE_URL}/history`; // 歷史端點 URL
const API_UPLOAD_AUDIO_URL = `${API_BASE_URL}/upload_audio_for_summary`; // <--- 新增：音頻上傳處理端點 URL
const API_LATEST_NEWSLETTER_URL = `${API_BASE_URL}/latest_newsletter`; // <--- 新增：獲取最新 Newsletter 端點 URL

// 暫時使用固定的 session ID，這應該與後端使用的 session_id 一致
// 未來可以通過使用者登入等方式獲取真正的 session ID
// const CURRENT_SESSION_ID = 'test_session_123_1'; // <--- 確保這個與後端保存歷史使用的 ID 一致
let currentSessionId = localStorage.getItem('ai_assistant_pwa_session_id'); // 從 localStorage 嘗試載入
let eventSource = null; 
let currentAiMessageElement = null;

// --- 接著是你所有全局函式的定義 ---
function updateSessionIdDisplay() {
    const sessionIdDisplay = document.getElementById('sessionIdDisplay'); // 可以在函式內部獲取，或假設已在全局獲取
    if (sessionIdDisplay) {
        sessionIdDisplay.textContent = currentSessionId ? currentSessionId : '新對話';
    }
}

// --- 修改 appendMessage 函式以支援串流 ---
function appendMessage(messageText, sender, isStreaming = false) {
    const messageElement = document.createElement('div');
    messageElement.classList.add('message');
    let prefix = "";

    if (sender === 'user') {
        messageElement.classList.add('user-message');
        prefix = "你: ";
        messageElement.textContent = prefix + messageText; // 使用者訊息直接用 textContent
    } else if (sender === 'ai') {
        messageElement.classList.add('ai-message');
        prefix = "AI: ";
        if (isStreaming) {
            if (currentAiMessageElement) { // 如果已有 AI 訊息元素在串流中，追加內容
                // 移除"思考中..." (如果有的話)
                if (currentAiMessageElement.textContent === `${prefix}思考中...` || currentAiMessageElement.textContent === `${prefix}...`) {
                    currentAiMessageElement.textContent = prefix;
                }
                currentAiMessageElement.textContent += messageText;
                chatBox.scrollTop = chatBox.scrollHeight; // 保持滾動
                return; // 不創建新元素，直接返回
            } else { // 這是 AI 串流訊息的第一塊
                messageElement.textContent = prefix + (messageText || "..."); // 初始文本或等待符
                currentAiMessageElement = messageElement; // 保存這個元素以便追加
            }
        } else { // 非串流的 AI 訊息，或串流結束後的最終渲染
            if (typeof marked !== 'undefined') {
                messageElement.innerHTML = prefix + marked.parse(messageText);
            } else {
                messageElement.textContent = prefix + messageText;
            }
            currentAiMessageElement = null; // 清除追蹤
        }
    } else { // system or error messages
        messageElement.classList.add(sender === 'system' ? 'system-message' : 'error-message');
        messageElement.textContent = messageText;
        currentAiMessageElement = null; // 清除追蹤
    }

    chatBox.appendChild(messageElement);
    chatBox.scrollTop = chatBox.scrollHeight;
}
// --- 從後端載入對話歷史 ---
async function loadChatHistory() {
    console.log("正在載入對話歷史..."); // 除錯信息，會在瀏覽器的 Developer Console 顯示
    try {
        // 向後端 /history 端點發送 GET 請求，包含 session_id 查詢參數
        const response = await fetch(`${API_HISTORY_URL}?session_id=${currentSessionId}`);

        // 檢查 HTTP 狀態碼
        if (!response.ok) {
            // 如果狀態碼不是 2xx，表示請求失敗
            const errorText = await response.text(); // 嘗試獲取錯誤訊息
            console.error('載入歷史錯誤:', response.status, errorText); // 在 Console 打印錯誤
            // 在聊天框中顯示錯誤提示
            appendMessage('AI 助理：未能載入對話歷史。', 'ai-message');
            return; // 停止執行
        }

        // 解析後端返回的 JSON 歷史數據
        // 期望返回格式為一個包含訊息對象的列表，每個對象應有 sender 和 text 字段
        const history = await response.json(); // 期望是 [{ id: ..., session_id: ..., sender: 'user'/'ai', text: '...', timestamp: '...' }, ...]

        console.log(`載入到 ${history.length} 條歷史記錄。`); // 除錯信息

        // 清空聊天框中任何靜態的初始歡迎訊息
        chatBox.innerHTML = ''; // 清空聊天框現有內容


        // 將歷史記錄添加到聊天框
        history.forEach(message => {
            // 確保訊息對象有 sender 和 text 字段
            if (message.sender && message.text) {
                appendMessage(message.text, message.sender); // 使用 sender 參數
            }
        });

         // 如果載入後沒有歷史記錄，顯示一個初始歡迎訊息
         if (history.length === 0) {
             appendMessage('AI 助理：您好！有什麼可以幫助您的嗎？', 'ai-message');
         }


    } catch (error) {
        // 捕獲發送請求或處理回應時的錯誤 (例如網絡錯誤)
        console.error('載入歷史錯誤:', error); // 在 Console 打印錯誤
        appendMessage('AI 助理：載入歷史記錄時發生網絡錯誤。', 'ai-message');
    }
}

// --- 新增：串流結束後，對 AI 訊息進行最終處理 (例如 Markdown 渲染) ---
function finalizeAiMessageStreaming() {
    if (currentAiMessageElement && typeof marked !== 'undefined') {
        const fullTextWithPrefix = currentAiMessageElement.textContent;
        const prefix = "AI: ";
        if (fullTextWithPrefix.startsWith(prefix)) {
            const markdownText = fullTextWithPrefix.substring(prefix.length);
            currentAiMessageElement.innerHTML = prefix + marked.parse(markdownText);
        }
    }
    currentAiMessageElement = null; // 清除追蹤的 AI 訊息元素
}

// --- 發送訊息到後端並處理回應 ---
// --- 新的 sendMessageSSE 函式，用於串流聊天 ---
async function sendMessageSSE() {
    const messageText = userInput.value.trim();
    if (!messageText) return;

    appendMessage(messageText, 'user');
    userInput.value = '';
    sendButton.disabled = true; // 禁用發送按鈕，防止重複發送

    // 為 AI 的回應先創建一個「思考中」的訊息元素
    appendMessage("思考中...", 'ai', true); // isStreaming = true, 會設定 currentAiMessageElement

    const url = new URL(API_CHAT_STREAM_URL);
    url.searchParams.append('message', messageText);
    if (currentSessionId) {
        url.searchParams.append('session_id', currentSessionId);
    }

    // 關閉任何已存在的 EventSource 連線
    if (eventSource) {
        eventSource.close();
    }

    eventSource = new EventSource(url.toString());

    eventSource.onopen = () => {
        console.log("SSE Connection opened successfully.");
        // "思考中..." 的訊息已經由 appendMessage 處理了
    };

    eventSource.onmessage = (event) => {
        try {
            const eventData = JSON.parse(event.data);
            if (eventData.text) {
                // 如果 "思考中..." 還在，先清除它 (第一次收到有效 text chunk 時)
                if (currentAiMessageElement && 
                    (currentAiMessageElement.textContent === "AI: 思考中..." || currentAiMessageElement.textContent === "AI: ...")) {
                    currentAiMessageElement.textContent = "AI: "; 
                }
                appendMessage(eventData.text, 'ai', true); // isStreaming = true
            }
            // 更新 session_id (如果後端在串流中返回了新的)
            if (eventData.session_id && (!currentSessionId || currentSessionId !== eventData.session_id)) {
                currentSessionId = eventData.session_id;
                localStorage.setItem('ai_assistant_pwa_session_id', currentSessionId);
                updateSessionIdDisplay();
                console.log("從 SSE 更新 Session ID:", currentSessionId);
            }
        } catch (e) {
            console.error("解析 SSE message 數據錯誤:", e, "原始數據:", event.data);
            // 如果解析失敗，可能直接將原始數據作為文本片段追加，或顯示錯誤
            // appendMessage(event.data, 'ai', true); // 作為備用方案
        }
    };

    eventSource.addEventListener('end', (event) => {
        console.log("SSE Stream ended by server:", event.data);
        finalizeAiMessageStreaming(); // 最終處理 AI 訊息 (例如 Markdown 渲染)
        eventSource.close();
        sendButton.disabled = false; // 重新啟用發送按鈕
        
        // 你也可以在這裡從 event.data 解析 session_id (如果後端有傳)
        try {
            const endData = JSON.parse(event.data);
            if (endData.session_id && (!currentSessionId || currentSessionId !== endData.session_id)) {
                currentSessionId = endData.session_id;
                localStorage.setItem('ai_assistant_pwa_session_id', currentSessionId);
                updateSessionIdDisplay();
            }
        } catch(e) { /* 解析失敗則忽略 */ }
    });

    eventSource.addEventListener('error', (event) => { // 監聽後端發送的 'error' 事件
        console.error("SSE Custom Error Event from server:", event.data);
        finalizeAiMessageStreaming();
        try {
            const errorData = JSON.parse(event.data);
            appendMessage(`AI 處理錯誤: ${errorData.error} ${errorData.detail || ''}`, 'error', false);
        } catch (e) {
            appendMessage('AI 處理時發生未知伺服器錯誤。', 'error', false);
        }
        if (eventSource) eventSource.close(); // 確保關閉
        sendButton.disabled = false;
    });

    eventSource.onerror = (error) => { // 監聽 EventSource 連線本身的錯誤
        console.error("EventSource connection failed:", error);
        finalizeAiMessageStreaming();
        appendMessage('與 AI 的連接中斷或發生錯誤。', 'error', false);
        if (eventSource) eventSource.close(); // 確保關閉
        sendButton.disabled = false;
    };
}


// --- 上傳文件到後端的非同步函式 --- (Modify to use new URL constant)
async function uploadDocument() {
    // ... (獲取文件和 FormData 的邏輯保持原樣) ...
    const file = documentUploadInput.files[0];
    if (!file) { alert('請先選擇一個文件。'); return; }
    console.log(`正在上傳文件: ${file.name}, 類型: ${file.type}, 大小: ${file.size} bytes`);
    uploadStatusDiv.textContent = `正在上傳 "${file.name}"...`;
    uploadStatusDiv.className = 'upload-status';
    const formData = new FormData();
    formData.append('file', file);

    if (currentSessionId) {
        formData.append('session_id', currentSessionId);
    } else {
        // 如果前端也沒有 currentSessionId，可以傳一個空字串或不傳，讓後端生成
        // 這裡我們假設如果前端有，就傳送
         formData.append('session_id', ''); // 或者後端處理 Optional[str] = None
    }

    try {
        // 修改這裡使用新的 URL 常量
        const response = await fetch(API_UPLOAD_DOCUMENT_URL, { // <--- 使用 API_UPLOAD_DOCUMENT_URL
            method: 'POST',
            body: formData
        });
        const result = await response.json();

        // ... (處理後端回應和更新狀態的邏輯保持原樣) ...
        if (response.ok) {
            if (result.status === 'success') {
                 console.log(`文件上傳成功: ${result.message}`);
                 uploadStatusDiv.textContent = `成功上傳 "${file.name}"：${result.message}`;
                 uploadStatusDiv.className = 'upload-status success';
                 alert("文件已添加到知識庫的持久化存儲中。要讓新的文件內容在對話檢索中生效，您需要重新啟動後端服務器。");
            } else {
                 console.error('文件上傳失敗 (後端錯誤):', result.message);
                 uploadStatusDiv.textContent = `文件上傳失敗: ${result.message}`;
                 uploadStatusDiv.className = 'upload-status error';
            }
        } else {
            const errorText = result.message || await response.text();
            console.error('文件上傳失敗 (HTTP 錯誤):', response.status, errorText);
            uploadStatusDiv.textContent = `文件上傳失敗 (HTTP 錯誤: ${response.status})。`;
            uploadStatusDiv.className = 'upload-status error';
        }

    } catch (error) {
        console.error('文件上傳錯誤:', error);
        uploadStatusDiv.textContent = '文件上傳時發生網絡錯誤。請檢查後台服務器。';
        uploadStatusDiv.className = 'upload-status error';
    }
     documentUploadInput.value = '';
}


// --- 上傳音頻文件到後端的非同步函式 --- (Keep existing uploadAudioFile function)
// 這個函數用於上傳**已存在的**音頻文件，現在也用於發送錄製的音頻 Blob
async function uploadAudioFile(audioFileBlob, originalFileName = 'recorded_audio.webm') {
     // 如果傳入的是 Blob，將其轉為 File 對象
     const audioFile = audioFileBlob instanceof File ? audioFileBlob : new File([audioFileBlob], originalFileName, { type: audioFileBlob.type });

    const file = audioFile; // 使用 File 對象進行後續處理

    console.log(`正在上傳音頻文件: ${file.name}, 類型: ${file.type}, 大小: ${file.size} bytes`); // 除錯信息

    // 顯示上傳狀態 (使用同一個狀態顯示區域)
    uploadStatusDiv.textContent = `正在上傳音頻 "${file.name}"...`;
    uploadStatusDiv.className = 'upload-status';


    const formData = new FormData();
    formData.append('audio_file', file); // 'audio_file' 必須與後端 Fast API 音頻處理端點函數參數名一致

    if (currentSessionId) {
        formData.append('session_id', currentSessionId);
    } else {
        // 如果前端也沒有 currentSessionId，可以傳一個空字串或不傳，讓後端生成
        // 這裡我們假設如果前端有，就傳送
         formData.append('session_id', ''); // 或者後端處理 Optional[str] = None
    }

    try {
        const response = await fetch(API_UPLOAD_AUDIO_URL, {
            method: 'POST',
            body: formData
        });

        const result = await response.json();

        if (response.ok) {
             if (result && result.reply) {
                 console.log('音頻處理成功。');
                 // 將後端返回的摘要或結果顯示在聊天框中，作為 AI 的回應
                 appendMessage(`音頻處理結果：${result.reply}`, 'ai');
                 uploadStatusDiv.textContent = `音頻 "${file.name}" 處理完成。`;
                 uploadStatusDiv.className = 'upload-status success';
             } else {
                 console.error('音頻處理失敗 (後端回應格式錯誤):', result);
                 appendMessage('AI 助理：音頻處理成功但後台返回格式有誤。', 'ai-message');
                 uploadStatusDiv.textContent = `音頻 "${file.name}" 處理完成，但回應格式有誤。`;
                 uploadStatusDiv.className = 'upload-status error';
             }
        } else {
            const errorText = result.reply || await response.text();
            console.error('音頻上傳或處理失敗 (HTTP 錯誤):', response.status, errorText);
            appendMessage(`AI 助理：音頻處理失敗 (HTTP 錯誤): ${response.status}`, 'ai-message');
            uploadStatusDiv.textContent = `音頻 "${file.name}" 處理失敗 (HTTP 錯誤: ${response.status})。`;
            uploadStatusDiv.className = 'upload-status error';
        }

    } catch (error) {
        console.error('音頻上傳錯誤:', error);
        appendMessage('AI 助理：上傳音頻時發生網絡錯誤。', 'ai-message');
        uploadStatusDiv.textContent = '音頻上傳時發生網絡錯誤。請檢查後台服務器。';
        uploadStatusDiv.className = 'upload-status error';
    }
     // 清空文件選擇框的值 (對於錄音 Blob 不適用，對於文件 input 可以清空)
     if (audioFileBlob instanceof File) {
         audioUploadInput.value = ''; // 如果是從文件 input 觸發，清空 input
     }
}


// --- 新增：開始語音錄製函式 ---
async function startRecording() {
    console.log("嘗試獲取麥克風並開始錄音...");
    // 重置狀態顯示
    uploadStatusDiv.textContent = '';
    uploadStatusDiv.className = 'upload-status';

    try {
        // 請求訪問使用者麥克風 (需要使用者授權)
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true }); // 只請求音頻軌道
        console.log("成功獲取麥克風串流。");

        // 創建 MediaRecorder 實例
        // 'audio/webm; codecs=opus' 是一種常見的音頻格式
        mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm; codecs=opus' });
        audioChunks = []; // 清空之前的音頻數據片段

        // 監聽 'dataavailable' 事件：當 MediaRecorder 有音頻數據可用時觸發
        mediaRecorder.ondataavailable = event => {
            audioChunks.push(event.data); // 將獲取到的數據片段添加到 audioChunks 列表中
        };

        // 監聽 'stop' 事件：當錄音停止時觸發
        mediaRecorder.onstop = () => {
            console.log("錄音停止。處理數據...");
            // 將所有音頻數據片段組合成一個 Blob (Binary Large Object)
            const audioBlob = new Blob(audioChunks, { type: 'audio/webm; codecs=opus' });

            // 將錄製好的音頻 Blob 發送到後端進行處理
            // 我們直接呼叫 uploadAudioFile 函式來處理發送，它現在可以接收 Blob 或 File
            uploadAudioFile(audioBlob); // <--- 呼叫 modified uploadAudioFile 函式發送 Blob

            // 停止麥克風軌道，釋放資源 (非常重要！)
            stream.getTracks().forEach(track => track.stop());
        };

        mediaRecorder.onerror = event => {
             console.error("MediaRecorder 錯誤:", event.error);
             alert("錄音發生錯誤：" + event.error.name);
             isRecording = false;
             // 確保按鈕狀態恢復
             startRecordingButton.disabled = false;
             stopRecordingButton.disabled = true;
             uploadButton.disabled = false;
             audioUploadInput.disabled = false;
             uploadAudioButton.disabled = false;
              uploadStatusDiv.textContent = '錄音啟動失敗。';
              uploadStatusDiv.className = 'upload-status error';
              // 停止可能的麥克風軌道
             if (stream) { stream.getTracks().forEach(track => track.stop()); }

        };


        mediaRecorder.start(); // 開始錄音
        isRecording = true;
        console.log("錄音開始！");
        // 提示使用者開始說話
        uploadStatusDiv.textContent = '正在錄音... 請說話。';
        uploadStatusDiv.className = 'upload-status'; // 可以添加一個特殊的錄音中樣式


        // 更新按鈕狀態
        startRecordingButton.disabled = true; // 禁用開始按鈕
        stopRecordingButton.disabled = false; // 啟用停止按鈕
        uploadButton.disabled = true; // 錄音時禁用文件上傳按鈕
        audioUploadInput.disabled = true; // 錄音時禁用音頻文件選擇
        uploadAudioButton.disabled = true; // 錄音時禁用音頻文件上傳按鈕


    } catch (error) {
        // 處理獲取麥克風或啟動錄音時的錯誤 (例如使用者拒絕麥克風權限)
        console.error('啟動錄音錯誤:', error);
        alert('無法開始錄音。請確保已授予麥克風權限並檢查瀏覽器是否支援 MediaRecorder API。錯誤：' + error.message);
        isRecording = false;
        // 確保按鈕狀態恢復
        startRecordingButton.disabled = false;
        stopRecordingButton.disabled = true;
        uploadButton.disabled = false;
        audioUploadInput.disabled = false;
        uploadAudioButton.disabled = false;
         uploadStatusDiv.textContent = '錄音啟動失敗。';
         uploadStatusDiv.className = 'upload-status error';
    }
}

// --- 新增：停止語音錄製函式 ---
function stopRecording() {
    console.log("停止錄音指令接收到。");
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop(); // 停止錄音，會觸發 onstop 事件
        // onstop 事件處理器會處理數據並發送
        isRecording = false;

        // 更新按鈕狀態
        startRecordingButton.disabled = false;
        stopRecordingButton.disabled = true;
        uploadButton.disabled = false;
        audioUploadInput.disabled = false;
        uploadAudioButton.disabled = false;
         uploadStatusDiv.textContent = '錄音停止。處理中...'; // 更新狀態提示


    } else {
         console.log("MediaRecorder 未激活或不存在，無法停止。");
    }
}

// --- 新增：發送錄製的音頻數據到後端的函式 ---
// 這個函數從 Blob 或 File 對象構造 FormData 並發送，與 uploadAudioFile 邏輯合併
// async function sendRecordedAudio(formData) { ... } // 刪除這個獨立函數，邏輯合併到 uploadAudioFile

// --- 輔助函數：在 Newsletter 顯示區域下方添加原始來源連結列表 ---
function appendNewsletterSourceLinks(sourceResults, displayAreaElement) {
    if (!sourceResults || sourceResults.length === 0 || !displayAreaElement) {
        return; // 沒有來源連結或顯示區域無效則不做任何事
    }

    const sourceLinksDiv = document.createElement('div');
    sourceLinksDiv.classList.add('newsletter-source-links'); // 添加樣式類別 (不同於聊天來源連結)

    const titleElement = document.createElement('strong');
    titleElement.textContent = '來源連結:';
    sourceLinksDiv.appendChild(titleElement);

    const sourceList = document.createElement('ul'); // 使用無序列表顯示連結
    sourceLinksDiv.appendChild(sourceList);

    // 遍歷來源結果，為每個結果創建一個列表項和一個連結
    sourceResults.forEach(source => {
        // 確保來源對象有 title 和 link 屬性
        if (source.title && source.link) {
            const listItem = document.createElement('li');
            const linkElement = document.createElement('a');
            linkElement.href = source.link; // 設置連結 URL
            linkElement.textContent = source.title; // 設置連結文本為標題
            linkElement.target = '_blank'; // 在新分頁打開連結
            listItem.appendChild(linkElement);
            sourceList.appendChild(listItem);
        }
    });

    // 將連結列表區域添加到指定的顯示區域下方
    displayAreaElement.appendChild(sourceLinksDiv);
}

async function fetchAndDisplayNewsletter() {
    console.log("點擊 Newsletter 按鈕。"); // 記錄點擊事件
    // 獲取按鈕的當前文本
    const buttonText = showNewsletterButton.textContent;

    // --- 新增：控制顯示/隱藏的邏輯 ---
    if (buttonText === '隱藏 Newsletter') {
        // 如果按鈕顯示為 "隱藏"，則隱藏區域並改按鈕文本
        console.log("按鈕文本是 '隱藏 Newsletter'，隱藏區域。");
        newsletterContentDiv.style.display = 'none'; // <--- 隱藏區域
        showNewsletterButton.textContent = '顯示最新AI新聞 Newsletter'; // 改回顯示文本
        return; // 結束函式，不進行獲取操作
    }

    // 如果按鈕顯示為 "顯示最新AI新聞 Newsletter" 或 "載入中..."
    console.log("按鈕文本是 '顯示...' 或 '載入中...'，開始獲取/顯示。");
    newsletterContentDiv.style.display = 'block'; // <--- 確保區域是顯示的
    newsletterContentDiv.innerHTML = '<p>載入中...</p>'; // 顯示載入中提示
    showNewsletterButton.textContent = '載入中...'; // 改變按鈕文本表示載入中

    try {
        const response = await fetch(API_LATEST_NEWSLETTER_URL);

        if (!response.ok) {
            const errorText = await response.text();
            console.error('獲取 Newsletter 錯誤:', response.status, errorText);
            newsletterContentDiv.innerHTML = `<p>錯誤：未能獲取 Newsletter (狀態碼: ${response.status})。</p>`;
            return;
        }

        const result = await response.json(); // 期望返回 LatestNewsletterResponse 結構

        // 處理後端返回的結構
        if (result && result.newsletter) { // 如果返回的結果中有 newsletter 對象
             console.log("成功獲取到 Newsletter 內容。");
             // 使用 Marked.js 渲染 Markdown 內容為 HTML
             // 確保 marked 函式庫已在 index.html 中載入
             const renderedHTML = typeof marked !== 'undefined' ? marked.parse(result.newsletter.content) : `<pre>${result.newsletter.content}</pre>`; // 如果 marked 未載入，顯示原始文本
             newsletterContentDiv.innerHTML = renderedHTML; // 將渲染後的 HTML 設置到顯示區域

             // 檢查 result 中是否有 source_results 字段且不為空列表
             // source_results 字段現在直接來自 LatestNewsletterResponse 模型
             if (result.source_results && Array.isArray(result.source_results) && result.source_results.length > 0) {
                console.log(`获取到 ${result.source_results.length} 条原始来源链接。`);
                // 呼叫輔助函式，將連結列表添加到 newsletterContentDiv 下方
                appendNewsletterSourceLinks(result.source_results, newsletterContentDiv); // <--- 呼叫新函数
           } else {
               console.log("未获取到原始来源链接。");
           }

           showNewsletterButton.textContent = '隱藏 Newsletter'; // 獲取成功後，改變按鈕文本為隱藏

             // 可選：顯示時間戳等信息
             // const timestamp = result.newsletter.timestamp;
             // newsletterContentDiv.innerHTML = `<h3>最新 Newsletter (${new Date(timestamp).toLocaleString()})</h3>` + renderedHTML;

        } else if (result && result.message) { // 如果返回的結果中有 message 字段 (例如：未找到)
            console.log("後端返回 Newsletter 提示信息:", result.message);
            newsletterContentDiv.innerHTML = `<p>${result.message}</p>`;
            showNewsletterButton.textContent = '顯示最新AI新聞 Newsletter';
        }
         else { // 其他意外情況
            console.error('獲取 Newsletter 成功但回應格式不對:', result);
            newsletterContentDiv.innerHTML = '<p>錯誤：獲取 Newsletter 成功但後端回應格式不對。</p>';
            showNewsletterButton.textContent = '顯示最新AI新聞 Newsletter';

         }
    } catch (error) {
        console.error('獲取 Newsletter 錯誤:', error);
        newsletterContentDiv.innerHTML = '<p>錯誤：獲取 Newsletter 時發生網絡錯誤。</p>';
        showNewsletterButton.textContent = '顯示最新AI新聞 Newsletter';
    }
}

// --- 頁面載入完成時執行的程式碼 ---
document.addEventListener('DOMContentLoaded', function() {
    console.log("DOM 完全載入。");

    // 在頁面載入完成時，載入對話歷史
    loadChatHistory();

     // --- 修改事件監聽器以使用 sendMessageSSE ---
     sendButton.addEventListener('click', sendMessageSSE);
     userInput.addEventListener('keypress', function(event) {
         if (event.key === 'Enter' && !sendButton.disabled) { // 檢查按鈕是否已禁用
             event.preventDefault();
             sendMessageSSE();
         }
     });

     // 設置文件上傳按鈕的事件監聽器
     uploadButton.addEventListener('click', uploadDocument);

     // 設置音頻文件上傳按鈕的事件監聽器
     uploadAudioButton.addEventListener('click', uploadAudioFile);

     // --- 新增：設置語音錄製按鈕的事件監聽器 ---
     startRecordingButton.addEventListener('click', startRecording);
     stopRecordingButton.addEventListener('click', stopRecording);

    // --- 新增：設置顯示 Newsletter 按鈕的事件監聽器 ---
    showNewsletterButton.addEventListener('click', fetchAndDisplayNewsletter); // <--- 新增：監聽按鈕點擊

     // 可選：載入後將焦點設置到輸入框
     // userInput.focus();
});


// 如果您需要更早執行（例如在 DOM 尚未完全構建時），可以使用 'load' 事件
// window.addEventListener('load', function() {
//     console.log("頁面完全載入 (包括圖片樣式等)。");
//     // ... 載入歷史或其他邏輯 ...
// });