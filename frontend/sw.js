// sw.js
const CACHE_NAME = 'ai-assistant-pwa-cache-v1'; // 快取名稱，版本更新時可以修改它
const urlsToCache = [
    '/', // 如果你的伺服器將 / 指向 index.html
    '/index.html',
    '/style.css',
    '/app.js',
    '/manifest.json',
    '/icons/icon-192x192.png',
    '/icons/icon-512x512.png'
    // 你可以加入更多需要快取的靜態資源
];

// 安裝 Service Worker 並快取核心檔案
self.addEventListener('install', event => {
    console.log('Service Worker: Installing...');
    event.waitUntil(
        caches.open(CACHE_NAME)
            .then(cache => {
                console.log('Service Worker: Caching app shell');
                return cache.addAll(urlsToCache);
            })
            .then(() => {
                console.log('Service Worker: Install completed, app shell cached.');
                return self.skipWaiting(); // 強制新的 Service Worker 立即生效
            })
            .catch(error => {
                console.error('Service Worker: Failed to cache app shell during install:', error);
            })
    );
});

// 啟用 Service Worker，並清理舊快取
self.addEventListener('activate', event => {
    console.log('Service Worker: Activating...');
    event.waitUntil(
        caches.keys().then(cacheNames => {
            return Promise.all(
                cacheNames.map(cache => {
                    if (cache !== CACHE_NAME) {
                        console.log('Service Worker: Clearing old cache:', cache);
                        return caches.delete(cache);
                    }
                })
            );
        }).then(() => {
            console.log('Service Worker: Activate completed, old caches cleared.');
            return self.clients.claim(); // 讓 Service Worker 控制所有符合範圍的頁面
        })
    );
});

// 攔截網路請求，優先從快取提供資源 (Cache-First Strategy for app shell)
self.addEventListener('fetch', event => {
    // 我們只對 GET 請求進行快取處理
    if (event.request.method !== 'GET') {
        return;
    }

    // 對於 API 請求 (例如 /chat, /history)，總是嘗試網路優先，如果失敗則不提供離線內容 (除非你有特定策略)
    if (event.request.url.includes('/chat') || event.request.url.includes('/history') || event.request.url.includes('/upload')) {
        event.respondWith(
            fetch(event.request).catch(() => {
                // 你可以返回一個通用的離線錯誤訊息，但對於 API 通常不這麼做
                // return new Response(JSON.stringify({ error: 'Offline and no cache available for API request.' }), { headers: { 'Content-Type': 'application/json' }});
                console.warn('Service Worker: API request failed while offline:', event.request.url);
            })
        );
        return;
    }

    // 對於其他請求（主要是應用程式外殼的靜態資源）
    event.respondWith(
        caches.match(event.request)
            .then(response => {
                if (response) {
                    // console.log('Service Worker: Serving from cache:', event.request.url);
                    return response; // 從快取中找到，直接返回
                }
                // console.log('Service Worker: Fetching from network (and caching for next time):', event.request.url);
                // 如果快取中沒有，則從網路獲取，並存入快取供下次使用
                return fetch(event.request).then(
                    networkResponse => {
                        // 檢查是否成功獲取
                        if (!networkResponse || networkResponse.status !== 200 || networkResponse.type !== 'basic') {
                            return networkResponse;
                        }
                        // 將獲取到的資源複製一份存入快取
                        const responseToCache = networkResponse.clone();
                        caches.open(CACHE_NAME)
                            .then(cache => {
                                cache.put(event.request, responseToCache);
                            });
                        return networkResponse;
                    }
                ).catch(error => {
                    console.error('Service Worker: Fetching from network failed:', error);
                    // 你可以在這裡提供一個通用的離線頁面，如果連網路和快取都沒有
                    // return caches.match('/offline.html'); (你需要創建這個 offline.html)
                });
            })
    );
});