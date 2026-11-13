/* BorderPulse service worker — cache-first shell, network-first data. */
/* 中文：Service Worker —— 让应用可离线打开（PWA）。
   策略：界面外壳（HTML/CSS/JS/图标）cache-first（快，很少变）；
   data.json network-first（在线拿最新数据、顺手更新缓存，离线回退）。
   VERSION 变更 = 缓存整体作废重装；发布新版时改这里即可。 */
const VERSION = "bp-0.3.8";
const SHELL = [   // 中文：预缓存的界面外壳清单（安装时一次性拉全）
  "./", "./index.html", "./styles.css", "./app.js",
  "./manifest.webmanifest", "./data.json",
  "./icons/icon-192.png", "./icons/icon-512.png",
  "./icons/icon-maskable-512.png", "./icons/favicon-32.png",
  "./icons/apple-touch-icon.png",
];

self.addEventListener("install", (e) => {
  // 中文：安装——预缓存外壳，skipWaiting 立即接管（不等旧标签页关闭）
  e.waitUntil(caches.open(VERSION).then((c) => c.addAll(SHELL)).then(() => self.skipWaiting()));
});

self.addEventListener("activate", (e) => {
  // 中文：激活——删掉所有旧版本缓存，claim 立即控制已打开的页面
  e.waitUntil(
    caches.keys()
      .then((ks) => Promise.all(ks.filter((k) => k !== VERSION).map((k) => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

self.addEventListener("fetch", (e) => {
  // 中文：拦截同源请求，按类型分流（见文件头策略）
  const url = new URL(e.request.url);
  if (url.origin !== location.origin) return;   // 中文：跨源请求不碰
  if (url.pathname.endsWith("/data.json")) {
    /* freshest data when online, cached copy when offline */
    /* 中文：数据文件 network-first——先试网络，成功则回填缓存；
       断网时回退到缓存里的上次数据 */
    e.respondWith(
      fetch(e.request)
        .then((r) => {
          const copy = r.clone();
          caches.open(VERSION).then((c) => c.put(e.request, copy));
          return r;
        })
        .catch(() => caches.match(e.request))
    );
  } else {
    /* 中文：其余资源 cache-first——命中直接回；未命中才上网并缓存 */
    e.respondWith(
      caches.match(e.request).then(
        (hit) => hit ||
          fetch(e.request).then((r) => {
            const copy = r.clone();
            caches.open(VERSION).then((c) => c.put(e.request, copy));
            return r;
          })
      )
    );
  }
});
