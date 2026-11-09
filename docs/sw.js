/* BorderPulse service worker — cache-first shell, network-first data. */
const VERSION = "bp-0.1.2";
const SHELL = [
  "./", "./index.html", "./styles.css", "./app.js",
  "./manifest.webmanifest", "./data.json",
  "./icons/icon-192.png", "./icons/icon-512.png",
  "./icons/icon-maskable-512.png", "./icons/favicon-32.png",
  "./icons/apple-touch-icon.png",
];

self.addEventListener("install", (e) => {
  e.waitUntil(caches.open(VERSION).then((c) => c.addAll(SHELL)).then(() => self.skipWaiting()));
});

self.addEventListener("activate", (e) => {
  e.waitUntil(
    caches.keys()
      .then((ks) => Promise.all(ks.filter((k) => k !== VERSION).map((k) => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

self.addEventListener("fetch", (e) => {
  const url = new URL(e.request.url);
  if (url.origin !== location.origin) return;
  if (url.pathname.endsWith("/data.json")) {
    /* freshest data when online, cached copy when offline */
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
