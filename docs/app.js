/* BorderPulse — v0.3.0 UI.
   Data: docs/data.json built by src/build_docs.py (ImmD daily passenger traffic).
   Forecast: own NumPy histogram GBT (LOF-cleaned training), recursive 14 days;
   bands self-calibrated P50/85/95/99; per-day Markov 4-state distribution.
   Day sheet: intraday hour-of-day model (class templates x Erlang-C queue per hour).
   Queue tab: Monte Carlo over an M/M/c toy queue (Erlang-C), runs fully client-side.
   POI tab: TF-IDF + logistic-regression sentiment ranking from docs/nlp.json. */

const S = { d: null, dir: "Departure", tab: "cal", dateIdx: 0, cp: "LoWu", mc: null };

const LV = [
  { n: 1, cls: "lv1", glyph: "✓", cn: "畅顺", en: "Smooth" },
  { n: 2, cls: "lv2", glyph: "●", cn: "正常", en: "Normal" },
  { n: 3, cls: "lv3", glyph: "▲", cn: "繁忙", en: "Busy" },
  { n: 4, cls: "lv4", glyph: "◆", cn: "爆满", en: "Packed" },
];
const DOW = ["一", "二", "三", "四", "五", "六", "日"];
const DIR_CN = { Departure: "北上", Arrival: "南下" };

const $ = (s, el) => (el || document).querySelector(s);
const key = (cp, dir) => cp + "|" + dir;
const fmtN = (n) => Math.round(n).toLocaleString("en-US");
const fmtK = (n) => (n >= 10000 ? (n / 10000).toFixed(1) + "万" : fmtN(n));
const pct = (p) => Math.round(p * 100) + "%";
const lv = (n) => LV[n - 1];

function cpOrder(dir) {
  // fixed row order by typical flow (P50 of the direction), largest first
  return Object.values(S.d.stats).filter((s) => s.dir === dir)
    .sort((a, b) => b.q["50"] - a.q["50"]).map((s) => s.short);
}

/* ---------------- tabs ---------------- */
function setTab(t) {
  S.tab = t;
  document.querySelectorAll("main > section").forEach((el) => (el.style.display = "none"));
  $("#v-" + t).style.display = "";
  document.querySelectorAll("#tabbar button").forEach((b) => b.classList.toggle("on", b.dataset.t === t));
  if (t === "cmp") renderCmp();
  if (t === "risk") renderRisk();
  if (t === "poi") renderPoi();
}

/* ---------------- shared bits ---------------- */
function segDir(id) {
  return `<div class="seg" id="${id}">
    <button data-d="Departure" class="${S.dir === "Departure" ? "on" : ""}">北上 North</button>
    <button data-d="Arrival" class="${S.dir === "Arrival" ? "on" : ""}">南下 South</button>
  </div>`;
}
function bindSeg(id) {
  $("#" + id).addEventListener("click", (e) => {
    const b = e.target.closest("button");
    if (!b || b.dataset.d === S.dir) return;
    S.dir = b.dataset.d;
    renderCal(); renderCmp(); renderRisk();
  });
}
function chip(n) {
  const l = lv(n);
  return `<span class="chip ${l.cls}"><span class="g">${l.glyph}</span>${l.cn} ${l.en}</span>`;
}
function legend() {
  return `<div class="legend">${LV.map((l) =>
    `<span class="chip ${l.cls}"><span class="g">${l.glyph}</span>${l.cn}</span>`).join("")}</div>`;
}

/* local calendar date as ISO yyyy-mm-dd (forecast rows carry the same format) */
function todayISO() {
  const d = new Date();
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
}

/* ---------------- calendar heatmap ---------------- */
function calBlock(cps, start, count) {
  const day = (i) => S.d.forecast[key(cps[0], S.dir)][i];
  const range = `${day(start).date.slice(5).replace("-", "/")} – ${day(start + count - 1).date.slice(5).replace("-", "/")}`;
  let h = `<div class="calblock"><div class="caltitle">${range}</div><div class="cal">`;
  h += `<div class="cp"></div>`;
  for (let i = start; i < start + count; i++) {
    const r = day(i);
    const isToday = r.date === todayISO();
    h += `<div class="hd ${r.dow >= 5 ? "we" : ""}${isToday ? " today" : ""}">${DOW[r.dow]}<br>${r.date.slice(5).replace("-", "/")}</div>`;
  }
  cps.forEach((cp) => {
    const st = S.d.stats[key(cp, S.dir)];
    h += `<div class="cp"><b>${st.short}</b>${st.cn}</div>`;
    for (let i = start; i < start + count; i++) {
      const r = S.d.forecast[key(cp, S.dir)][i];
      const l = lv(r.level);
      const g = r.level >= 3 ? l.glyph : "";
      const isToday = r.date === todayISO();
      h += `<button class="cell g${r.level} ${r.dow >= 5 ? "we" : ""}${isToday ? " today" : ""}" data-cp="${cp}" data-i="${i}"
        aria-label="${st.cn} ${r.date} ${l.cn}">${g}</button>`;
    }
  });
  return h + `</div></div>`;
}

function renderCal() {
  const el = $("#v-cal");
  const cps = cpOrder(S.dir);
  const n = S.d.forecast[key(cps[0], S.dir)].length;
  let blocks = "";
  for (let s = 0; s < n; s += 7) blocks += calBlock(cps, s, Math.min(7, n - s));
  el.innerHTML = `<div class="row">${segDir("seg-cal")}${legend()}</div>`
    + blocks
    + `<p class="hint">点按格子看当天详情 · 粗边格 = 今天 · 颜色代表拥挤程度，按每个口岸自己的客流水平分级</p>`;
  bindSeg("seg-cal");
  el.addEventListener("click", (e) => {
    const c = e.target.closest(".cell");
    if (c) openCellSheet(c.dataset.cp, +c.dataset.i);
  });
}

/* Markov 4-state distribution row (r.mk = [p1..p4], from the state chain
   stepped from the last actual day to this date) */
function mkBlock(r) {
  if (!r.mk || !r.mk.length) return "";
  const segs = r.mk.map((p, i) => (p > 0.004 ? `<i class="g${i + 1}" style="width:${(p * 100).toFixed(1)}%"></i>` : "")).join("");
  const lg = r.mk.map((p, i) => `<span><i class="g${i + 1}"></i>${LV[i].cn} ${pct(p)}</span>`).join("");
  return `<div class="kv"><span>按历史规律推算</span></div>
    <div class="mkbar" role="img" aria-label="按历史规律推算的拥堵概率">${segs}</div>
    <div class="mklg">${lg}</div>`;
}

function openCellSheet(cp, i) {
  const st = S.d.stats[key(cp, S.dir)];
  const r = S.d.forecast[key(cp, S.dir)][i];
  const isToday = r.date === todayISO();
  const sp = S.d.spark[key(cp, S.dir)];
  const ai = sp && sp.flow.length ? sp.flow.length - 1 : -1;   // latest ImmD actual
  openSheet(`<h3>${st.cn} ${st.cp}</h3>
    <div class="sub">${r.date}（周${DOW[r.dow]}）${isToday ? " · 今天" : ""} · ${DIR_CN[S.dir]} · ${r.festive ? "节日期 " + r.festive : "非节日"}</div>
    ${chip(r.level)}
    <div class="kv"><span>预计过关人次</span><b class="num">${fmtN(r.base)}</b></div>
    ${ai >= 0 ? `<div class="kv"><span>最近实测（${sp.dates[ai]}）</span><b class="num">${fmtN(sp.flow[ai])}</b></div>` : ""}
    <div class="kv"><span>人流可能范围（八成日子在内）</span><b class="num">${fmtK(r.lo)} – ${fmtK(r.hi)}</b></div>
    <div class="kv"><span>拥挤概率</span><b class="num">${pct(r.p_busy)}</b></div>
    ${mkBlock(r)}
    <div class="kv"><span>本口岸参考（平日/偏忙/高峰）</span><b class="num small">${fmtK(st.q["50"])} / ${fmtK(st.q["85"])} / ${fmtK(st.q["95"])}</b></div>
    ${hourlyBlock(st, r)}`);
}

/* ---------------- intraday (hour-of-day) model ---------------- */
const fmtH = (h) => String(h).padStart(2, "0") + ":00";
const fmtOpen = (t) => {
  const h = Math.floor(t), m = Math.round((t - h) * 60);
  return String(h).padStart(2, "0") + ":" + String(m).padStart(2, "0");
};

function dayType(r) {
  if (r.festive) return { first: "hf", mid: "hm", last: "hl" }[r.ph || "hm"];
  return r.dow === 5 ? "sa" : r.dow === 6 ? "su" : "wk";
}

function hourlyWeights(st, r) {
  /* class template masked to this CP's clearing hours, normalised to sum 1 */
  const tpl = S.d.intraday.templates[st.profile][st.dir][dayType(r)];
  const [o, c] = S.d.intraday.open[st.short];
  const lo = Math.floor(o), hi = Math.ceil(c) - 1;
  let s = 0;
  const w = tpl.map((v, h) => (h >= lo && h <= hi ? (s += v, v) : 0));
  return w.map((v) => v / s);
}

function hourlyBlock(st, r) {
  const w = hourlyWeights(st, r);
  const MU = 2, T = 30;                       // service 1/2 per min per counter, 30-min line
  const q90 = 0.5 * (st.q["85"] + st.q["95"]);
  const c = Math.max(4, Math.ceil((q90 * 0.12 / 60 / MU) / 0.95));  // same calibration as risk tab
  const now = new Date();
  const isToday = r.date === todayISO();
  const nowH = now.getHours() + now.getMinutes() / 60;
  const met = [];
  w.forEach((x, h) => {
    if (x <= 0) return;
    met.push({ h, x, lam: (r.base * x) / 60, p30: pWait30((r.base * x) / 60, c, st.sigma, hash32(st.short + r.date + h)) });
  });
  const pk = met.reduce((a, b) => (b.lam > a.lam ? b : a));
  const lvl = (m) => (m.p30 < 0.05 ? 1 : m.p30 < 0.25 ? 2 : m.p30 < 0.6 ? 3 : 4);

  /* best 2h window within civil hours 08-21 (lowest combined queue risk);
     viewing TODAY -> only hours still ahead of the clock qualify (fall back
     to the full-day window late at night when <2 open hours remain) */
  const findBest = (from) => {
    let b = null;
    for (let j = 0; j < met.length - 1; j++) {
      if (met[j + 1].h !== met[j].h + 1) continue;
      if (met[j].h < Math.max(8, from) || met[j].h > 20) continue;
      const s = met[j].p30 + met[j + 1].p30;
      if (!b || s < b.s) b = { s, h: met[j].h };
    }
    return b;
  };
  const ahead = isToday ? findBest(Math.ceil(nowH)) : null;
  const best = ahead || findBest(8);

  const mx = Math.max(...w);
  const bars = w.map((x, h) => {
    const m = met.find((q) => q.h === h);
    if (!m) return `<i class="off"></i>`;
    const ht = Math.round(8 + (x / mx) * 92);
    const isBest = best !== null && (h === best.h || h === best.h + 1);
    return `<i class="g${lvl(m)}${isBest ? " best" : ""}" style="height:${ht}%"></i>`;
  }).join("");
  const axis = [0, 3, 6, 9, 12, 15, 18, 21].map((h) => `<span>${fmtH(h)}</span>`).join("");
  const [o, cH] = S.d.intraday.open[st.short];
  const nowLine = isToday ? `<i class="nowline" style="left:${(nowH / 24 * 100).toFixed(1)}%" aria-hidden="true"></i>` : "";
  const nowTag = isToday
    ? ` · <b>现在 ${String(now.getHours()).padStart(2, "0")}:${String(now.getMinutes()).padStart(2, "0")}</b>` : "";
  const pkState = !isToday ? "" : nowH >= pk.h + 1 ? "（已过）" : nowH >= pk.h ? "（现在）" : "";

  return `<div class="hr">几点出发最顺 · 逐小时（${fmtOpen(o)}–${fmtOpen(cH)} 通关）${nowTag}</div>
    <div class="hwrap"><div class="hours" role="img" aria-label="逐小时客流与排队风险">${bars}</div>${nowLine}</div>
    <div class="haxis">${axis}</div>
    <div class="advice" style="margin-top:8px">
      高峰 <b>${fmtH(pk.h)}–${fmtH(pk.h + 1)}</b>${pkState}（约 ${fmtK(pk.lam * 60)} 人次/时 · <span class="nw">排队超 30 分钟概率约 ${pct(pk.p30)}</span>）<br>
      ${best ? `${ahead ? "接下来" : "最顺"} <b>${fmtH(best.h)}–${fmtH(best.h + 2)}</b>，建议此时段出发` : ""}
    </div>
    <p class="hint">逐小时曲线为模型估算（入境处只公布每日总数，没有分时官方数据），非实测；
    颜色代表该时段排队超过 30 分钟的风险：绿=很低 · 黄=较低 · 橙=偏高 · 红=很高。</p>`;
}

/* deterministic per (cp, date, hour): the same day always renders the same
   numbers, so reopening a sheet never contradicts the previous look */
function hash32(s) {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619); }
  return h >>> 0;
}
function mulberry32(seed) {
  return function () {
    seed |= 0; seed = (seed + 0x6D2B79F5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function pWait30(lam, c, sigma, seed) {
  /* P(wait > 30 min) at an M/M/c counter fed by a lognormal-uncertain arrival
     rate: analytic Erlang-C tail per draw, saturated draws use the same
     triangular-dwell cap as the Monte Carlo tab (20/45/120 min -> P=.96). */
  const MU = 2, T = 30, N = 200, rnd = mulberry32(seed);
  let acc = 0;
  for (let t = 0; t < N; t++) {
    const z = Math.sqrt(-2 * Math.log(rnd() + 1e-12)) * Math.cos(2 * Math.PI * rnd());
    const lam1 = lam * Math.exp(sigma * z);
    const rho = lam1 / (c * MU);
    acc += rho < 0.97 ? erlangC(c, rho) * Math.exp(-(c * MU - lam1) * T) : 0.96;
  }
  return acc / N;
}

/* ---------------- compare ---------------- */
function renderCmp() {
  const el = $("#v-cmp");
  const cps = cpOrder(S.dir);
  const days = S.d.forecast[key(cps[0], S.dir)];
  if (S.dateIdx >= days.length) S.dateIdx = 0;
  let chips = `<div class="dates" id="dt">`;
  days.forEach((r, i) => {
    chips += `<button data-i="${i}" class="${i === S.dateIdx ? "on" : ""} ${r.dow >= 5 ? "we" : ""}">
      ${DOW[r.dow]} ${r.date.slice(5).replace("-", "/")}<small>${r.festive ? r.festive : ""}</small></button>`;
  });
  chips += `</div>`;

  const rows = cps.map((cp) => ({ cp, st: S.d.stats[key(cp, S.dir)], r: S.d.forecast[key(cp, S.dir)][S.dateIdx] }));
  const best = rows.slice().sort((a, b) => (a.r.level - b.r.level) || (a.r.p_busy - b.r.p_busy))[0];
  rows.forEach((x) => (x.diff = (x.r.base - best.r.base) / Math.max(best.r.base, 1)));
  /* small denominators (STK ~1k/day) blow up percentages — switch to a multiple */
  const relTxt = (d) => (d > 2 ? "×" + Math.round(1 + d) : "+" + Math.round(d * 100) + "%");

  let h = `<div class="row">${segDir("seg-cmp")}</div>${chips}`;
  rows.sort((a, b) => (b.r.level - a.r.level) || (b.r.p_busy - a.r.p_busy));
  rows.forEach((x, idx) => {
    const sp = spark(key(x.cp, S.dir));
    const isBest = x.cp === best.cp;
    h += `<div class="card ${isBest ? "pick" : ""}">
      <div class="cmp-h"><div class="nm">${x.st.short} <span>${x.st.cn} · ${x.st.cp}</span></div>${chip(x.r.level)}</div>
      <div class="stats">
        <div><div class="v num">${fmtK(x.r.base)}</div><div class="k">预计人次</div></div>
        <div><div class="v num">${pct(x.r.p_busy)}</div><div class="k">拥挤概率</div></div>
        <div><div class="v num">${isBest ? "—" : relTxt(x.diff)}</div><div class="k">对比最佳</div></div>
      </div>
      <div class="meter"><i style="width:${Math.min(100, x.r.p_busy * 100)}%"></i></div>
      ${sp}
      <div class="advice">${isBest
        ? `✅ 当日最佳选择 —— 预计 ${fmtN(x.r.base)} 人次，拥挤概率仅 ${pct(x.r.p_busy)}`
        : `比 <b>${best.st.cn}</b> 多约 ${fmtN(x.r.base - best.r.base)} 人次（${relTxt(x.diff)}），建议优先考虑 <b>${best.st.cn}</b>`}</div>
    </div>`;
  });
  el.innerHTML = h;
  bindSeg("seg-cmp");
  $("#dt").addEventListener("click", (e) => {
    const b = e.target.closest("button");
    if (b) { S.dateIdx = +b.dataset.i; renderCmp(); }
  });
}

function spark(k) {
  const s = S.d.spark[k];
  if (!s) return "";
  const w = 440, hgt = 44, pad = 3;
  const mx = Math.max(...s.flow), mn = Math.min(...s.flow);
  const pts = s.flow.map((v, i) => [
    pad + (i / (s.flow.length - 1)) * (w - 2 * pad),
    hgt - pad - ((v - mn) / Math.max(mx - mn, 1)) * (hgt - 2 * pad),
  ]);
  const last = pts[pts.length - 1];
  return `<svg class="spark" viewBox="0 0 ${w} ${hgt}" preserveAspectRatio="none" aria-label="近 60 日客流">
    <polyline points="${pts.map((p) => p[0].toFixed(1) + "," + p[1].toFixed(1)).join(" ")}"
      fill="none" stroke="var(--accent)" stroke-width="2" vector-effect="non-scaling-stroke"/>
    <circle cx="${last[0]}" cy="${last[1]}" r="3.5" fill="var(--accent)"/>
  </svg>`;
}

/* ---------------- risk (Monte Carlo queue) ---------------- */
function erlangC(c, rho) {
  /* P(all servers busy), M/M/c; rho = offered erlangs / c. Iterative, O(c). */
  const a = rho * c;
  let sum = 1, term = 1;
  for (let i = 1; i < c; i++) { term *= a / i; sum += term; }
  const last = term * a / c;
  return last / ((1 - rho) * sum + last);
}

function runMC() {
  const k = key(S.cp, S.dir);
  const st = S.d.stats[k], f = S.d.forecast[k][S.dateIdx];
  const MU = 2;                    // service rate per counter, 1/min (30 s mean)
  const PEAK_SHARE = 0.12;         // share of daily flow in the busiest hour
  const PEAK_MIN = 60, TRIALS = 3000, CAP = 120;

  /* counters calibrated so the P90 historical day runs at 95% utilization */
  const q90 = 0.5 * (st.q["85"] + st.q["95"]);
  const load90 = (q90 * PEAK_SHARE / PEAK_MIN) / MU;
  const c = Math.max(4, Math.ceil(load90 / 0.95));
  const wq = new Float64Array(TRIALS);
  for (let t = 0; t < TRIALS; t++) {
    const z = Math.sqrt(-2 * Math.log(Math.random() + 1e-12)) * Math.cos(2 * Math.PI * Math.random());
    const flow = f.base * Math.exp(st.sigma * z);
    const lam = flow * PEAK_SHARE / PEAK_MIN;
    const rho = lam / (c * MU);
    if (rho < 0.97) {
      wq[t] = erlangC(c, rho) / (c * MU - lam);
    } else {
      /* demand exceeds clearance capacity: model hall congestion as a
         triangular dwell distribution (20 / 45 / 120 min) */
      const u = Math.random();
      wq[t] = u < 0.25 ? 20 + 50 * Math.sqrt(u) : CAP - 86.6 * Math.sqrt(1 - u);
      wq[t] = Math.min(CAP, Math.max(20, wq[t]));
    }
  }
  wq.sort();
  S.mc = { wq, c, base: f.base, cap: st.q["95"], k, date: f.date };
  renderRiskResult();
}

function renderRiskResult() {
  const { wq, c, base, cap, date } = S.mc;
  const q = (p) => wq[Math.min(wq.length - 1, Math.floor(p * wq.length))];
  const p30 = wq.filter((x) => x > 30).length / wq.length;

  /* histogram 0..120 min, 24 bins + CDF */
  const BINS = 24, MAXM = 120, w = 460, hgt = 200, pl = 34, pr = 8, pt = 10, pb = 26;
  const bins = new Array(BINS).fill(0);
  wq.forEach((x) => { const b = Math.min(BINS - 1, Math.floor(x / MAXM * BINS)); bins[b]++; });
  const bm = Math.max(...bins);
  const x0 = (m) => pl + (m / MAXM) * (w - pl - pr);
  const y0 = (n) => hgt - pb - (n / bm) * (hgt - pt - pb);
  let bars = "", cdf = "";
  let cum = 0;
  const cdfPts = [];
  for (let i = 0; i < BINS; i++) {
    const bx = x0(i * MAXM / BINS), bw = (w - pl - pr) / BINS - 2; /* 2px spacer */
    bars += `<rect x="${bx.toFixed(1)}" y="${y0(bins[i]).toFixed(1)}" width="${bw.toFixed(1)}"
      height="${(hgt - pb - y0(bins[i])).toFixed(1)}" rx="3" fill="var(--accent)"/>`;
    cum += bins[i];
    cdfPts.push([bx + bw / 2, hgt - pb - (cum / wq.length) * (hgt - pt - pb)]);
    if (bins[i]) bars += `<text x="${(bx + bw / 2).toFixed(1)}" y="${hgt - pb + 12}" font-size="8"
      fill="var(--muted)" text-anchor="middle"></text>`;
  }
  cdf = `<polyline points="${cdfPts.map((p) => p[0].toFixed(1) + "," + p[1].toFixed(1)).join(" ")}"
    fill="none" stroke="var(--ink)" stroke-width="2"/>`;
  const mx30 = x0(30);
  const ticks = [0, 30, 60, 90, 120].map((m) =>
    `<line x1="${x0(m)}" y1="${pt}" x2="${x0(m)}" y2="${hgt - pb}" stroke="var(--grid)" stroke-width="1"/>
     <text x="${x0(m)}" y="${hgt - 8}" font-size="9" fill="var(--muted)" text-anchor="middle">${m}</text>`).join("");
  const cdfAt30 = wq.filter((x) => x <= 30).length / wq.length;
  const ylab = hgt - pb - cdfAt30 * (hgt - pt - pb);

  $("#mc-out").innerHTML = `
    <div class="tiles">
      <div class="tile"><div class="v num">${pct(p30)}</div><div class="k">排队超 30 分钟概率</div></div>
      <div class="tile"><div class="v num">${q(0.5).toFixed(1)}<span class="small"> min</span></div><div class="k">典型等待</div></div>
      <div class="tile"><div class="v num">${q(0.95).toFixed(0)}<span class="small"> min</span></div><div class="k">基本不会超过</div></div>
    </div>
    <div class="card">
      <svg id="riskplot" viewBox="0 0 ${w} ${hgt}" role="img" aria-label="等待时间分布">
        ${ticks}${bars}${cdf}
        <line x1="${mx30}" y1="${pt}" x2="${mx30}" y2="${hgt - pb}" stroke="var(--critical)"
          stroke-width="1.5" stroke-dasharray="5 4"/>
        <circle cx="${mx30}" cy="${ylab}" r="4" fill="var(--critical)"/>
        <text x="${Math.min(mx30 + 6, w - 90)}" y="${Math.max(pt + 10, ylab - 8)}" font-size="10"
          fill="var(--ink)" font-weight="700">30 分钟内排完：${pct(cdfAt30)}</text>
      </svg>
      <div class="legend small" style="margin-top:8px">
        <span class="small">▮ 排队时长分布（模拟 3000 次）</span><span class="small">— 不超过比例曲线</span>
      </div>
    </div>
    <p class="hint">说明：以上为简化排队模型的估算，演示用，实际以现场为准（正式版应由实测清关数据标定）。
    主要假设：最忙 1 小时约占全天 12% 客流 · 每个柜台平均 30 秒办理一位 · 柜台数 ${c} 个（按该口岸历史高峰日的负荷估出）·
    严重超出负荷时按大厅滞留 20–120 分钟估算 · 当日人次本身有随机波动（预计 ${fmtN(base)}，该口岸历史高峰日约 ${fmtN(cap)}）。</p>`;
}

function renderRisk() {
  const el = $("#v-risk");
  const cps = cpOrder(S.dir);
  const days = S.d.forecast[key(cps[0], S.dir)];
  if (!S.d.stats[key(S.cp, S.dir)]) S.cp = cps[0];
  const st = S.d.stats[key(S.cp, S.dir)];
  let h = `<div class="row">${segDir("seg-rk")}</div>
    <div class="card">
      <div class="row" style="margin-bottom:8px">
        <select id="sel-cp">${cps.map((cp) => {
          const s = S.d.stats[key(cp, S.dir)];
          return `<option value="${cp}" ${cp === S.cp ? "selected" : ""}>${s.cn} ${s.short}</option>`;
        }).join("")}</select>
        <select id="sel-dt">${days.map((r, i) =>
          `<option value="${i}" ${i === S.dateIdx ? "selected" : ""}>${r.date}（周${DOW[r.dow]}${r.festive ? " · " + r.festive : ""}）</option>`).join("")}</select>
      </div>
      <div class="kv small" style="display:flex;justify-content:space-between;padding:6px 0;border-top:1px solid var(--grid)">
        <span class="hint">预计当日 ${fmtN(S.d.forecast[key(S.cp, S.dir)][S.dateIdx].base)} 人次</span>
        <span class="hint">历史高峰日约 ${fmtN(st.q["95"])} 人次</span></div>
      <button class="runbtn" id="run">▶ 模拟这一天的排队（3000 次）</button>
    </div>
    <div id="mc-out">${S.mc ? "" : '<p class="hint">选好口岸和日期，点上方按钮模拟排队情况。</p>'}</div>`;
  el.innerHTML = h;
  bindSeg("seg-rk");
  $("#sel-cp").addEventListener("change", (e) => { S.cp = e.target.value; S.mc = null; renderRisk(); });
  $("#sel-dt").addEventListener("change", (e) => { S.dateIdx = +e.target.value; S.mc = null; renderRisk(); });
  $("#run").addEventListener("click", () => {
    const b = $("#run"); b.disabled = true; b.textContent = "模拟中 Running…";
    setTimeout(() => { runMC(); const rb = $("#run"); if (rb) { rb.disabled = false; rb.textContent = "▶ 再跑一次 Re-run"; } }, 30);
  });
  if (S.mc) renderRiskResult();
}

/* ---------------- POI: sentiment-ranked off-peak attractions ---------------- */
function renderPoi() {
  const el = $("#v-poi");
  const nlp = S.d.nlp;
  if (!nlp || !nlp.attractions) {
    el.innerHTML = '<p class="hint">景点数据未生成（docs/nlp.json 缺失）。</p>';
    return;
  }
  const pretty = (t) => (t.startsWith("NOT_") ? "not·" + t.slice(4) : t);
  let h = `<p class="hint" style="margin:2px 0 10px">
    用 AI 读了 <b>${fmtN(nlp.train_reviews)}</b> 条真实游客评论（TripAdvisor 景点点评），自动判断每条是好评还是差评，
    据此为 12 个香港景点排出好评率——口岸人多的日子，挑个口碑好、相对清闲的去处。
    香港景点的评论为<b>演示用示例</b>（详见「关于」页；判断模型只用真实评论训练，留出测试准确率 <b>${pct(nlp.acc)}</b>）。</p>`;
  nlp.attractions.forEach((a, i) => {
    h += `<div class="card${i === 0 ? " pick" : ""}">
      <div class="cmp-h"><div class="nm">${a.cn} <span>${a.en}</span></div>
        <div class="v num" style="font-size:18px;font-weight:700">${pct(a.pos)}</div></div>
      <div class="meter"><i style="width:${Math.round(a.pos * 100)}%"></i></div>
      <div class="terms">${a.terms.map((t) => `<span>${pretty(t)}</span>`).join("")}</div>
      <div class="advice">💡 ${a.tip}</div>
    </div>`;
  });
  h += `<p class="hint">好评率 = AI 判为好评的评论占比；
    高亮词是该景点好评里最有代表性的用词；「not·词」表示否定语气（例如 not·worth =「不值得」）。</p>`;
  el.innerHTML = h;
}

/* ---------------- sheet ---------------- */
function openSheet(html) {
  $("#sheet").innerHTML = `<i class="grab" aria-hidden="true"></i><button class="close" aria-label="关闭">✕</button>` + html;
  $("#sheet-bg").classList.add("show");
  $("#sheet").classList.add("show");
}
function closeSheet() {
  $("#sheet-bg").classList.remove("show");
  $("#sheet").classList.remove("show");
}

/* ---------------- boot ---------------- */
async function boot() {
  const r = await fetch("./data.json");
  S.d = await r.json();
  const cps = cpOrder(S.dir);
  S.cp = cps[0];
  const fs = S.d.forecast[key(S.cp, S.dir)];
  const sat = fs.findIndex((x) => x.dow === 5);
  S.dateIdx = sat >= 0 ? sat : 0;
  $("#meta").textContent =
    `入境处数据更新至 ${S.d.data_through} · ${S.d.model ? "提前 14 天预测 · " : ""}v${S.d.version}`;
  renderCal();
  $("#tabbar").addEventListener("click", (e) => {
    const b = e.target.closest("button");
    if (b) setTab(b.dataset.t);
  });
  $("#sheet-bg").addEventListener("click", closeSheet);
  document.addEventListener("click", (e) => { if (e.target.closest("#sheet .close")) closeSheet(); });
  if ("serviceWorker" in navigator && (location.protocol === "https:" || location.hostname === "localhost")) {
    navigator.serviceWorker.register("./sw.js").catch(() => {});
    /* when an updated worker takes over, offer a one-tap reload
       (first install claims silently — no toast on a visitor's first load) */
    let refreshing = false;
    const hadController = !!navigator.serviceWorker.controller;
    navigator.serviceWorker.addEventListener("controllerchange", () => {
      if (refreshing || !hadController) return;
      refreshing = true;
      const t = document.createElement("div");
      t.id = "toast";
      t.innerHTML = `新版本 v${S.d.version} 已就绪
        <button onclick="location.reload()">刷新</button>
        <button aria-label="关闭" style="background:none;padding:6px;color:inherit;font-weight:400">✕</button>`;
      document.body.appendChild(t);
      t.querySelector("button[aria-label=关闭]").onclick = () => t.remove();
      setTimeout(() => t.remove(), 12000);
      requestAnimationFrame(() => t.classList.add("show"));
    });
  }
}
boot();
