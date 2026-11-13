/* BorderPulse — v0.3.7 UI (languages: 简体 / 繁體 / English, switchable in the app bar).
   Primary author: Lu Zengyi (front end, trilingual UI, accessibility pass).
   Data: docs/data.json built by src/build_docs.py (ImmD daily passenger traffic).
   Forecast: own NumPy histogram GBT (LOF-cleaned training), recursive 14 days;
   bands self-calibrated P50/85/95/99; per-day Markov 4-state distribution.
   Day sheet: intraday hour-of-day model (class templates x Erlang-C queue per hour).
   Queue tab: Monte Carlo over an M/M/c toy queue (Erlang-C), runs fully client-side.
   POI tab: TF-IDF + logistic-regression sentiment ranking from docs/nlp.json.

   【中文说明 —— 前端主程序】
   这是单页应用（PWA）的全部交互逻辑，无框架、无构建步骤，直接由
   index.html 引入。数据只 fetch 一次（docs/data.json，由
   src/build_docs.py 离线生成），之后所有计算都在浏览器里完成。
   五个页签：
     日历 cal   —— 14 天 × 8 口岸拥挤热力图，点格子弹出当日详情
     对比 cmp   —— 选定一天，8 个口岸并排比较并给出当日最佳
     排队 risk  —— 蒙特卡洛排队模拟（Erlang-C + 饱和三角分布，纯前端）
     景点 poi   —— 情感分析排序的错峰景点（数据来自 nlp.json）
     关于 about —— 三语静态说明页（ABOUT 字典整段渲染）
   文件结构（自上而下）：i18n 词典 → 状态与小工具 → 日历 → 日内模型 →
   对比 → 排队模拟 → 景点 → 弹层 → 启动流程。 */

/* ---------------- i18n ---------------- */
/* 中文：三语词典。T 的每个条目是 [简体, 繁體, English] 三元组；
   需要拼变量的条目写成函数（调用时传 live 值）。
   LANG 存在 localStorage("bp-lang")，刷新后保持。 */
let LANG = "sc";
try { const l = localStorage.getItem("bp-lang"); if (l === "tc" || l === "en") LANG = l; } catch (e) {}
const LI = () => (LANG === "tc" ? 1 : LANG === "en" ? 2 : 0);
const pick = (trio) => trio[LI()];

/* dictionary: each entry is [简体, 繁體, English]; functions receive live values */
const T = {
  title: ["BorderPulse 口岸脉搏 — 香港口岸错峰过关助手", "BorderPulse 口岸脈搏 — 香港口岸錯峰過關助手", "BorderPulse — HK border congestion forecaster"],
  brand: ["口岸脉搏", "口岸脈搏", ""],
  loading: ["载入数据中…", "載入數據中…", "Loading data…"],
  meta: [(d) => `入境处数据更新至 ${d} · 提前 14 天预测`, (d) => `入境處數據更新至 ${d} · 提前 14 天預測`, (d) => `ImmD data through ${d} · 14-day forecast`],
  langAria: ["语言", "語言", "Language"],
  tab: { cal: ["日历", "日曆", "Calendar"], cmp: ["对比", "對比", "Compare"], risk: ["排队", "排隊", "Queue"], poi: ["景点", "景點", "Spots"], about: ["关于", "關於", "About"] },
  tabAria: { cal: ["日历", "日曆", "Calendar"], cmp: ["对比", "對比", "Compare"], risk: ["排队模拟", "排隊模擬", "Queue simulator"], poi: ["错峰景点", "錯峰景點", "Off-peak spots"], about: ["关于", "關於", "About"] },
  dirN: ["北上 North", "北上 North", "Northbound"], dirS: ["南下 South", "南下 South", "Southbound"],
  calHint: ["点格子看详情 · 粗边格 = 今天 · 颜色越红越挤", "點格子看詳情 · 粗邊格 = 今天 · 顏色越紅越擠", "Tap a cell for details · outlined cell = today · redder = busier"],
  today: ["今天", "今天", "today"],
  festive: [(f) => `节日期 ${f}`, (f) => `節日期 ${f}`, (f) => `${f} holiday`],
  nonFestive: ["非节日", "非節日", "not a holiday"],
  dowFull: [(i) => `周${["一", "二", "三", "四", "五", "六", "日"][i]}`, (i) => `週${["一", "二", "三", "四", "五", "六", "日"][i]}`, (i) => ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][i]],
  estCrossings: ["预计过关人次", "預計過關人次", "Est. crossings"],
  latestActual: [(d) => `最近实测（${d}）`, (d) => `最近實測（${d}）`, (d) => `Latest actual (${d})`],
  range80: ["人流可能范围（八成日子在内）", "人流可能範圍（八成日子在內）", "Likely range (80% of days)"],
  busyP: ["拥挤概率", "擁擠概率", "Busy chance"],
  fromHist: ["按历史规律推算", "按歷史規律推算", "From historical patterns"],
  fromHistAria: ["按历史规律推算的拥堵概率", "按歷史規律推算的擁堵概率", "Congestion odds from historical patterns"],
  cpRef: ["本口岸参考（平日/偏忙/高峰）", "本口岸參考（平日/偏忙/高峰）", "This crossing (typical/busy/peak)"],
  bestHours: [(o, c) => `几点出发最顺（${o}–${c} 通关）`, (o, c) => `幾點出發最順（${o}–${c} 過關）`, (o, c) => `Best hours to cross (${o}–${c})`],
  nowTag: [(hm) => `现在 ${hm}`, (hm) => `現在 ${hm}`, (hm) => `Now ${hm}`],
  pkPast: ["（已过）", "（已過）", " (past)"], pkNow: ["（现在）", "（現在）", " (now)"],
  peak: [(a, b, st, rate, p) => `高峰 <b>${a}–${b}</b>${st}（约 ${rate} 人次/时 · <span class="nw">排队超 30 分钟概率约 ${p}</span>）`,
         (a, b, st, rate, p) => `高峰 <b>${a}–${b}</b>${st}（約 ${rate} 人次/時 · <span class="nw">排隊超 30 分鐘概率約 ${p}</span>）`,
         (a, b, st, rate, p) => `Peak <b>${a}–${b}</b>${st} (~${rate}/h · <span class="nw">~${p} chance of a 30-min+ queue</span>)`],
  adviceAhead: [(a, b) => `接下来 <b>${a}–${b}</b>，建议此时段出发`, (a, b) => `接下來 <b>${a}–${b}</b>，建議此時段出發`, (a, b) => `Coming up <b>${a}–${b}</b> — go in this window`],
  adviceBest: [(a, b) => `最顺 <b>${a}–${b}</b>，建议此时段出发`, (a, b) => `最順 <b>${a}–${b}</b>，建議此時段出發`, (a, b) => `Smoothest window <b>${a}–${b}</b> — go in this window`],
  hourlyHint: ["逐小时为模型估算，非实测 · 颜色越红，排队超 30 分钟风险越高", "逐小時為模型估算，非實測 · 顏色越紅，排隊超 30 分鐘風險越高", "Hourly figures are model estimates, not measurements · redder = higher risk of a 30-min+ queue"],
  hoursAria: ["逐小时客流与排队风险", "逐小時客流與排隊風險", "Hourly flow and queue risk"],
  estN: ["预计人次", "預計人次", "Est. crossings"],
  vsBest: ["对比最佳", "對比最佳", "vs best"],
  pickTxt: [(n, p) => `✅ 当日最佳选择 —— 预计 ${n} 人次，拥挤概率仅 ${p}`, (n, p) => `✅ 當日最佳選擇 —— 預計 ${n} 人次，擁擠概率僅 ${p}`, (n, p) => `✅ Best pick today — about ${n} crossings, only ${p} busy chance`],
  worseTxt: [(n, cp, r) => `比 <b>${cp}</b> 多约 ${n} 人次（${r}），建议优先考虑 <b>${cp}</b>`, (n, cp, r) => `比 <b>${cp}</b> 多約 ${n} 人次（${r}），建議優先考慮 <b>${cp}</b>`, (n, cp, r) => `About ${n} more crossings than <b>${cp}</b> (${r}) — <b>${cp}</b> is the better pick`],
  sparkAria: ["近 60 日客流", "近 60 日客流", "Daily flow, last 60 days"],
  estDay: [(n) => `预计当日 ${n} 人次`, (n) => `預計當日 ${n} 人次`, (n) => `Est. ${n} crossings that day`],
  histPeak: [(n) => `历史高峰日约 ${n} 人次`, (n) => `歷史高峰日約 ${n} 人次`, (n) => `Peak days historically ~${n}`],
  runBtn: ["▶ 模拟这一天的排队", "▶ 模擬這一天的排隊", "▶ Simulate this day's queue"],
  running: ["模拟中…", "模擬中…", "Simulating…"],
  rerun: ["▶ 再跑一次", "▶ 再跑一次", "▶ Run again"],
  mcPlaceholder: ["选口岸和日期，点上方按钮开始", "選口岸和日期，點上方按鈕開始", "Pick a crossing and date, then tap the button above"],
  tileP30: ["排队超 30 分钟概率", "排隊超 30 分鐘概率", "30-min+ queue chance"],
  tileTyp: ["典型等待", "典型等待", "Typical wait"],
  tileP95: ["基本不会超过", "基本不會超過", "Rarely above"],
  cleared30: [(p) => `30 分钟内排完：${p}`, (p) => `30 分鐘內排完：${p}`, (p) => `Cleared within 30 min: ${p}`],
  lgDist: ["▮ 等待时长分布", "▮ 等待時長分佈", "▮ Wait-time spread"],
  lgCdf: ["— 累计排完比例", "— 累計排完比例", "— share cleared"],
  riskAria: ["等待时间分布", "等待時間分佈", "Wait-time distribution"],
  mcHint: ["简化排队模型估算，实际以现场为准 · 假设见「关于」页", "簡化排隊模型估算，實際以現場為準 · 假設見「關於」頁", "Simplified queue-model estimate — actual queues may differ · assumptions in About"],
  poiIntro: [(n, c) => `根据 <b>${n}</b> 条真实游客评论排出的 ${c} 个景点好评率 · 香港部分为演示示例（见「关于」）`, (n, c) => `根據 <b>${n}</b> 條真實遊客評論排出的 ${c} 個景點好評率 · 香港部分為演示示例（見「關於」）`, (n, c) => `${c} spots ranked by positive-review rate from <b>${n}</b> real visitor reviews · HK picks are demo examples (see About)`],
  poiFoot: ["高亮词 = 好评里最有代表性的用词 · not· = 否定（not·worth = 不值得）", "高亮詞 = 好評裡最有代表性的用詞 · not· = 否定（not·worth = 不值得）", "Highlighted words = terms most typical of positive reviews · not· = negation (not·worth)"],
  poiMissing: ["景点数据未生成（docs/nlp.json 缺失）。", "景點數據未生成（docs/nlp.json 缺失）。", "Attraction data not generated (docs/nlp.json missing)."],
  closeAria: ["关闭", "關閉", "Close"],
  newVer: [(v) => `新版本 v${v} 已就绪`, (v) => `新版本 v${v} 已就緒`, (v) => `New version v${v} ready`],
  refresh: ["刷新", "刷新", "Refresh"],
};
const t = (k, ...a) => {
  /* 中文：取词条。k 是键名；子词典（如 tab 标签）第一个参数是子键。
     返回值若是函数则立即以剩余参数调用（支持插值）。 */
  let e = T[k];
  if (!e) return k;
  if (!Array.isArray(e)) { e = e[a.shift()]; if (!e) return k; }   // keyed sub-dictionaries (tab labels)
  const v = e[LI()];
  return typeof v === "function" ? v(...a) : v;
};

/* control-point display names per language (stats keys use the short code) */
/* 中文：口岸显示名（三语）。data.json 的键用短代码（LoWu/LMC…），
   CPN 负责把短代码翻成各语言的显示名。 */
const CPN = {
  LoWu: ["罗湖", "羅湖", "Lo Wu"],
  "LMC-Spur": ["落马洲支线", "落馬洲支線", "Lok Ma Chau Spur"],
  SZB: ["深圳湾", "深圳灣", "Shenzhen Bay"],
  HZMB: ["港珠澳大桥", "港珠澳大橋", "HZM Bridge"],
  ERL: ["高铁西九龙", "高鐵西九龍", "West Kowloon"],
  HYW: ["香园围", "香園圍", "Heung Yuen Wai"],
  LMC: ["落马洲", "落馬洲", "Lok Ma Chau"],
  MKT: ["文锦渡", "文錦渡", "Man Kam To"],
};
const cpn = (short) => (CPN[short] ? CPN[short][LI()] : short);

/* POI traditional names + tips (data.json carries simplified tips) */
/* 中文：景点的繁体名与繁/英贴士（data.json 只带简体贴士，
   这里做显示层的语言映射）。poiName/poiTip 按当前语言取值。 */
const POI_TC = {
  "Hong Kong Disneyland": "香港迪士尼", "Big Buddha & Ngong Ping 360": "天壇大佛 · 昂坪360",
  "M+ Museum, West Kowloon": "M+ 博物館", "Victoria Peak & Peak Tram": "太平山頂",
  "Cheung Chau Island": "長洲", "Hong Kong Palace Museum": "香港故宮文化博物館",
  "Temple Street Night Market": "廟街夜市", "Tai Kwun": "大館", "Ocean Park": "海洋公園",
  "Lamma Island": "南丫島", "Mong Kok Street Markets": "旺角街市", "Avenue of Stars": "星光大道",
};
const POI_TIP = {
  "Hong Kong Disneyland": ["週末與內地黃金週爆滿；平日園區輕鬆一天", "Packed on weekends and mainland Golden Weeks; relaxed on weekdays"],
  "Big Buddha & Ngong Ping 360": ["週末纜車排隊 1 小時+，工作日早上 9 點前最順", "Weekend cable-car queues top 1 hour; before 9am on weekdays is smoothest"],
  "M+ Museum, West Kowloon": ["週五夜間延長開放，人流反而最少", "Late Friday openings draw the smallest crowds"],
  "Victoria Peak & Peak Tram": ["工作日上午首班纜車，避開日落人潮", "Take the first weekday-morning tram; skip the sunset rush"],
  "Cheung Chau Island": ["週末渡輪大排長龍，工作日班次寬鬆", "Weekend ferries queue long; weekday sailings are relaxed"],
  "Hong Kong Palace Museum": ["週一休館；工作日下午最空", "Closed Mondays; weekday afternoons are quietest"],
  "Temple Street Night Market": ["每晚 19–21 點最熱鬧也最擠，21 點後漸散", "Liveliest and most crowded 7–9pm; thins out after 9"],
  "Tai Kwun": ["週末展覽排隊，工作日午間基本即入", "Weekend exhibition queues; weekday midday is near walk-in"],
  "Ocean Park": ["週六下午家庭客最多，工作日從容", "Saturday afternoons draw the most families; weekdays are easy"],
  "Lamma Island": ["週末遠足客集中，工作日近於包島", "Hikers crowd weekends; on weekdays the island is nearly yours"],
  "Mong Kok Street Markets": ["週五晚與週末晚間水洩不通，工作日白日可逛", "Packed Friday and weekend nights; manageable on weekday days"],
  "Avenue of Stars": ["工作日清晨看維港，晚上燈光秀前 1 小時最擠", "Weekday dawn for the harbour view; most crowded in the hour before the light show"],
};
const poiName = (a) => (LANG === "en" ? a.en : LANG === "tc" ? (POI_TC[a.en] || a.cn) : a.cn);
const poiTip = (a) => (LANG === "sc" ? a.tip : POI_TIP[a.en] ? POI_TIP[a.en][LI() - 1] : a.tip);

/* about tab (rendered whole per language) */
const ABOUT = {
  sc: `<div class="card">
        <h3 style="margin-top:2px">这是什么</h3>
        <p>周末北上、黄金周返乡、长假日返港——哪天挤、哪个口岸顺、几点出发不用排长队？
        BorderPulse 用香港入境处每天公布的过关人次，<b>提前 14 天</b>预估 8 个陆路 / 跨界口岸的拥挤程度，
        帮你决定：换日子？换口岸？换方向？还是照常出发、挑个顺的时段？</p>
        <h3>怎么用</h3>
        <ul>
          <li><b>日历</b> —— 14 天 × 8 口岸一览，一眼挑出最顺的日子</li>
          <li><b>对比</b> —— 同一天各口岸放一起，直接告诉你当日最佳选择</li>
          <li><b>排队</b> —— 选口岸和日期，模拟排队：超半小时的概率、一般要排多久</li>
          <li><b>景点</b> —— 人多的日子，看看口碑好又相对清闲的去处</li>
        </ul>
        <h3>技术说明</h3>
        <p class="flowline">
          入境处日客流 → <b>LOF 异常清洗</b>（先剔除台风 / 暴雨等异常日）→ <b>GBT 预测</b>（直方图梯度提升树，
          学习节假日 / 星期规律，递归 14 天）→ <b>马尔可夫 4 档拥挤状态</b>（按历史流转规律给出独立推算）→
          <b>排队模拟</b>（Erlang-C 排队论 + 蒙特卡洛）→ 推荐（哪天 / 哪个口岸 / 哪个方向 / 几点）<br>
          旁路：<b>TF-IDF + 逻辑回归情感分析</b> → 错峰景点推荐
        </p>
        <p class="hint">全部模型为自写实现（NumPy 为主，scikit-learn 仅作对照基准）：主预测模型经滚动回测，
        误差比「按星期几估」的基线低约 31%，亦优于 sklearn 对照实现；评测细节见课程报告。
        逐小时曲线为口岸类型模板 + 排队模型估算（入境处不公布分时数据）。
        排队模拟主要假设：最忙 1 小时约占全天 12% 客流 · 每柜台平均 30 秒办理一位 ·
        柜台数按该口岸历史高峰日负荷估出 · 严重超载时按大厅滞留 20–120 分钟估算。</p>
        <h3>数据来源</h3>
        <ul>
          <li><a href="https://data.gov.hk/en-data/dataset/hk-immd-set5-statistics-daily-passenger-traffic">入境处 · 每日各管制站旅客人次</a>（核心时序，2021-01 至今）</li>
          <li><a href="https://data.gov.hk/en-data/dataset/hk-customs-ced_stat-passenger-clearance">海关 · 管制站旅客吞吐量</a>（年度容量参考）</li>
          <li>TripAdvisor 景点评论（好评率模型训练用，HuggingFace 镜像）</li>
        </ul>
        <h3>声明</h3>
        <p>预测仅基于历史统计规律，不构成出行保证；口岸临时管制、天气及突发事件不在模型内。
        景点页的香港评论为<b>演示用示例</b>（好评率判断模型本身仅用真实评论训练）。
        MATH7005 课程项目数据产品。</p>
      </div>`,
  tc: `<div class="card">
        <h3 style="margin-top:2px">這是什麼</h3>
        <p>週末北上、黃金週返鄉、長假日返港——哪天擠、哪個口岸順、幾點出發不用排長隊？
        BorderPulse 用香港入境處每天公佈的過關人次，<b>提前 14 天</b>預估 8 個陸路 / 跨界口岸的擁擠程度，
        幫你決定：換日子？換口岸？換方向？還是照常出發、挑個順的時段？</p>
        <h3>怎麼用</h3>
        <ul>
          <li><b>日曆</b> —— 14 天 × 8 口岸一覽，一眼挑出最順的日子</li>
          <li><b>對比</b> —— 同一天各口岸放一起，直接告訴你當日最佳選擇</li>
          <li><b>排隊</b> —— 選口岸和日期，模擬排隊：超半小時的概率、一般要排多久</li>
          <li><b>景點</b> —— 人多的日子，看看口碑好又相對清閒的去處</li>
        </ul>
        <h3>技術說明</h3>
        <p class="flowline">
          入境處日客流 → <b>LOF 異常清洗</b>（先剔除颱風 / 暴雨等異常日）→ <b>GBT 預測</b>（直方圖梯度提升樹，
          學習節假日 / 星期規律，遞歸 14 天）→ <b>馬爾可夫 4 檔擁擠狀態</b>（按歷史流轉規律給出獨立推算）→
          <b>排隊模擬</b>（Erlang-C 排隊論 + 蒙特卡洛）→ 推薦（哪天 / 哪個口岸 / 哪個方向 / 幾點）<br>
          旁路：<b>TF-IDF + 邏輯回歸情感分析</b> → 錯峰景點推薦
        </p>
        <p class="hint">全部模型為自寫實現（NumPy 為主，scikit-learn 僅作對照基準）：主預測模型經滾動回測，
        誤差比「按星期幾估」的基線低約 31%，亦優於 sklearn 對照實現；評測細節見課程報告。
        逐小時曲線為口岸類型模板 + 排隊模型估算（入境處不公佈分時數據）。
        排隊模擬主要假設：最忙 1 小時約佔全天 12% 客流 · 每櫃檯平均 30 秒辦理一位 ·
        櫃檯數按該口岸歷史高峰日負荷估出 · 嚴重超載時按大廳滯留 20–120 分鐘估算。</p>
        <h3>數據來源</h3>
        <ul>
          <li><a href="https://data.gov.hk/en-data/dataset/hk-immd-set5-statistics-daily-passenger-traffic">入境處 · 每日各管制站旅客人次</a>（核心時序，2021-01 至今）</li>
          <li><a href="https://data.gov.hk/en-data/dataset/hk-customs-ced_stat-passenger-clearance">海關 · 管制站旅客吞吐量</a>（年度容量參考）</li>
          <li>TripAdvisor 景點評論（好評率模型訓練用，HuggingFace 鏡像）</li>
        </ul>
        <h3>聲明</h3>
        <p>預測僅基於歷史統計規律，不構成出行保證；口岸臨時管制、天氣及突發事件不在模型內。
        景點頁的香港評論為<b>演示用示例</b>（好評率判斷模型本身僅用真實評論訓練）。
        MATH7005 課程項目數據產品。</p>
      </div>`,
  en: `<div class="card">
        <h3 style="margin-top:2px">What this is</h3>
        <p>Weekend trip north, homecoming over Golden Week, long-holiday return — which day is packed,
        which crossing is smooth, what hour skips the long queue? BorderPulse uses the Hong Kong
        Immigration Department's daily published crossing counts to estimate congestion at
        <b>8 land / cross-boundary control points 14 days ahead</b>, helping you decide: change the day?
        the crossing? the direction? Or leave as planned and pick a smooth hour?</p>
        <h3>How to use</h3>
        <ul>
          <li><b>Calendar</b> — 14 days × 8 crossings at a glance; spot the smoothest days instantly</li>
          <li><b>Compare</b> — every crossing side by side for one day; the best pick is called out</li>
          <li><b>Queue</b> — pick a crossing and date to simulate the queue: chance of waiting 30+ minutes, typical wait</li>
          <li><b>Spots</b> — on crowded days, well-rated and relatively quiet places to go</li>
        </ul>
        <h3>Technical notes</h3>
        <p class="flowline">
          ImmD daily flow → <b>LOF anomaly cleaning</b> (drops typhoon / rainstorm outliers first) →
          <b>GBT forecast</b> (histogram gradient-boosted trees learning holiday / weekday patterns,
          recursive 14 days) → <b>Markov 4-level congestion states</b> (an independent estimate from
          historical state transitions) → <b>queue simulation</b> (Erlang-C queueing + Monte Carlo) →
          recommendation (day / crossing / direction / hour)<br>
          Side branch: <b>TF-IDF + logistic-regression sentiment</b> → off-peak attraction picks
        </p>
        <p class="hint">Every model is a from-scratch implementation (primarily NumPy; scikit-learn is used
        only as a benchmark): in rolling backtests the main forecast model's error is about 31% below the
        day-of-week baseline and also beats the sklearn reference; evaluation details are in the course report.
        Hourly curves come from crossing-type templates plus the queue model (ImmD publishes no hourly data).
        Main queue-simulation assumptions: the busiest hour carries ~12% of daily flow · ~30 seconds per
        traveller per counter · counter counts are calibrated to each crossing's historical peak-day load ·
        under heavy overload, hall dwell of 20–120 minutes is assumed.</p>
        <h3>Data sources</h3>
        <ul>
          <li><a href="https://data.gov.hk/en-data/dataset/hk-immd-set5-statistics-daily-passenger-traffic">ImmD · Daily passenger traffic at control points</a> (core time series, 2021-01 to present)</li>
          <li><a href="https://data.gov.hk/en-data/dataset/hk-customs-ced_stat-passenger-clearance">C&amp;ED · Control point passenger throughput</a> (annual capacity reference)</li>
          <li>TripAdvisor attraction reviews (training data for the positive-review model, HuggingFace mirror)</li>
        </ul>
        <h3>Disclaimer</h3>
        <p>Forecasts are based on historical patterns only and are no travel guarantee; ad-hoc control
        measures, weather and incidents are outside the models. Hong Kong reviews on the Spots tab are
        <b>demo examples</b> (the positive-review classifier itself is trained on real reviews only).
        Data product for the MATH7005 course.</p>
      </div>`,
};

/* ---------------- state & helpers ---------------- */
/* 中文：全局状态 S —— d=数据(data.json)、dir=方向、tab=当前页签、
   dateIdx=选中天数组下标、cp=选中口岸、mc=最近一次模拟结果。
   LV = 4 档拥挤等级（级别/样式类/符号/三语名）。 */
const S = { d: null, dir: "Departure", tab: "cal", dateIdx: 0, cp: "LoWu", mc: null };

const LV = [
  { n: 1, cls: "lv1", glyph: "✓", w: ["畅顺", "暢順", "Smooth"] },
  { n: 2, cls: "lv2", glyph: "●", w: ["正常", "正常", "Normal"] },
  { n: 3, cls: "lv3", glyph: "▲", w: ["繁忙", "繁忙", "Busy"] },
  { n: 4, cls: "lv4", glyph: "◆", w: ["爆满", "爆滿", "Packed"] },
];
const sw = (n) => LV[n - 1].w[LI()];
const DOWC = ["一", "二", "三", "四", "五", "六", "日"];
const DOWE = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
const dowC = (i) => (LANG === "en" ? DOWE[i] : DOWC[i]);

const $ = (s, el) => (el || document).querySelector(s);
const key = (cp, dir) => cp + "|" + dir;
const fmtN = (n) => Math.round(n).toLocaleString("en-US");
const fmtK = (n) =>
  n >= 10000
    ? (LANG === "en" ? Math.round(n / 1000) + "k" : (n / 10000).toFixed(1) + (LANG === "tc" ? "萬" : "万"))
    : fmtN(n);
const pct = (p) => Math.round(p * 100) + "%";
const lv = (n) => LV[n - 1];

function cpOrder(dir) {
  // fixed row order by typical flow (P50 of the direction), largest first
  // 中文：口岸行的固定排序——按该方向的中位客流（P50）从大到小，
  // 大口岸永远在上、位置不随页签变化（减少视觉跳动）。
  return Object.values(S.d.stats).filter((s) => s.dir === dir)
    .sort((a, b) => b.q["50"] - a.q["50"]).map((s) => s.short);
}

/* ---------------- language switch ---------------- */
/* 中文：切换语言。写 localStorage → 更新 <html lang>（无障碍/字体）→
   重渲标题与全部界面。EN 模式下品牌副标题隐藏（品牌词太长）。 */
function setLang(l) {
  if (l !== "sc" && l !== "tc" && l !== "en") return;
  LANG = l;
  try { localStorage.setItem("bp-lang", l); } catch (e) {}
  document.documentElement.lang = l === "tc" ? "zh-HK" : l === "en" ? "en" : "zh-CN";
  document.title = t("title");
  renderChrome();
  closeSheet();
  renderCal();
  setTab(S.tab);
}

/* static chrome: app bar brand, meta line, tab bar, about tab, lang buttons */
function renderChrome() {
  const brand = $("#appbar .t span");
  brand.textContent = t("brand");
  brand.style.display = LANG === "en" ? "none" : "";
  $("#meta").textContent = S.d ? t("meta", S.d.data_through) : t("loading");
  document.querySelectorAll("#tabbar button").forEach((b) => {
    const svg = b.querySelector("svg").outerHTML;
    b.textContent = "";
    b.insertAdjacentHTML("beforeend", svg + t("tab", b.dataset.t));
    b.setAttribute("aria-label", t("tabAria", b.dataset.t));
  });
  const lg = $("#lang");
  lg.setAttribute("aria-label", t("langAria"));
  lg.querySelectorAll("button").forEach((b) => {
    const on = b.dataset.l === LANG;
    b.classList.toggle("on", on);
    b.setAttribute("aria-pressed", on);
  });
  $("#v-about").innerHTML = ABOUT[LANG];
}

/* ---------------- tabs ---------------- */
function setTab(tb) {
  S.tab = tb;
  document.querySelectorAll("main > section").forEach((el) => (el.style.display = "none"));
  $("#v-" + tb).style.display = "";
  document.querySelectorAll("#tabbar button").forEach((b) => b.classList.toggle("on", b.dataset.t === tb));
  if (tb === "cmp") renderCmp();
  if (tb === "risk") renderRisk();
  if (tb === "poi") renderPoi();
}

/* ---------------- shared bits ---------------- */
function segDir(id) {
  return `<div class="seg" id="${id}">
    <button data-d="Departure" class="${S.dir === "Departure" ? "on" : ""}">${t("dirN")}</button>
    <button data-d="Arrival" class="${S.dir === "Arrival" ? "on" : ""}">${t("dirS")}</button>
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
  const l = lv(n), w = sw(n);
  return `<span class="chip ${l.cls}"><span class="g">${l.glyph}</span>${LANG === "en" ? w : `${w} ${l.w[2]}`}</span>`;
}
function legend() {
  return `<div class="legend">${LV.map((l) =>
    `<span class="chip ${l.cls}"><span class="g">${l.glyph}</span>${l.w[LI()]}</span>`).join("")}</div>`;
}

/* local calendar date as ISO yyyy-mm-dd (forecast rows carry the same format) */
function todayISO() {
  const d = new Date();
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
}

/* ---------------- calendar heatmap ---------------- */
/* 中文：日历热力图。一次渲染一周（calBlock），周与周叠成块。
   每格是 <button>（键盘可达），aria-label 带口岸/日期/档位；
   周末格加 we 样式、今天格加粗边框（today）。 */
function calBlock(cps, start, count) {
  const day = (i) => S.d.forecast[key(cps[0], S.dir)][i];
  const range = `${day(start).date.slice(5).replace("-", "/")} – ${day(start + count - 1).date.slice(5).replace("-", "/")}`;
  /* pad a short final week with empty slots so the 7-column grid keeps its
     geometry (like a month ending mid-week) — without them the trailing
     empty tracks mis-size and the cells blow up */
  /* 中文：末周不足 7 天时补空白占位格，保持 7 列网格几何不变形
     （同月历月末留空的惯例）；不补的话尾部空轨道尺寸异常、格子会被撑大 */
  const pad = count < 7 ? `<div class="gh" aria-hidden="true"></div>`.repeat(7 - count) : "";
  let h = `<div class="calblock"><div class="caltitle">${range}</div><div class="cal">`;
  h += `<div class="cp"></div>`;
  for (let i = start; i < start + count; i++) {
    const r = day(i);
    const isToday = r.date === todayISO();
    h += `<div class="hd ${r.dow >= 5 ? "we" : ""}${isToday ? " today" : ""}">${dowC(r.dow)}<br>${r.date.slice(5).replace("-", "/")}</div>`;
  }
  h += pad;
  cps.forEach((cp) => {
    const st = S.d.stats[key(cp, S.dir)];
    h += `<div class="cp"><b>${st.short}</b>${cpn(cp)}</div>`;
    for (let i = start; i < start + count; i++) {
      const r = S.d.forecast[key(cp, S.dir)][i];
      const l = lv(r.level);
      const g = r.level >= 3 ? l.glyph : "";
      const isToday = r.date === todayISO();
      h += `<button class="cell g${r.level} ${r.dow >= 5 ? "we" : ""}${isToday ? " today" : ""}" data-cp="${cp}" data-i="${i}"
        aria-label="${cpn(cp)} ${r.date} ${sw(r.level)}">${g}</button>`;
    }
    h += pad;
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
    + `<p class="hint">${t("calHint")}</p>`;
  bindSeg("seg-cal");
}

/* Markov 4-state distribution row (r.mk = [p1..p4], from the state chain
   stepped from the last actual day to this date) */
/* 中文：马尔可夫档位分布条——详情弹层里的四段彩色概率条。
   r.mk = [p1..p4]（自最近实测日起推到该天），宽度即概率，
   <0.4% 的段不渲染。 */
function mkBlock(r) {
  if (!r.mk || !r.mk.length) return "";
  const segs = r.mk.map((p, i) => (p > 0.004 ? `<i class="g${i + 1}" style="width:${(p * 100).toFixed(1)}%"></i>` : "")).join("");
  const lg = r.mk.map((p, i) => `<span><i class="g${i + 1}"></i>${LV[i].w[LI()]} ${pct(p)}</span>`).join("");
  return `<div class="kv"><span>${t("fromHist")}</span></div>
    <div class="mkbar" role="img" aria-label="${t("fromHistAria")}">${segs}</div>
    <div class="mklg">${lg}</div>`;
}

function openCellSheet(cp, i) {
  const st = S.d.stats[key(cp, S.dir)];
  const r = S.d.forecast[key(cp, S.dir)][i];
  const isToday = r.date === todayISO();
  const sp = S.d.spark[key(cp, S.dir)];
  const ax = sp && sp.flow.length ? sp.flow.length - 1 : -1;   // latest ImmD actual
  const name = LANG === "en" ? st.cp : `${cpn(cp)} ${st.cp}`;
  openSheet(`<h3>${name}</h3>
    <div class="sub">${r.date}（${t("dowFull", r.dow)}）${isToday ? " · " + t("today") : ""} · ${t(S.dir === "Departure" ? "dirN" : "dirS")} · ${r.festive ? t("festive", r.festive) : t("nonFestive")}</div>
    ${chip(r.level)}
    <div class="kv"><span>${t("estCrossings")}</span><b class="num">${fmtN(r.base)}</b></div>
    ${ax >= 0 ? `<div class="kv"><span>${t("latestActual", sp.dates[ax])}</span><b class="num">${fmtN(sp.flow[ax])}</b></div>` : ""}
    <div class="kv"><span>${t("range80")}</span><b class="num">${fmtK(r.lo)} – ${fmtK(r.hi)}</b></div>
    <div class="kv"><span>${t("busyP")}</span><b class="num">${pct(r.p_busy)}</b></div>
    ${mkBlock(r)}
    <div class="kv"><span>${t("cpRef")}</span><b class="num small">${fmtK(st.q["50"])} / ${fmtK(st.q["85"])} / ${fmtK(st.q["95"])}</b></div>
    ${hourlyBlock(st, r)}`);
}

/* ---------------- intraday (hour-of-day) model ---------------- */
/* 中文：日内（逐小时）模型。ImmD 不公布分时数据，这里用"口岸类型
   模板曲线 × 每小时 Erlang-C 排队"估算：
   小时客流 = 当日预测 × 模板权重；每小时算 P(排队>30 分钟) 上色。
   模板/开放时间来自 data.json（build_docs.py 的 TEMPLATES/OPEN）。 */
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
  /* 中文：取出该(口岸类型, 方向, 日类型)的模板曲线，裁剪到该口岸
     实际开放的小时段（闭馆小时权重清零），再归一化到和为 1。 */
  const tpl = S.d.intraday.templates[st.profile][st.dir][dayType(r)];
  const [o, c] = S.d.intraday.open[st.short];
  const lo = Math.floor(o), hi = Math.ceil(c) - 1;
  let s = 0;
  const w = tpl.map((v, h) => (h >= lo && h <= hi ? (s += v, v) : 0));
  return w.map((v) => v / s);
}

function hourlyBlock(st, r) {
  /* 中文：详情弹层里的"几点出发最顺"块。逐小时算排队风险后：
     - 找峰值小时（λ 最大）→ "高峰"提示行
     - 找 8-21 点里风险最低的连续 2 小时 → 高亮为最佳时段
       （看的是今天 → 只考虑当前时刻之后的小时）
     渲染为 24 根彩色竖条 + "现在"指示线。柜台数标定与排队页一致。 */
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
    ? ` · <b>${t("nowTag", String(now.getHours()).padStart(2, "0") + ":" + String(now.getMinutes()).padStart(2, "0"))}</b>` : "";
  const pkState = !isToday ? "" : nowH >= pk.h + 1 ? t("pkPast") : nowH >= pk.h ? t("pkNow") : "";

  return `<div class="hr">${t("bestHours", fmtOpen(o), fmtOpen(cH))}${nowTag}</div>
    <div class="hwrap"><div class="hours" role="img" aria-label="${t("hoursAria")}">${bars}</div>${nowLine}</div>
    <div class="haxis">${axis}</div>
    <div class="advice" style="margin-top:8px">
      ${t("peak", fmtH(pk.h), fmtH(pk.h + 1), pkState, fmtK(pk.lam * 60), pct(pk.p30))}<br>
      ${best ? t(ahead ? "adviceAhead" : "adviceBest", fmtH(best.h), fmtH(best.h + 2)) : ""}
    </div>
    <p class="hint">${t("hourlyHint")}</p>`;
}

/* deterministic per (cp, date, hour): the same day always renders the same
   numbers, so reopening a sheet never contradicts the previous look */
/* 中文：确定性随机源——种子由 (口岸, 日期, 小时) 哈希得到。
   好处：同一天不管打开多少次，逐小时数字完全一致，不会"上次看 37%
   这次看 41%"自相矛盾。mulberry32 是轻量 PRNG。 */
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
  /* 中文：某小时的 P(排队>30 分钟)。200 次确定性抽样：
     每次抽一个对数正态到达率 → 未饱和用 Erlang-C 尾概率解析式
     exp(-(c·MU-λ)·30)，饱和（ρ≥0.97）按三角滞留取 0.96。
     与排队页同一套常数，只是解析式替代了直方图模拟。 */
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
/* 中文：对比页。选定一天 → 8 个口岸按"最顺"排序成卡片，每张带
   三块数字瓦片（预计人次/拥挤概率/相对最佳）、风险进度条和
   近 60 天 spark 折线；当日最佳卡片加高亮边框。 */
function renderCmp() {
  const el = $("#v-cmp");
  const cps = cpOrder(S.dir);
  const days = S.d.forecast[key(cps[0], S.dir)];
  if (S.dateIdx >= days.length) S.dateIdx = 0;
  let chips = `<div class="dates" id="dt">`;
  days.forEach((r, i) => {
    chips += `<button data-i="${i}" class="${i === S.dateIdx ? "on" : ""} ${r.dow >= 5 ? "we" : ""}">
      ${dowC(r.dow)} ${r.date.slice(5).replace("-", "/")}<small>${r.festive ? r.festive : ""}</small></button>`;
  });
  chips += `</div>`;

  const rows = cps.map((cp) => ({ cp, st: S.d.stats[key(cp, S.dir)], r: S.d.forecast[key(cp, S.dir)][S.dateIdx] }));
  const best = rows.slice().sort((a, b) => (a.r.level - b.r.level) || (a.r.p_busy - b.r.p_busy))[0];
  rows.forEach((x) => (x.diff = (x.r.base - best.r.base) / Math.max(best.r.base, 1)));
  /* small denominators (STK ~1k/day) blow up percentages — switch to a multiple */
  const relTxt = (d) => (d > 2 ? "×" + Math.round(1 + d) : "+" + Math.round(d * 100) + "%");

  let h = `<div class="row">${segDir("seg-cmp")}</div>${chips}`;
  rows.sort((a, b) => (a.r.level - b.r.level) || (a.r.p_busy - b.r.p_busy)); /* 最顺的排最前 */
  rows.forEach((x) => {
    const sp = spark(key(x.cp, S.dir));
    const isBest = x.cp === best.cp;
    const bestName = LANG === "en" ? CPN[best.cp][2] : cpn(best.cp);
    h += `<div class="card ${isBest ? "pick" : ""}">
      <div class="cmp-h"><div class="nm">${x.st.short} <span>${LANG === "en" ? x.st.cp : cpn(x.cp) + " · " + x.st.cp}</span></div>${chip(x.r.level)}</div>
      <div class="stats">
        <div><div class="v num">${fmtK(x.r.base)}</div><div class="k">${t("estN")}</div></div>
        <div><div class="v num">${pct(x.r.p_busy)}</div><div class="k">${t("busyP")}</div></div>
        <div><div class="v num">${isBest ? "—" : relTxt(x.diff)}</div><div class="k">${t("vsBest")}</div></div>
      </div>
      <div class="meter"><i style="width:${Math.min(100, x.r.p_busy * 100)}%"></i></div>
      ${sp}
      <div class="advice">${isBest
        ? t("pickTxt", fmtN(x.r.base), pct(x.r.p_busy))
        : t("worseTxt", fmtN(x.r.base - best.r.base), bestName, relTxt(x.diff))}</div>
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
  /* 中文：近 60 天客流 spark 迷你折线（SVG，末点加圆点标记最新值）。 */
  const s = S.d.spark[k];
  if (!s) return "";
  const w = 440, hgt = 44, pad = 3;
  const mx = Math.max(...s.flow), mn = Math.min(...s.flow);
  const pts = s.flow.map((v, i) => [
    pad + (i / (s.flow.length - 1)) * (w - 2 * pad),
    hgt - pad - ((v - mn) / Math.max(mx - mn, 1)) * (hgt - 2 * pad),
  ]);
  const last = pts[pts.length - 1];
  return `<svg class="spark" viewBox="0 0 ${w} ${hgt}" preserveAspectRatio="none" aria-label="${t("sparkAria")}">
    <polyline points="${pts.map((p) => p[0].toFixed(1) + "," + p[1].toFixed(1)).join(" ")}"
      fill="none" stroke="var(--accent)" stroke-width="2" vector-effect="non-scaling-stroke"/>
    <circle cx="${last[0]}" cy="${last[1]}" r="3.5" fill="var(--accent)"/>
  </svg>`;
}

/* ---------------- risk (Monte Carlo queue) ---------------- */
/* 中文：排队页——浏览器里跑的蒙特卡洛排队模拟（与后端 mc_queue.py
   同一套模型和常数）。流程：抽 3000 个"可能的当天客流"（对数正态）
   → 最忙小时到达率 λ → 未饱和用 Erlang-C 稳态均值，饱和用三角
   滞留封顶 → 出直方图 + 累计曲线 + 三块数字瓦片。 */
function erlangC(c, rho) {
  /* P(all servers busy), M/M/c; rho = offered erlangs / c. Iterative, O(c). */
  /* 中文：Erlang-C——M/M/c 系统里"所有柜台都忙"的概率。
     递推算泊松项之和，避免大数阶乘。与 src/models/mc_queue.py 相同。 */
  const a = rho * c;
  let sum = 1, term = 1;
  for (let i = 1; i < c; i++) { term *= a / i; sum += term; }
  const last = term * a / c;
  return last / ((1 - rho) * sum + last);
}

function runMC() {
  /* 中文：跑一次模拟（点"模拟这一天的排队"触发）。逐次抽样：
     Box-Muller 变换抽标准正态 → 日客流 = 预测 × exp(σz)（对数正态）
     → λ = 日客流 × 12% / 60（每分钟）→ ρ = λ/(c·MU) 分流：
     ρ<0.97 走 Erlang-C 稳态平均等待；否则三角分布(20/45/120)封顶。
     结果排序后存 S.mc（renderRiskResult 消费）。 */
  const k = key(S.cp, S.dir);
  const st = S.d.stats[k], f = S.d.forecast[k][S.dateIdx];
  const MU = 2;                    // service rate per counter, 1/min (30 s mean)
  const PEAK_SHARE = 0.12;         // share of daily flow in the busiest hour
  const PEAK_MIN = 60, TRIALS = 3000, CAP = 120;

  /* counters calibrated so the P90 historical day runs at 95% utilization */
  /* 中文：柜台数标定——让历史 P90 日（取 P85/P95 中点）恰好 95% 利用率，
     至少 4 个。q90 用两个分位数的中点而非真 P90，避免前端重算百分位。 */
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
  /* 中文：渲染模拟结果——三块瓦片（超 30 分钟概率/典型等待/P95）、
     0-120 分钟 24 桶直方图 + 累计"排完"曲线（CDF）、30 分钟参考线。 */
  const { wq } = S.mc;
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
     <text x="${x0(m)}" y="${hgt - pb + 14}" font-size="9" fill="var(--muted)" text-anchor="middle">${m}</text>`).join("");
  const cdfAt30 = wq.filter((x) => x <= 30).length / wq.length;
  const ylab = hgt - pb - cdfAt30 * (hgt - pt - pb);

  $("#mc-out").innerHTML = `
    <div class="tiles">
      <div class="tile"><div class="v num">${pct(p30)}</div><div class="k">${t("tileP30")}</div></div>
      <div class="tile"><div class="v num">${q(0.5).toFixed(1)}<span class="small"> min</span></div><div class="k">${t("tileTyp")}</div></div>
      <div class="tile"><div class="v num">${q(0.95).toFixed(0)}<span class="small"> min</span></div><div class="k">${t("tileP95")}</div></div>
    </div>
    <div class="card">
      <svg id="riskplot" viewBox="0 0 ${w} ${hgt}" role="img" aria-label="${t("riskAria")}">
        ${ticks}${bars}${cdf}
        <line x1="${mx30}" y1="${pt}" x2="${mx30}" y2="${hgt - pb}" stroke="var(--critical)"
          stroke-width="1.5" stroke-dasharray="5 4"/>
        <circle cx="${mx30}" cy="${ylab}" r="4" fill="var(--critical)"/>
        <text x="${Math.min(mx30 + 6, w - 90)}" y="${Math.max(pt + 10, ylab - 8)}" font-size="10"
          fill="var(--ink)" font-weight="700">${t("cleared30", pct(cdfAt30))}</text>
      </svg>
      <div class="legend small" style="margin-top:8px">
        <span class="small">${t("lgDist")}</span><span class="small">${t("lgCdf")}</span>
      </div>
    </div>
    <p class="hint">${t("mcHint")}</p>`;
}

function renderRisk() {
  /* 中文：排队页框架——口岸/日期两个下拉 + 估算提示行 + 运行按钮。
     改口岸或日期会清掉上次模拟（S.mc=null）逼用户重跑（参数变了）。 */
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
          return `<option value="${cp}" ${cp === S.cp ? "selected" : ""}>${LANG === "en" ? s.cp : cpn(cp) + " " + s.short}</option>`;
        }).join("")}</select>
        <select id="sel-dt">${days.map((r, i) =>
          `<option value="${i}" ${i === S.dateIdx ? "selected" : ""}>${r.date}（${t("dowFull", r.dow)}${r.festive ? " · " + r.festive : ""}）</option>`).join("")}</select>
      </div>
      <div class="kv small" style="display:flex;justify-content:space-between;padding:6px 0;border-top:1px solid var(--grid)">
        <span class="hint">${t("estDay", fmtN(S.d.forecast[key(S.cp, S.dir)][S.dateIdx].base))}</span>
        <span class="hint">${t("histPeak", fmtN(st.q["95"]))}</span></div>
      <button class="runbtn" id="run">${t("runBtn")}</button>
    </div>
    <div id="mc-out">${S.mc ? "" : `<p class="hint">${t("mcPlaceholder")}</p>`}</div>`;
  el.innerHTML = h;
  bindSeg("seg-rk");
  $("#sel-cp").addEventListener("change", (e) => { S.cp = e.target.value; S.mc = null; renderRisk(); });
  $("#sel-dt").addEventListener("change", (e) => { S.dateIdx = +e.target.value; S.mc = null; renderRisk(); });
  $("#run").addEventListener("click", () => {
    const b = $("#run"); b.disabled = true; b.textContent = t("running");
    setTimeout(() => { runMC(); const rb = $("#run"); if (rb) { rb.disabled = false; rb.textContent = t("rerun"); } }, 30);
  });
  if (S.mc) renderRiskResult();
}

/* ---------------- POI: sentiment-ranked off-peak attractions ---------------- */
/* 中文：景点页——按好评率排序的 12 个景点卡片（好评率大数字 + 进度条 +
   指示关键词标签 + 错峰贴士）。NOT_ 前缀显示为 "not·"（否定标记）。
   数据来自 data.json 里合并的 nlp 段。 */
function renderPoi() {
  const el = $("#v-poi");
  const nlp = S.d.nlp;
  if (!nlp || !nlp.attractions) {
    el.innerHTML = `<p class="hint">${t("poiMissing")}</p>`;
    return;
  }
  const pretty = (x) => (x.startsWith("NOT_") ? "not·" + x.slice(4) : x);
  let h = `<p class="hint" style="margin:2px 0 10px">${t("poiIntro", fmtN(nlp.train_reviews), nlp.attractions.length)}</p>`;
  nlp.attractions.forEach((a, i) => {
    h += `<div class="card${i === 0 ? " pick" : ""}">
      <div class="cmp-h"><div class="nm">${poiName(a)} <span>${LANG === "en" ? (POI_TC[a.en] || a.cn) : a.en}</span></div>
        <div class="v num" style="font-size:18px;font-weight:700">${pct(a.pos)}</div></div>
      <div class="meter"><i style="width:${Math.round(a.pos * 100)}%"></i></div>
      <div class="terms">${a.terms.map((x) => `<span>${pretty(x)}</span>`).join("")}</div>
      <div class="advice">💡 ${poiTip(a)}</div>
    </div>`;
  });
  h += `<p class="hint">${t("poiFoot")}</p>`;
  el.innerHTML = h;
}

/* ---------------- sheet ---------------- */
/* 中文：底部弹层（详情 sheet）。openSheet 填充内容并加 show 类滑入；
   点背景或 ✕ 关闭。 */
function openSheet(html) {
  $("#sheet").innerHTML = `<i class="grab" aria-hidden="true"></i><button class="close" aria-label="${t("closeAria")}">✕</button>` + html;
  $("#sheet-bg").classList.add("show");
  $("#sheet").classList.add("show");
}
function closeSheet() {
  $("#sheet-bg").classList.remove("show");
  $("#sheet").classList.remove("show");
}

/* ---------------- boot ---------------- */
/* 中文：启动流程。设 <html lang> → fetch data.json（唯一网络请求）→
   默认选中第一个口岸和最近的周六（周末最有决策场景）→ 首渲染 →
   挂一次性事件委托（后续渲染只替换 innerHTML，监听不重复绑）。
   Service Worker：HTTPS/localhost 下注册；新版本接管时弹一键刷新
   提示（首次安装静默，不打扰访客）。 */
async function boot() {
  document.documentElement.lang = LANG === "tc" ? "zh-HK" : LANG === "en" ? "en" : "zh-CN";
  document.title = t("title");
  const r = await fetch("./data.json");
  S.d = await r.json();
  /* anchor the forecast window to TODAY: when the device date falls inside
     the 14-day window, drop the already-past days so every view (calendar,
     compare chips, queue sim) opens on today; if the data is stale (today
     lies beyond the window) keep the original day-1 start */
  /* 中文：把预测窗口锚定到“今天”——设备日期落在 14 天窗口内时，
     丢弃已过去的日期，日历第一列永远是今天（对比页/排队页同步从今天起）；
     数据过期（今天已超出窗口）则保持原样从第 1 天展示 */
  {
    const rows = S.d.forecast[key(cpOrder(S.dir)[0], S.dir)];
    const t0 = rows.findIndex((x) => x.date === todayISO());
    if (t0 > 0) for (const k of Object.keys(S.d.forecast)) S.d.forecast[k] = S.d.forecast[k].slice(t0);
  }
  const cps = cpOrder(S.dir);
  S.cp = cps[0];
  const fs = S.d.forecast[key(S.cp, S.dir)];
  const sat = fs.findIndex((x) => x.dow === 5);
  S.dateIdx = sat >= 0 ? sat : 0;
  renderChrome();
  renderCal();
  /* one-time delegated listeners (renders replace innerHTML only) */
  $("#v-cal").addEventListener("click", (e) => {
    const c = e.target.closest(".cell");
    if (c) openCellSheet(c.dataset.cp, +c.dataset.i);
  });
  $("#tabbar").addEventListener("click", (e) => {
    const b = e.target.closest("button");
    if (b) setTab(b.dataset.t);
  });
  $("#lang").addEventListener("click", (e) => {
    const b = e.target.closest("button");
    if (b) setLang(b.dataset.l);
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
      const el = document.createElement("div");
      el.id = "toast";
      el.innerHTML = `${t("newVer", S.d.version)}
        <button onclick="location.reload()">${t("refresh")}</button>
        <button class="x" aria-label="${t("closeAria")}" style="background:none;padding:6px;color:inherit;font-weight:400">✕</button>`;
      document.body.appendChild(el);
      el.querySelector("button.x").onclick = () => el.remove();
      setTimeout(() => el.remove(), 12000);
      requestAnimationFrame(() => el.classList.add("show"));
    });
  }
}
boot();
