# BorderPulse — Data Sources (all URLs verified 2026-11-09)

| # | Dataset | Role | data.gov.hk page | Direct URL | Granularity / coverage | Status |
|---|---------|------|------------------|------------|------------------------|--------|
| 1 | ImmD — Statistics on Daily Passenger Traffic | 核心（客流时序主体） | https://data.gov.hk/en-data/dataset/hk-immd-set5-statistics-daily-passenger-traffic | CSV: https://www.immd.gov.hk/opendata/eng/transport/immigration_clearance/statistics_on_daily_passenger_traffic.csv | 日；2021-01-01 → 昨天；口径：口岸 × 入/出 × 港人/内地客/其他访客；~63k 行 | ✅ 已下载 3.4 MB |
| 2 | ImmD — Passenger Traffic for Festive Periods | 核心（节日特征） | https://data.gov.hk/en-data/dataset/hk-immd-set5-statistics-passenger-traffic-festive-period | CSV: https://www.immd.gov.hk/opendata/eng/transport/immigration_clearance/statistics_passenger_traffic_festival_periods.csv | 节日区间（日） | ✅ HTTP 200 |
| 3 | C&ED — Control Points Passenger Throughput | 核心（容量约束） | https://data.gov.hk/en-data/dataset/hk-customs-ced_stat-passenger-clearance | CSV: https://www.customs.gov.hk/filemanager/common/csv/statistics/passenger_entry_exit_en.csv （⚠️ 301→200，需跟随重定向）<br>XLSX: https://www.customs.gov.hk/filemanager/common/xlsx/statistics/passenger_entry_exit_en.xlsx | ⚠️ **年度**，每口岸一行/年，单位 '000 人（2023–2025 等） | ✅ 200（-L）；粒度粗于预期 |
| 4 | HKO — Weather API | 辅助（特征/异常日） | 文档页: https://data.weather.gov.hk/en/weather/opendata.htm | 9 天预报 JSON: https://data.weather.gov.hk/weatherAPI/opendata/weather.php?dataType=fnd&lang=en<br>历史日值端点 Day-2 定稿 | 预报：9 天滚动；历史：日 | ✅ fnd 200 |
| 5 | Kaggle — TripAdvisor HK 景点评论 | 补充（错峰 NLP） | — | 搜索: https://www.kaggle.com/datasets?search=tripadvisor+hong+kong | 视具体数据集 | ⏳ 待选定 |
| 备 | C&ED — Land Boundary Vehicular Throughput | 备选 | https://data.gov.hk/en-data/dataset/hk-customs-ced_stat-vehicle-clearance | CSV: https://www.customs.gov.hk/filemanager/common/csv/statistics/passenger_land_en.csv | 年度（同上） | 未验证 |

## Day-1 发现

- **C&ED 只有年度粒度** → 排队模型的「容量」不能直接用它做日分母。替代方案：
  1. 从 ImmD 日数据自标定：各口岸历史日吞吐的高分位（如 P95/P99）作为「有效日容量」；
  2. XLSX 里可能有月度分表（Day-1 EDA 时检查 `passenger_entry_exit_en.xlsx`）；
  3. C&ED 年度值仅作 sanity anchor（自标定容量 × 365 ≈ 年度总量）。
