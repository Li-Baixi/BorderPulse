# BorderPulse 口岸脉搏

基于香港政府开放数据（[data.gov.hk](https://data.gov.hk/en/)）的**口岸错峰过关助手**：
预测 9 个陆路 / 跨界管制站未来 14 天的拥挤度，建议最佳过关日 / 口岸 / 方向，并用蒙特卡洛排队模拟量化
「排队超过 30 分钟」的风险。

**在线使用 / 手机加到主屏幕（PWA）**：<https://li-baixi.github.io/BorderPulse/>

## 功能

- **日历 Calendar** — 14 天 × 9 口岸拥挤度热力图（4 档分位带自标定），点按看预计客流与区间
- **对比 Compare** — 按日期比较各口岸：预计人次、繁忙概率、近 60 日走势、最佳口岸推荐
- **风险 Risk** — 浏览器内现场运行蒙特卡洛排队模拟（M/M/c，Erllang-C），输出 P(排队>30min)、
  中位 / P95 等待时间与分布图

## 数据

| 数据集 | 用途 |
|---|---|
| [ImmD 每日各管制站旅客人次](https://data.gov.hk/en-data/dataset/hk-immd-set5-statistics-daily-passenger-traffic) | 客流时序主体（2021-01 至今） |
| [C&ED 管制站旅客吞吐量](https://data.gov.hk/en-data/dataset/hk-customs-ced_stat-passenger-clearance) | 年度容量锚 |
| [HKO 开放数据](https://data.weather.gov.hk/en/weather/opendata.htm) | 天气特征 |

方法流水线：LOF 异常清洗 → GBT 主预测 → 马尔可夫 4 档拥挤状态 → 蒙特卡洛排队 → 推荐；
旁路 TF-IDF + 逻辑回归情感分析做错峰景点推荐。核心算法均为自写 NumPy / 原生 JS 实现。

## 目录结构

```
docs/            PWA 静态站（GitHub Pages 源，build_docs.py 产出 data.json）
src/build_docs.py 数据管线：清洗 → 分位标定 → 基线预测 → data.json
src/make_icons.py PWA 图标生成
data/raw/        原始 CSV（ImmD 日客流）
```

## 重建数据

```bash
python src/build_docs.py     # 重新下载节日日历逻辑 + 生成 docs/data.json
```

MATH7005 数据产品课程项目。
