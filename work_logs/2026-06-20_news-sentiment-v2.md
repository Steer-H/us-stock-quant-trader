# 新闻/财报情感特征集成 v2

**日期**: 2026-06-20
**严重程度**: 🟢 功能新增 + 🟡 改进

## 修改文件

| 文件 | 操作 | 说明 |
|------|------|------|
| `crawler/news_sentiment.py` | ✏️ 修复 | 财报数据时区归一化 (tz-aware→naive) |
| `crawler/news_scraper.py` | 🆕 新建 | RSS多源新闻抓取器 (Yahoo + Google News) |
| `scripts/build_sentiment_features.py` | 🆕 新建 | 批量构建情感特征脚本 |
| `scripts/update_daily_sentiment.py` | 🆕 新建 | 每日新闻情感更新脚本 |
| `config/settings.py` | ✏️ 新增 | sentiment_features列表, 清理重复字段 |
| `ml_model/transformer.py` | ✏️ 改进 | 特征分组加权 (情感×0.05) |

## 实现内容

### 1. 财报数据集成 ✅
- 通过 yfinance.Ticker.earnings_dates 抓取 EPS 惊喜
- **修复时区bug**: yfinance返回tz-aware索引，归一化为naive datetime
- 40只股票，882条财报记录 (22季度/股)
- 填充 `earnings_surprise_pct`、`has_earnings_report`

### 2. 新闻情感代理 ✅
- yfinance stock.news 仅返回10条最近新闻，与历史数据无交集
- 使用财报惊喜值作为新闻情感代理信号
- 前向填充3-7天传播情感影响
- 填充 `news_sentiment_3d`、`news_sentiment_7d`

### 3. 真实新闻抓取器 ✅ (crawler/news_scraper.py)
- 多源RSS: Yahoo Finance RSS + Google News RSS
- 关键词情感分析 (沿用 news_sentiment.py 词典)
- 测试结果: NVDA 58条/7天, AAPL 58条/30天, MSFT 85条/30天
- **限制**: RSS仅提供近期新闻，与历史训练数据(截止2025-12)无交集
- **价值**: 用于每日实时更新，提升 live trading 预测质量

### 4. 模型特征加权 ✅
- StockTransformer 新增 `feature_weights` buffer
- 价格/技术特征权重=1.0，情感特征权重=0.05
- Forward中应用: `x * feature_weights` → 降低噪音特征影响

### 5. 每日更新脚本 ✅ (scripts/update_daily_sentiment.py)
- 从Google News RSS抓取最新新闻
- 更新最近14天parquet数据的情感列
- 建议cron每日运行: `PYTHONPATH=. python3 scripts/update_daily_sentiment.py`

## 特征统计

```
特征总数: 24 → 28 (+4)
  - news_sentiment_3d (权重0.05)
  - news_sentiment_7d (权重0.05)  
  - earnings_surprise_pct (权重0.05)
  - has_earnings_report (权重0.05)
```

## 验证结果

```
✅ 全项目 .py 文件通过 ast.parse()
✅ 模型前向传播测试通过 (28特征 → 权重应用正确)
✅ 服务运行正常: curl /api/status → Yahoo Finance (缓存 [v7])
✅ 训练进行中: screen train_weighted, 28特征
```

## 已知限制

1. **RSS新闻无法历史回填**: 仅能获取近期新闻，历史训练数据使用财报代理
2. **关键词情感分析精度有限**: 12/58条有非零情感信号，可考虑升级到FinBERT
3. **Google News RSS可能被限流**: 40只股票批量抓取需加延迟

## 后续优化建议

1. 训练完成后对比准确率变化 (24特征 vs 28特征)
2. 接入金融情感API (如FinBERT) 提升情感分析精度
3. 部署cron每日运行 update_daily_sentiment.py
4. 考虑添加新闻量、新闻多样性等特征
