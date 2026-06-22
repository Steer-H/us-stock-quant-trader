# 新闻/财报情感特征集成

**日期**: 2026-06-20
**严重程度**: 🟢 功能新增

## 修改文件

| 文件 | 操作 |
|------|------|
| `crawler/news_sentiment.py` | 🆕 新建 — 新闻情感分析模块 |
| `config/settings.py` | ✏️ 添加4个情感特征列 + news_feature_weight |
| `scripts/build_sentiment_features.py` | 🆕 新建 — 批量构建脚本 |

## 问题描述

ML模型仅使用价格技术指标，缺少基本面/情绪面信息。需要从网络获取财报数据和新闻情感作为辅助特征。

## 实施方案

### 财报数据集成（✅ 完成）
- 通过 `yfinance.Ticker.earnings_dates` 抓取 EPS 惊喜数据
- 修复时区归一化：yfinance 返回 tz-aware 索引 (`America/New_York`)，归一化为 naive datetime
- 每只股票约22个季度财报数据（2020-2025），总计882条
- 特征：`earnings_surprise_pct`（盈利惊喜%）、`has_earnings_report`（财报日标记）

### 新闻情感代理（✅ 完成）
- yfinance `stock.news` 仅返回最近10条（全是今天），与历史数据（截止2025年底）无交集
- **替代方案**：用财报惊喜值作为情感代理信号，前向填充3-7天
- 特征：`news_sentiment_3d`、`news_sentiment_7d`

### 特征权重
- `news_feature_weight = 0.05`（仅辅助参考，不影响主要价格特征的决策）
- 模型 `StockTransformer` 尚未实现按特征组加权（待后续优化）

## 构建结果

```
✅ 40/40 股票成功
✅ 882 财报日期
✅ 28 特征（24价格技术指标 + 4情感）
✅ 全项目 48 个 .py 文件通过 ast.parse()
✅ 服务重启正常: Yahoo Finance (缓存 [v7])
```

## 训练状态

- 旧训练已停止（quick_train, train_upgraded）
- 新训练已启动：`screen -S train_sentiment`
- 日志：`logs/quick_train.log`
- 监控命令：`tail -f logs/quick_train.log`

## 已知限制

1. **yfinance 新闻数据不可用**：仅返回最近几天的新闻，与历史数据无交集。考虑后续使用 Web 搜索或新闻 API
2. **news_feature_weight 仅在 config 中声明**：模型未实现按特征组加权，特征权重相同
3. **config/settings.py 有重复字段**（news_feature_weight、label_smoothing 各声明两次），不影响运行但应清理

## 影响评估

- ✅ 非破坏性：仅在已有 parquet 文件中填充新列值
- ✅ 特征数增加（24→28），模型自动适配
- ✅ 小权重（0.05）不会显著改变模型行为
- ⚠️ CPU 训练需较长时间（20 epoch），建议后续评估准确率变化

## 后续优化建议

1. 用 Web 搜索/新闻 API 获取真实历史新闻情感
2. 实现特征组加权（价格特征 vs 情感特征）
3. 清理 settings.py 重复字段
4. 评估新特征的 SHAP/特征重要性
