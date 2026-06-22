# 代码结构 — 每个文件的职责

## live_trading/ — 核心交易系统

### `web_server.py` (1326行) ⭐ 主控模块
**职责**: Flask 服务器 + 交易引擎 + 全局状态管理
- `tick_engine()` — 每秒执行的交易循环
- `fetch_yahoo_prices()` — Yahoo v7 API 批量获取价格
- `fetch_kline_data()` — 获取K线数据
- `init_positions()` — 开盘建仓
- `build_status_data()` — 构建 `/api/status` 响应
- `engine_loop()` — 后台引擎主循环
- `init_ml_model()` — 异步加载 ML 模型
- `_collect_globals_dict()` — 收集全局状态供持久化
- `start_server()` — 启动入口
- **8个 REST 端点**: health, tickers, signals, kline, kline/multi, status, benchmark_curve, backtest_summary

### `portfolio.py` (581行) ⭐ 持仓管理
**核心类**: `PortfolioManager`, `HoldingPosition`, `TradeRecord`
- `execute_buy/sell/short()` — 执行交易
- `update_prices()` — 批量更新持仓价格
- `get_total_equity()` — 净资产: cash + MV - borrowed
- `get_leverage_ratio()` — 杠杆率
- `get_margin_ratio()` — 保证金比率
- `get_trade_summary()` — 交易摘要DataFrame
- `accrue_interest()` — 每日利息计提

### `predictor.py` (446行) ⭐ 统计预测引擎
**核心类**: `RealtimePredictor`
- `update_price()` — 更新价格滑动窗口
- `predict()` — 综合预测 (方向 + 置信度)
- 7个因子: momentum, mean_reversion, volume, volatility, trend, rsi, macd

### `leverage_engine.py` (410行) ⭐ 动态杠杆引擎
**核心类**: `LeverageEngine`
- `calculate()` — 多因子杠杆计算
- `_kelly_fraction()` — Kelly公式
- `_volatility_multiplier()` — 波动率调节
- `_performance_multiplier()` — 绩效反馈
- `_portfolio_heat_multiplier()` — 组合热度
- `_drawdown_cap()` — 回撤约束

### `benchmark.py` (409行) ⭐ 基准对比
**核心类**: `BenchmarkTracker`
- `initialize_from_history()` — 从历史数据初始化
- `fetch_nasdaq_history()` — Yahoo获取纳指历史
- `update()` — 每分钟更新基准
- `get_snapshot()` — 获取对比快照
- `get_comparison_summary()` — 文本摘要

### `market_clock.py` (461行) 市场时钟
**核心类**: `MarketClock`
- `get_status()` — 当前市场状态 (4种)
- `is_trading_session()` — 是否可交易
- `is_early_close()` — 是否早收盘
- `is_holiday()` — 是否假期
- `countdown_to_next_open()` — 距离开市倒计时

### `state_persistence.py` (399行) 状态持久化
- `save_state()` — 保存完整状态到JSON
- `load_state()` — 从JSON恢复状态
- `serialize_portfolio/deserialize_portfolio()` — 持仓序列化
- `serialize_benchmark/deserialize_benchmark()` — 基准序列化
- `serialize_accuracy/deserialize_accuracy()` — 准确率序列化

### `accuracy_tracker.py` 准确率追踪
**核心类**: `AccuracyTracker`
- `record_prediction()` — 记录预测
- `confirm_prediction()` — 确认预测结果
- `get_snapshot()` — 获取准确率快照

### `model_inference.py` ML模型推理
**核心类**: `ModelInference`
- `load()` — 加载checkpoint
- `predict()` — 执行推理
- 特征构建 + 过滤 + 前向传播

### `templates/dashboard.html` (646行) 前端仪表盘
- 5个面板: 总览, K线图, 分析, 模型, 交易
- LightweightCharts 4.1.3 图表渲染
- 1秒轮询实时更新
- `safeResize()` 图表自适应

### 其他文件
| 文件 | 用途 |
|------|------|
| `daemon.py` | 守护进程管理 |
| `dashboard.py` | 独立仪表盘后端 |
| `live_simulator.py` | 实时模拟器 |
| `run_watch.py` | 运行监控 |
| `watchdog.py` | 进程守护 |

---

## ml_model/ — 机器学习模块

### `transformer.py` (739行)
**核心类**: `StockTransformer`
- Pre-LN Transformer Encoder 架构
- 28特征 + 4情感输入
- 双头输出: Direction (Sigmoid) + Magnitude (Linear)
- `TimeSeriesTransformer` — 旧版encoder-decoder (保留参考)

### `trainer.py` (1017行)
**核心类**: `ModelTrainer`, `HyperparameterTuner`
- 训练循环 + 验证循环
- 早停 + 学习率调度 (Cosine Annealing)
- 梯度裁剪 + 混合精度训练
- 超参数调优 (dropout, weight_decay, reward_weights)
- `EvaluationResult` — 评估结果数据类

### `data_loader.py`
**职责**: 从Parquet文件加载训练数据
- 数据清洗 + 归一化
- 序列构建 (60步滑动窗口)
- Train/Val 分割

---

## config/ — 全局配置

### `settings.py` (399行)
**核心类**: `ModelConfig`
- 模型架构参数 (d_model, n_heads, n_layers, dropout等)
- 训练参数 (batch_size, lr, epochs, weight_decay等)
- 数据参数 (seq_len, feature_columns)
- 情感特征配置
- 路径配置

### `logging_config.py`
**职责**: 统一日志配置
- 文件handler + 控制台handler
- 日志级别 + 格式

---

## backtesting/ — 回测引擎

| 文件 | 用途 |
|------|------|
| `engine.py` | 回测主引擎 |
| `broker_sim.py` | 券商模拟 (佣金、滑点) |
| `performance.py` | 绩效分析 (夏普、回撤等) |

---

## data_pipeline/ — 数据管道

| 文件 | 用途 |
|------|------|
| `fetcher.py` | 数据获取 (Yahoo/Wikipedia) |
| `cleaner.py` | 数据清洗 |
| `indicators.py` | 技术指标计算 |
| `storage.py` | Parquet存储 (list_keys) |

---

## crawler/ — 新闻爬虫

| 文件 | 用途 |
|------|------|
| `stock_crawler.py` | 股票数据爬取 |
| `news_scraper.py` | 新闻抓取 |
| `news_sentiment.py` | 情感分析 |

---

## scripts/ — 工具脚本

| 脚本 | 用途 |
|------|------|
| `quick_train.py` | 快速训练 (30 epochs) |
| `train_30ep.py` | 30轮训练 |
| `train_100ep_robust.py` | 100轮鲁棒训练 |
| `train_upgraded.py` | 升级版训练 (MPS) |
| `train_upgraded_cpu.py` | CPU版训练 |
| `train_cpu_40.py` | CPU 40轮训练 |
| `quick_train_mps.py` | MPS加速训练 |
| `build_sentiment_features.py` | 构建情感特征 |
| `update_daily_sentiment.py` | 每日情感更新 |
| `codex_monitor.py` | 系统监控 (CPU/内存) |
| `market_monitor.py` | 市场监控 |
| `desktop_notify.py` | 桌面通知 |
| `compare_results.py` | 结果对比 |

---

## 其他模块

| 目录 | 用途 |
|------|------|
| `execution/oms.py` | 订单管理系统 |
| `risk/manager.py` | 风控管理器 |
| `compliance/checker.py` | 合规检查 |
| `monitoring/alerting.py` | 告警系统 |
| `monitoring/system_monitor.py` | 系统监控 |
| `utils/helpers.py` | 工具函数 (safe_divide, 假期等) |
| `utils/constants.py` | 常量定义 |
| `utils/exceptions.py` | 自定义异常 |

---

## 依赖关系图

```
web_server.py
├── portfolio.py
├── predictor.py
├── benchmark.py
│   └── yfinance (^IXIC)
├── leverage_engine.py
│   └── portfolio + accuracy
├── market_clock.py
│   └── utils/helpers (假期)
├── state_persistence.py
├── accuracy_tracker.py
└── model_inference.py
    └── ml_model/transformer.py (StockTransformer)

ml_model/trainer.py
├── ml_model/transformer.py
├── ml_model/data_loader.py
└── config/settings.py (ModelConfig)
```

---

> 📖 **下一步**: 阅读 [CONFIGURATION.md](CONFIGURATION.md) 了解全部配置项
