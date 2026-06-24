# 代碼結構 — 每個文件的職責

## live_trading/ — 核心交易系統

### `web_server.py` (1326行) ⭐ 主控模塊
**職責**: Flask 伺服器 + 交易引擎 + 全局狀態管理
- `tick_engine()` — 每秒執行的交易循環
- `fetch_yahoo_prices()` — Yahoo v7 API 批量獲取價格
- `fetch_kline_data()` — 獲取K線數據
- `init_positions()` — 開盤建倉
- `build_status_data()` — 構建 `/api/status` 響應
- `engine_loop()` — 後臺引擎主循環
- `init_ml_model()` — 異步加載 ML 模型
- `_collect_globals_dict()` — 收集全局狀態供持久化
- `start_server()` — 啟動入口
- **8個 REST 端點**: health, tickers, signals, kline, kline/multi, status, benchmark_curve, backtest_summary

### `portfolio.py` (581行) ⭐ 持倉管理
**核心類**: `PortfolioManager`, `HoldingPosition`, `TradeRecord`
- `execute_buy/sell/short()` — 執行交易
- `update_prices()` — 批量更新持倉價格
- `get_total_equity()` — 淨資產: cash + MV - borrowed
- `get_leverage_ratio()` — 槓桿率
- `get_margin_ratio()` — 保證金比率
- `get_trade_summary()` — 交易摘要DataFrame
- `accrue_interest()` — 每日利息計提

### `predictor.py` (446行) ⭐ 統計預測引擎
**核心類**: `RealtimePredictor`
- `update_price()` — 更新價格滑動窗口
- `predict()` — 綜合預測 (方向 + 置信度)
- 7個因子: momentum, mean_reversion, volume, volatility, trend, rsi, macd

### `leverage_engine.py` (410行) ⭐ 動態槓桿引擎
**核心類**: `LeverageEngine`
- `calculate()` — 多因子槓桿計算
- `_kelly_fraction()` — Kelly公式
- `_volatility_multiplier()` — 波動率調節
- `_performance_multiplier()` — 績效反饋
- `_portfolio_heat_multiplier()` — 組合熱度
- `_drawdown_cap()` — 回撤約束

### `benchmark.py` (409行) ⭐ 基準對比
**核心類**: `BenchmarkTracker`
- `initialize_from_history()` — 從歷史數據初始化
- `fetch_nasdaq_history()` — Yahoo獲取納指歷史
- `update()` — 每分鐘更新基準
- `get_snapshot()` — 獲取對比快照
- `get_comparison_summary()` — 文本摘要

### `market_clock.py` (461行) 市場時鐘
**核心類**: `MarketClock`
- `get_status()` — 當前市場狀態 (4種)
- `is_trading_session()` — 是否可交易
- `is_early_close()` — 是否早收盤
- `is_holiday()` — 是否假期
- `countdown_to_next_open()` — 距離開市倒計時

### `state_persistence.py` (399行) 狀態持久化
- `save_state()` — 保存完整狀態到JSON
- `load_state()` — 從JSON恢復狀態
- `serialize_portfolio/deserialize_portfolio()` — 持倉序列化
- `serialize_benchmark/deserialize_benchmark()` — 基準序列化
- `serialize_accuracy/deserialize_accuracy()` — 準確率序列化

### `accuracy_tracker.py` 準確率追蹤
**核心類**: `AccuracyTracker`
- `record_prediction()` — 記錄預測
- `confirm_prediction()` — 確認預測結果
- `get_snapshot()` — 獲取準確率快照

### `model_inference.py` ML模型推理
**核心類**: `ModelInference`
- `load()` — 加載checkpoint
- `predict()` — 執行推理
- 特徵構建 + 過濾 + 前向傳播

### `templates/dashboard.html` (646行) 前端儀錶盤
- 5個面板: 總覽, K線圖, 分析, 模型, 交易
- LightweightCharts 4.1.3 圖表渲染
- 1秒輪詢實時更新
- `safeResize()` 圖表自適應

### 其他文件
| 文件 | 用途 |
|------|------|
| `daemon.py` | 守護進程管理 |
| `dashboard.py` | 獨立儀錶盤後端 |
| `live_simulator.py` | 實時模擬器 |
| `run_watch.py` | 運行監控 |
| `watchdog.py` | 進程守護 |

---

## ml_model/ — 機器學習模塊

### `transformer.py` (739行)
**核心類**: `StockTransformer`
- Pre-LN Transformer Encoder 架構
- 28特徵 + 4情感輸入
- 雙頭輸出: Direction (Sigmoid) + Magnitude (Linear)
- `TimeSeriesTransformer` — 舊版encoder-decoder (保留參考)

### `trainer.py` (1017行)
**核心類**: `ModelTrainer`, `HyperparameterTuner`
- 訓練循環 + 驗證循環
- 早停 + 學習率調度 (Cosine Annealing)
- 梯度裁剪 + 混合精度訓練
- 超參數調優 (dropout, weight_decay, reward_weights)
- `EvaluationResult` — 評估結果數據類

### `data_loader.py`
**職責**: 從Parquet文件加載訓練數據
- 數據清洗 + 歸一化
- 序列構建 (60步滑動窗口)
- Train/Val 分割

---

## config/ — 全局配置

### `settings.py` (399行)
**核心類**: `ModelConfig`
- 模型架構參數 (d_model, n_heads, n_layers, dropout等)
- 訓練參數 (batch_size, lr, epochs, weight_decay等)
- 數據參數 (seq_len, feature_columns)
- 情感特徵配置
- 路徑配置

### `logging_config.py`
**職責**: 統一日誌配置
- 文件handler + 控制臺handler
- 日誌級別 + 格式

---

## backtesting/ — 回測引擎

| 文件 | 用途 |
|------|------|
| `engine.py` | 回測主引擎 |
| `broker_sim.py` | 券商模擬 (佣金、滑點) |
| `performance.py` | 績效分析 (夏普、回撤等) |

---

## data_pipeline/ — 數據管道

| 文件 | 用途 |
|------|------|
| `fetcher.py` | 數據獲取 (Yahoo/Wikipedia) |
| `cleaner.py` | 數據清洗 |
| `indicators.py` | 技術指標計算 |
| `storage.py` | Parquet存儲 (list_keys) |

---

## crawler/ — 新聞爬蟲

| 文件 | 用途 |
|------|------|
| `stock_crawler.py` | 股票數據爬取 |
| `news_scraper.py` | 新聞抓取 |
| `news_sentiment.py` | 情感分析 |

---

## scripts/ — 工具腳本

| 腳本 | 用途 |
|------|------|
| `quick_train.py` | 快速訓練 (30 epochs) |
| `train_30ep.py` | 30輪訓練 |
| `train_100ep_robust.py` | 100輪魯棒訓練 |
| `train_upgraded.py` | 升級版訓練 (MPS) |
| `train_upgraded_cpu.py` | CPU版訓練 |
| `train_cpu_40.py` | CPU 40輪訓練 |
| `quick_train_mps.py` | MPS加速訓練 |
| `build_sentiment_features.py` | 構建情感特徵 |
| `update_daily_sentiment.py` | 每日情感更新 |
| `codex_monitor.py` | 系統監控 (CPU/內存) |
| `market_monitor.py` | 市場監控 |
| `desktop_notify.py` | 桌面通知 |
| `compare_results.py` | 結果對比 |

---

## 其他模塊

| 目錄 | 用途 |
|------|------|
| `execution/oms.py` | 訂單管理系統 |
| `risk/manager.py` | 風控管理器 |
| `compliance/checker.py` | 合規檢查 |
| `monitoring/alerting.py` | 告警系統 |
| `monitoring/system_monitor.py` | 系統監控 |
| `utils/helpers.py` | 工具函數 (safe_divide, 假期等) |
| `utils/constants.py` | 常量定義 |
| `utils/exceptions.py` | 自定義異常 |

---

## 依賴關係圖

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

> 📖 **下一步**: 閱讀 [CONFIGURATION.md](CONFIGURATION.md) 了解全部配置項
