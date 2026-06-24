# 配置參考

## 1. ModelConfig — 模型配置 (`config/settings.py`)

### 架構參數

| 參數 | 默認值 | 說明 |
|------|--------|------|
| `d_model` | 192 | Transformer 隱藏維度 |
| `n_heads` | 8 | 多頭注意力頭數 |
| `n_layers` | 4 | Encoder 層數 |
| `d_ff` | 768 | Feed-Forward 中間維度 (4×d_model) |
| `dropout` | 0.15 | 通用 Dropout 概率 |
| `attn_dropout` | 0.1 | 注意力 Dropout |
| `drop_path_rate` | 0.1 | Stochastic Depth 概率 |
| `activation` | `'gelu'` | 激活函數 (gelu/relu) |

### 訓練參數

| 參數 | 默認值 | 說明 |
|------|--------|------|
| `batch_size` | 64 | 訓練批次大小 |
| `epochs` | 100 | 訓練輪數 |
| `learning_rate` | 1e-4 | 初始學習率 |
| `min_lr` | 1e-6 | 最小學習率 |
| `weight_decay` | 1e-4 | L2 正則化係數 |
| `grad_clip` | 1.0 | 梯度裁剪閾值 |
| `patience` | 10 | 早停耐心值 |
| `label_smoothing` | 0.1 | 標籤平滑因子 |
| `focal_gamma` | 2.0 | Focal Loss γ |
| `direction_reward_weight` | 0.6 | 回歸損失權重 (MSE) |
| `magnitude_reward_weight` | 0.4 | 分類損失權重 (BCE) |

### 數據參數

| 參數 | 默認值 | 說明 |
|------|--------|------|
| `seq_len` | 60 | 輸入序列長度 |
| `feature_columns` | 28 | 技術指標數量 |
| `prediction_horizon` | 1 | 預測步長 |
| `use_sentiment` | True | 是否使用情感特徵 |

### 硬體參數

| 參數 | 默認值 | 說明 |
|------|--------|------|
| `device` | `'mps'` (Apple Silicon) | 訓練設備 (mps/cpu/cuda) |
| `use_amp` | True | 混合精度訓練 |
| `num_workers` | 0 | DataLoader 工作線程 |

### 路徑配置

| 參數 | 默認值 | 說明 |
|------|--------|------|
| `data_dir` | `Path('data')` | 數據目錄 |
| `model_dir` | `Path('models')` | 模型保存目錄 |
| `log_dir` | `Path('logs')` | 日誌目錄 |

---

## 2. 交易參數 (`live_trading/web_server.py`)

### 跟蹤股票 (40隻)

```python
TRACKED_TICKERS = [
    # 科技七巨頭
    'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'META', 'TSLA',
    # 軟體/SaaS
    'NFLX', 'ADBE', 'CRM', 'NOW', 'ORCL',
    # 金融
    'JPM', 'V', 'MA', 'BAC',
    # 消費
    'WMT', 'HD', 'NKE', 'SBUX', 'UBER',
    # 晶片/半導體
    'AVGO', 'AMD', 'INTC', 'QCOM', 'TXN',
    # 光通信 (高波動)
    'AAOI', 'COHR', 'LITE', 'FN',
    # 存儲 (高波動)
    'WDC', 'STX', 'NTAP',
    # 數據中心晶片
    'MRVL', 'MU',
    # 半導體設備
    'LRCX', 'AMAT', 'KLAC',
    # EDA軟體
    'SNPS', 'CDNS',
]
```

### 風險控制

| 參數 | 默認值 | 說明 |
|------|--------|------|
| `PROFIT_TAKE_THRESHOLD` | 0.03 (3%) | 止盈閾值 |
| `STOP_LOSS_THRESHOLD` | -0.04 (4%) | 止損閾值 |
| `MAX_POSITION_HOLD_TIME` | 30 (分鐘) | 最大持倉時間 |
| `REENTRY_COOLDOWN` | 5 (分鐘) | 賣出後冷卻時間 |
| `POSITION_MAX_PCT` | 0.08 (8%) | 單只最大倉位 |
| `PREDICTIVE_SELL_THRESHOLD` | 0.55 | 預測賣出觸發準確率 |

### 槓桿參數

| 參數 | 默認值 | 說明 |
|------|--------|------|
| `MAX_LEVERAGE` | 2.0 | 最大槓桿倍數 |
| `MIN_LEVERAGE` | 0.25 | 最小槓桿倍數 |
| `MAX_POSITION_PCT_LEVERAGED` | 0.12 (12%) | 槓桿模式最大倉位 |
| `LEVERAGE_STOP_LOSS` | -0.025 (2.5%) | 槓桿止損 |
| `DRAWDOWN_DELEVERAGE` | 0.10 (10%) | 回撤去槓桿觸發 |

### 槓桿引擎內部常量 (`leverage_engine.py`)

| 參數 | 默認值 | 說明 |
|------|--------|------|
| `MAX_LEVERAGE` | 2.0 | 全局最大槓桿 |
| `MIN_LEVERAGE` | 0.25 | 全局最小槓桿 |
| `DEFAULT_WIN_LOSS_RATIO` | 1.5 | 默認盈虧比 |
| `MAX_WIN_LOSS_RATIO` | 3.0 | 最大盈虧比 |
| `HALF_KELLY_FRACTION` | 0.5 | Half-Kelly係數 |
| `PERF_BOOST_WINRATE` | 0.60 | 加碼觸發勝率 |
| `PERF_REDUCE_WINRATE` | 0.45 | 減倉觸發勝率 |

### Yahoo Finance 抓取

| 參數 | 默認值 | 說明 |
|------|--------|------|
| `YAHOO_INTERVALS['REGULAR']` | 12 (秒) | 正常交易時段抓取間隔 |
| `YAHOO_INTERVALS['PRE_MARKET']` | 60 (秒) | 盤前抓取間隔 |
| `YAHOO_INTERVALS['AFTER_HOURS']` | 60 (秒) | 盤後抓取間隔 |
| `YAHOO_INTERVALS['CLOSED']` | 60 (秒) | 閉市抓取間隔 |

---

## 3. 市場時間 (`market_clock.py`)

```python
MARKET_TIMES = {
    'pre_market_start':   time(4, 0),     # 04:00 ET
    'regular_start':      time(9, 30),    # 09:30 ET
    'regular_end':        time(16, 0),    # 16:00 ET
    'early_close_end':    time(13, 0),    # 13:00 ET (Black Friday)
    'after_hours_end':    time(20, 0),    # 20:00 ET
}
```

**假期數據**: 2025-2027年完整美股假期 (Martin Luther King Jr. Day, Presidents Day, Good Friday, Memorial Day, Juneteenth, Independence Day, Labor Day, Thanksgiving, Christmas, New Year's Day + 浮動調整)

---

## 4. 資源配置建議

### 開發/測試環境
```python
ModelConfig:
    d_model = 128
    n_layers = 2
    n_heads = 4
    batch_size = 32
    epochs = 30
    device = 'cpu'
```

### 生產環境 (Apple Silicon)
```python
ModelConfig:
    d_model = 192
    n_layers = 4
    n_heads = 8
    batch_size = 64
    epochs = 100
    device = 'mps'
    use_amp = True
```

### 生產環境 (NVIDIA GPU)
```python
ModelConfig:
    d_model = 256
    n_layers = 6
    n_heads = 8
    batch_size = 128
    epochs = 100
    device = 'cuda'
    use_amp = True
```

---

## 5. 修改配置後

⚠️ 修改 `ModelConfig` 的參數（尤其是架構參數如 `d_model`、`n_layers`）會導致舊 checkpoint 不兼容。需要：

1. 刪除舊模型: `rm models/*.pt`
2. 重新訓練: `python3 scripts/quick_train.py`
3. 重啟服務

---

> 📖 **返回**: [README.md](README.md) 文檔首頁
