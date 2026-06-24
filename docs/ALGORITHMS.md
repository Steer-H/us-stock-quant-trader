# 算法詳解

## 1. StockTransformer — 時序預測模型

### 1.1 架構

基於 "Attention Is All You Need" (Vaswani et al., 2017)，針對金融時間序列優化：

```
輸入層 (Batch, SeqLen, Features)    28個技術指標 + 4個情感特徵
    │
    ▼
Input Projection (Linear)           特徵 → d_model (默認192維)
    │
    ▼
Positional Encoding                 正弦/餘弦位置編碼
    │
    ▼
Transformer Encoder × N_Layers      (默認4層)
    ├── Multi-Head Self-Attention   (默認8頭)
    │   ├── Q, K, V 投影
    │   ├── Scaled Dot-Product Attention
    │   └── 可選: Flash Attention (MPS後端)
    ├── Feed-Forward Network        d_model → 4×d_model → d_model
    │   └── GELU 激活
    ├── Pre-LN (LayerNorm before each sublayer)
    ├── Dropout + DropPath (Stochastic Depth)
    └── Residual Connections
    │
    ▼
Global Average Pooling             聚合時序維度
    │
    ▼
Final LayerNorm
    │
    ▼
輸出頭 (並行雙頭)
├── Direction Head: Linear → 1 + Sigmoid  → 漲跌概率 [0,1]
└── Magnitude Head: Linear → 1            → 預期漲跌幅
```

### 1.2 關鍵設計

| 特性 | 說明 |
|------|------|
| Pre-LN | LayerNorm 在子層之前，訓練更穩定 |
| GELU | 比 ReLU 更平滑，適合金融數據 |
| Stochastic Depth | DropPath 概率丟棄層，正則化 |
| Label Smoothing | 0.1 平滑因子，防止過置信 |
| Focal Loss | 降低易分類樣本權重，關注難例 |

### 1.3 輸入特徵 (28個)

```
技術指標:
  - 價格: Open, High, Low, Close (歸一化)
  - 收益率: 1d/5d/10d/20d 對數收益率
  - 波動率: 5d/10d/20d 歷史波動率
  - 動量: 5d/10d/20d 價格變化率
  - 成交量: Volume, Volume MA, Volume Ratio
  - 技術指標: RSI(14), MACD, MACD Signal, MACD Hist
  - 均線: MA5, MA10, MA20, MA60 偏離度
  - 布林帶: BB Upper/Lower 偏離度

情感特徵 (4個，小權重):
  - news_sentiment_3d: 3日新聞情感 (-1 ~ +1)
  - news_sentiment_7d: 7日新聞情感 (-1 ~ +1)
  - earnings_surprise_pct: 財報超預期幅度
  - has_earnings_report: 是否有財報 (0/1)
```

### 1.4 損失函數

```python
Total_Loss = w_reg × MSE(predicted_return, actual_return)
           + w_cls × BCE(direction_prob, actual_direction)
           + w_reg  × L2_Regularization
```

- `w_reg` = 0.6 (回歸損失權重)
- `w_cls` = 0.4 (分類損失權重)
- Focal Loss γ = 2.0 (聚焦難分類樣本)

### 1.5 訓練配置

| 參數 | 默認值 |
|------|--------|
| 序列長度 | 60 個時間步 |
| Batch Size | 64 |
| 學習率 | 1e-4 → 1e-6 (Cosine Annealing) |
| 優化器 | AdamW (weight_decay=1e-4) |
| Epochs | 30 (快速) / 100 (完整) |
| 早停 | patience=10, 監控 val_loss |
| 梯度裁剪 | 1.0 |

---

## 2. 統計預測器 — RealtimePredictor

### 2.1 設計目的

輕量級實時預測引擎，當 ML 模型不可用時作為 fallback。

### 2.2 預測流程

```
1. 價格更新 → update_price(ticker, price)
   └── 維護滑動窗口: 最近60個價格

2. 預測調用 → predict(ticker)
   ├── 數據檢查: len(prices) >= 10?
   ├── 計算7個因子:
   │   ├── _calc_momentum(prices)        權重: 0.25
   │   ├── _calc_mean_reversion(prices)  權重: 0.20
   │   ├── _calc_volume_signal(prices, volumes) 權重: 0.10
   │   ├── _calc_volatility(prices)      權重: 0.15
   │   ├── _calc_trend_strength(prices)  權重: 0.15
   │   ├── _calc_rsi(prices)             權重: 0.10
   │   └── _calc_macd(prices)            權重: 0.05
   └── score = Σ(factor × weight) → [-1, +1]
       direction = 1 if score > 0 else -1
       confidence = 0.5 + |score| × 0.5
```

### 2.3 因子詳解

#### 動量因子 (權重 0.25)
```
momentum_5 = (p[-1] - p[-5]) / p[-5]
momentum_10 = (p[-1] - p[-10]) / p[-10]
momentum_20 = (p[-1] - p[-20]) / p[-20]
signal = 0.5×m5 + 0.3×m10 + 0.2×m20
clamp(signal, -1, 1)
```

#### 均值回歸因子 (權重 0.20)
```
ma_5 = mean(p[-5:]), ma_20 = mean(p[-20:])
deviation = (p[-1] - ma_20) / ma_20
signal = -tanh(deviation × 5)  # 偏離越大，回歸信號越強
```

#### 波動率因子 (權重 0.15)
```
short_vol = std(returns[-5:])
long_vol = std(returns[-20:])
vol_ratio = short_vol / long_vol
signal = (1 - vol_ratio) × 0.5  # 高波動 → 謹慎
```

#### 趨勢強度 (權重 0.15)
```
x = [0, 1, ..., 19], y = prices[-20:]
slope, r2 = linear_regression(x, y)
signal = tanh(slope × 100) × r2  # 強趨勢 + 高R² = 強信號
```

---

## 3. 動態槓桿引擎 — LeverageEngine

### 3.1 Kelly 公式基礎

```
f* = (p × b - q) / b

其中:
  p = 勝率 (模型置信度)
  q = 1 - p
  b = 盈虧比 (win_loss_ratio)

Half-Kelly (保守):
  f_half = f* / 2

槓桿轉換:
  leverage = 1.0 + f_half × (MAX_LEVERAGE - 1.0) / 0.5
```

### 3.2 多因子疊加

```
leverage_raw = Kelly × VolatilityMult × PerfMult × HeatMult

約束鏈:
  1. leverage = clamp(leverage_raw, 0, MAX_LEVERAGE)
  2. leverage = min(leverage, DrawdownCap)
  3. leverage = min(leverage, MarginCap)
  4. Kelly ≤ 0 → leverage = 0 (不下注)
  5. 否則 leverage = max(leverage, MIN_LEVERAGE)
  6. 四捨五入到 0.05
```

### 3.3 波動率乘數

```
annual_vol = σ_daily × √252

if annual_vol < 0.15:   multiplier = 1.3   (低波動 → 加槓桿)
elif annual_vol < 0.25:  multiplier = 1.0   (正常)
elif annual_vol < 0.40:  multiplier = 0.7   (高波動 → 減槓桿)
else:                    multiplier = 0.5   (極高波動 → 大幅減槓桿)
```

### 3.4 回撤約束

```
drawdown = (peak_equity - current_equity) / peak_equity

if drawdown < 0.05:    cap = MAX_LEVERAGE
elif drawdown < 0.10:   cap = 1.5
elif drawdown < 0.15:   cap = 1.0
else:                   cap = 0.5
```

### 3.5 保證金約束

```
margin_ratio = equity / (MV + borrowed)

if margin_ratio > 0.90:  cap = 0.0   (停止開倉)
elif margin_ratio > 0.80: cap = 1.0
elif margin_ratio > 0.60: cap = 1.5
else:                     cap = MAX_LEVERAGE
```

---

## 4. 交易決策引擎

### 4.1 決策樹 (每個tick)

```
for each 持倉 position:
    ├── 條件1: 止盈
    │   └── unrealized_pnl_pct >= 3% → 賣出
    ├── 條件2: 止損
    │   └── unrealized_pnl_pct <= -4% → 賣出
    ├── 條件3: 預測性賣出
    │   └── ML/統計預測做空 + 準確率 > 55% → 50%概率賣出
    └── 條件4: 時間平倉
        └── 持倉 > 30分鐘 → 平倉

for each 候選買入:
    ├── 現金 > $1000?
    ├── 預測為漲 (direction > 0)?
    ├── 置信度 > 55%?
    ├── 槓桿允許?
    └── 權重不超標? → 執行買入
```

### 4.2 交易門控

```
if not is_trading_session:   # 休市/盤前/盤後
    return                    # 不執行任何交易

if _price_is_stale:           # 價格過期
    return                    # 跳過交易決策
```

---

## 5. 基準對比算法

### 5.1 納指基準構建

```
啟動時:
  1. Yahoo Finance 獲取 ^IXIC 6個月歷史數據
  2. nasdaq_shares = initial_capital / nasdaq_start_price
  3. nasdaq_equity = nasdaq_shares × nasdaq_price(t)

運行時:
  1. 每分鐘更新 nasdaq_price
  2. 追加到 equity_curve (dict, 每100次同步到Series)

比較:
  strategy_return = (equity - initial) / initial
  nasdaq_return = (nasdaq_equity - initial) / initial
  excess_return = strategy_return - nasdaq_return
```

### 5.2 夏普比率

```
日收益率: daily_returns = equity_curve.pct_change()
年化收益: annual_return = (1 + total_return)^(365/days) - 1
年化波動: annual_vol = daily_returns.std() × √252
夏普比率: sharpe = annual_return / annual_vol
```

### 5.3 Alpha / Beta

```
對齊策略日收益和納指日收益
Beta = Cov(strategy, nasdaq) / Var(nasdaq)
Alpha = strategy_mean - Beta × nasdaq_mean
信息比率: IR = Alpha / tracking_error
```

---

## 6. 準確率追蹤

### 6.1 方向準確率

```
prediction: 方向預測 (漲/跌)
actual:     實際結果 (收盤價 vs 預測價格)

等待確認: 最多10個tick後與實際結果比較
direction_accuracy = correct_predictions / total_confirmed
```

### 6.2 分層統計

```
recent_50_accuracy:   最近50次預測準確率
correct_long/total_long:   做多正確率
correct_short/total_short: 做空正確率
accuracy_trend:  'improving' / 'stable' / 'declining'
```

---

> 📖 **下一步**: 閱讀 [DEPLOYMENT.md](DEPLOYMENT.md) 了解部署指南
