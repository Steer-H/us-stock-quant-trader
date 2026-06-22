# 算法详解

## 1. StockTransformer — 时序预测模型

### 1.1 架构

基于 "Attention Is All You Need" (Vaswani et al., 2017)，针对金融时间序列优化：

```
输入层 (Batch, SeqLen, Features)    28个技术指标 + 4个情感特征
    │
    ▼
Input Projection (Linear)           特征 → d_model (默认192维)
    │
    ▼
Positional Encoding                 正弦/余弦位置编码
    │
    ▼
Transformer Encoder × N_Layers      (默认4层)
    ├── Multi-Head Self-Attention   (默认8头)
    │   ├── Q, K, V 投影
    │   ├── Scaled Dot-Product Attention
    │   └── 可选: Flash Attention (MPS后端)
    ├── Feed-Forward Network        d_model → 4×d_model → d_model
    │   └── GELU 激活
    ├── Pre-LN (LayerNorm before each sublayer)
    ├── Dropout + DropPath (Stochastic Depth)
    └── Residual Connections
    │
    ▼
Global Average Pooling             聚合时序维度
    │
    ▼
Final LayerNorm
    │
    ▼
输出头 (并行双头)
├── Direction Head: Linear → 1 + Sigmoid  → 涨跌概率 [0,1]
└── Magnitude Head: Linear → 1            → 预期涨跌幅
```

### 1.2 关键设计

| 特性 | 说明 |
|------|------|
| Pre-LN | LayerNorm 在子层之前，训练更稳定 |
| GELU | 比 ReLU 更平滑，适合金融数据 |
| Stochastic Depth | DropPath 概率丢弃层，正则化 |
| Label Smoothing | 0.1 平滑因子，防止过置信 |
| Focal Loss | 降低易分类样本权重，关注难例 |

### 1.3 输入特征 (28个)

```
技术指标:
  - 价格: Open, High, Low, Close (归一化)
  - 收益率: 1d/5d/10d/20d 对数收益率
  - 波动率: 5d/10d/20d 历史波动率
  - 动量: 5d/10d/20d 价格变化率
  - 成交量: Volume, Volume MA, Volume Ratio
  - 技术指标: RSI(14), MACD, MACD Signal, MACD Hist
  - 均线: MA5, MA10, MA20, MA60 偏离度
  - 布林带: BB Upper/Lower 偏离度

情感特征 (4个，小权重):
  - news_sentiment_3d: 3日新闻情感 (-1 ~ +1)
  - news_sentiment_7d: 7日新闻情感 (-1 ~ +1)
  - earnings_surprise_pct: 财报超预期幅度
  - has_earnings_report: 是否有财报 (0/1)
```

### 1.4 损失函数

```python
Total_Loss = w_reg × MSE(predicted_return, actual_return)
           + w_cls × BCE(direction_prob, actual_direction)
           + w_reg  × L2_Regularization
```

- `w_reg` = 0.6 (回归损失权重)
- `w_cls` = 0.4 (分类损失权重)
- Focal Loss γ = 2.0 (聚焦难分类样本)

### 1.5 训练配置

| 参数 | 默认值 |
|------|--------|
| 序列长度 | 60 个时间步 |
| Batch Size | 64 |
| 学习率 | 1e-4 → 1e-6 (Cosine Annealing) |
| 优化器 | AdamW (weight_decay=1e-4) |
| Epochs | 30 (快速) / 100 (完整) |
| 早停 | patience=10, 监控 val_loss |
| 梯度裁剪 | 1.0 |

---

## 2. 统计预测器 — RealtimePredictor

### 2.1 设计目的

轻量级实时预测引擎，当 ML 模型不可用时作为 fallback。

### 2.2 预测流程

```
1. 价格更新 → update_price(ticker, price)
   └── 维护滑动窗口: 最近60个价格

2. 预测调用 → predict(ticker)
   ├── 数据检查: len(prices) >= 10?
   ├── 计算7个因子:
   │   ├── _calc_momentum(prices)        权重: 0.25
   │   ├── _calc_mean_reversion(prices)  权重: 0.20
   │   ├── _calc_volume_signal(prices, volumes) 权重: 0.10
   │   ├── _calc_volatility(prices)      权重: 0.15
   │   ├── _calc_trend_strength(prices)  权重: 0.15
   │   ├── _calc_rsi(prices)             权重: 0.10
   │   └── _calc_macd(prices)            权重: 0.05
   └── score = Σ(factor × weight) → [-1, +1]
       direction = 1 if score > 0 else -1
       confidence = 0.5 + |score| × 0.5
```

### 2.3 因子详解

#### 动量因子 (权重 0.25)
```
momentum_5 = (p[-1] - p[-5]) / p[-5]
momentum_10 = (p[-1] - p[-10]) / p[-10]
momentum_20 = (p[-1] - p[-20]) / p[-20]
signal = 0.5×m5 + 0.3×m10 + 0.2×m20
clamp(signal, -1, 1)
```

#### 均值回归因子 (权重 0.20)
```
ma_5 = mean(p[-5:]), ma_20 = mean(p[-20:])
deviation = (p[-1] - ma_20) / ma_20
signal = -tanh(deviation × 5)  # 偏离越大，回归信号越强
```

#### 波动率因子 (权重 0.15)
```
short_vol = std(returns[-5:])
long_vol = std(returns[-20:])
vol_ratio = short_vol / long_vol
signal = (1 - vol_ratio) × 0.5  # 高波动 → 谨慎
```

#### 趋势强度 (权重 0.15)
```
x = [0, 1, ..., 19], y = prices[-20:]
slope, r2 = linear_regression(x, y)
signal = tanh(slope × 100) × r2  # 强趋势 + 高R² = 强信号
```

---

## 3. 动态杠杆引擎 — LeverageEngine

### 3.1 Kelly 公式基础

```
f* = (p × b - q) / b

其中:
  p = 胜率 (模型置信度)
  q = 1 - p
  b = 盈亏比 (win_loss_ratio)

Half-Kelly (保守):
  f_half = f* / 2

杠杆转换:
  leverage = 1.0 + f_half × (MAX_LEVERAGE - 1.0) / 0.5
```

### 3.2 多因子叠加

```
leverage_raw = Kelly × VolatilityMult × PerfMult × HeatMult

约束链:
  1. leverage = clamp(leverage_raw, 0, MAX_LEVERAGE)
  2. leverage = min(leverage, DrawdownCap)
  3. leverage = min(leverage, MarginCap)
  4. Kelly ≤ 0 → leverage = 0 (不下注)
  5. 否则 leverage = max(leverage, MIN_LEVERAGE)
  6. 四舍五入到 0.05
```

### 3.3 波动率乘数

```
annual_vol = σ_daily × √252

if annual_vol < 0.15:   multiplier = 1.3   (低波动 → 加杠杆)
elif annual_vol < 0.25:  multiplier = 1.0   (正常)
elif annual_vol < 0.40:  multiplier = 0.7   (高波动 → 减杠杆)
else:                    multiplier = 0.5   (极高波动 → 大幅减杠杆)
```

### 3.4 回撤约束

```
drawdown = (peak_equity - current_equity) / peak_equity

if drawdown < 0.05:    cap = MAX_LEVERAGE
elif drawdown < 0.10:   cap = 1.5
elif drawdown < 0.15:   cap = 1.0
else:                   cap = 0.5
```

### 3.5 保证金约束

```
margin_ratio = equity / (MV + borrowed)

if margin_ratio > 0.90:  cap = 0.0   (停止开仓)
elif margin_ratio > 0.80: cap = 1.0
elif margin_ratio > 0.60: cap = 1.5
else:                     cap = MAX_LEVERAGE
```

---

## 4. 交易决策引擎

### 4.1 决策树 (每个tick)

```
for each 持仓 position:
    ├── 条件1: 止盈
    │   └── unrealized_pnl_pct >= 3% → 卖出
    ├── 条件2: 止损
    │   └── unrealized_pnl_pct <= -4% → 卖出
    ├── 条件3: 预测性卖出
    │   └── ML/统计预测做空 + 准确率 > 55% → 50%概率卖出
    └── 条件4: 时间平仓
        └── 持仓 > 30分钟 → 平仓

for each 候选买入:
    ├── 现金 > $1000?
    ├── 预测为涨 (direction > 0)?
    ├── 置信度 > 55%?
    ├── 杠杆允许?
    └── 权重不超标? → 执行买入
```

### 4.2 交易门控

```
if not is_trading_session:   # 休市/盘前/盘后
    return                    # 不执行任何交易

if _price_is_stale:           # 价格过期
    return                    # 跳过交易决策
```

---

## 5. 基准对比算法

### 5.1 纳指基准构建

```
启动时:
  1. Yahoo Finance 获取 ^IXIC 6个月历史数据
  2. nasdaq_shares = initial_capital / nasdaq_start_price
  3. nasdaq_equity = nasdaq_shares × nasdaq_price(t)

运行时:
  1. 每分钟更新 nasdaq_price
  2. 追加到 equity_curve (dict, 每100次同步到Series)

比较:
  strategy_return = (equity - initial) / initial
  nasdaq_return = (nasdaq_equity - initial) / initial
  excess_return = strategy_return - nasdaq_return
```

### 5.2 夏普比率

```
日收益率: daily_returns = equity_curve.pct_change()
年化收益: annual_return = (1 + total_return)^(365/days) - 1
年化波动: annual_vol = daily_returns.std() × √252
夏普比率: sharpe = annual_return / annual_vol
```

### 5.3 Alpha / Beta

```
对齐策略日收益和纳指日收益
Beta = Cov(strategy, nasdaq) / Var(nasdaq)
Alpha = strategy_mean - Beta × nasdaq_mean
信息比率: IR = Alpha / tracking_error
```

---

## 6. 准确率追踪

### 6.1 方向准确率

```
prediction: 方向预测 (涨/跌)
actual:     实际结果 (收盘价 vs 预测价格)

等待确认: 最多10个tick后与实际结果比较
direction_accuracy = correct_predictions / total_confirmed
```

### 6.2 分层统计

```
recent_50_accuracy:   最近50次预测准确率
correct_long/total_long:   做多正确率
correct_short/total_short: 做空正确率
accuracy_trend:  'improving' / 'stable' / 'declining'
```

---

> 📖 **下一步**: 阅读 [DEPLOYMENT.md](DEPLOYMENT.md) 了解部署指南
