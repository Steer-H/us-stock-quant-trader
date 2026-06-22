# 系统架构文档

## 1. 总体架构

```
                          ┌──────────────────────┐
                          │   浏览器 (Dashboard)   │
                          │   localhost:8080       │
                          └──────────┬───────────┘
                                     │ HTTP/轮询 1s
                          ┌──────────▼───────────┐
                          │   Flask Web Server     │
                          │   (threaded=True)      │
                          └──────────┬───────────┘
                                     │
              ┌──────────────────────┼──────────────────────┐
              │                      │                      │
     ┌────────▼────────┐   ┌────────▼────────┐   ┌────────▼────────┐
     │   tick_engine()  │   │   REST API      │   │  state_persist  │
     │   (后台线程)      │   │   (8个端点)      │   │  (每分钟)        │
     │   每秒1次循环      │   │                 │   │                 │
     └────────┬────────┘   └────────────────┘   └────────────────┘
              │
    ┌─────────┼─────────────────────────────────────────┐
    │         │         数据流 (每tick)                   │
    │         ▼                                          │
    │  ┌─────────────┐                                   │
    │  │ 价格获取      │  Yahoo Finance v7 API            │
    │  │ fetch_yahoo  │  → 40只股票 + ^IXIC               │
    │  └──────┬──────┘                                   │
    │         ▼                                          │
    │  ┌─────────────┐                                   │
    │  │ 持仓更新      │  Portfolio.update_prices()        │
    │  │ 市值/PnL     │  → 每只持仓的浮动盈亏              │
    │  └──────┬──────┘                                   │
    │         ▼                                          │
    │  ┌─────────────┐                                   │
    │  │ ML预测       │  StockTransformer 推理            │
    │  │ + 统计预测   │  RealtimePredictor fallback       │
    │  └──────┬──────┘                                   │
    │         ▼                                          │
    │  ┌─────────────┐                                   │
    │  │ 杠杆计算      │  Kelly × 波动率 × 绩效 × 热度     │
    │  │ LeverageEngine│ → 0.25x ~ 2.0x                   │
    │  └──────┬──────┘                                   │
    │         ▼                                          │
    │  ┌─────────────┐                                   │
    │  │ 交易决策      │  止盈/止损/预测卖出/时间平仓      │
    │  │ tick_engine  │  → execute_buy/execute_sell       │
    │  └──────┬──────┘                                   │
    │         ▼                                          │
    │  ┌─────────────┐                                   │
    │  │ 基准更新      │  Benchmark.update()               │
    │  │ Benchmark    │  → 纳指权益曲线 + 策略曲线          │
    │  └─────────────┘                                   │
    └────────────────────────────────────────────────────┘
```

## 2. 核心模块详解

### 2.1 web_server.py — 主控模块 (1326行)

**职责**: Flask 服务器 + 交易引擎 + 全局状态管理

**关键全局变量**:
| 变量 | 类型 | 说明 |
|------|------|------|
| `_current_prices` | dict | 40+1只股票最新价格 |
| `_previous_prices` | dict | 前次价格（波动率计算） |
| `_iteration_count` | int | 引擎迭代计数 |
| `_portfolio` | PortfolioManager | 持仓管理器 |
| `_predictor` | RealtimePredictor | 统计预测器 |
| `_benchmark` | BenchmarkTracker | 基准对比 |
| `_leverage_engine` | LeverageEngine | 动态杠杆 |
| `_ml_inference` | ModelInference | ML模型推理 |
| `_ml_ready` | bool | ML模型是否就绪 |

**数据流**:
```
tick_engine() 每秒执行:
  1. fetch_yahoo_prices()     → 获取实时价格
  2. Portfolio.update_prices() → 更新持仓市值
  3. Predictor.update_price()  → 更新统计特征
  4. Benchmark.update()        → 更新基准曲线
  5. 交易门控检查             → is_trading_session?
  6. 价格过期检查             → _price_is_stale?
  7. 交易信号生成             → 止盈/止损/预测卖出
  8. 每60秒 save_state()      → 持久化
```

**REST API 端点** (详见 [API_REFERENCE.md](API_REFERENCE.md)):
| 端点 | 方法 | 说明 |
|------|------|------|
| `/` | GET | 仪表盘HTML |
| `/api/health` | GET | 健康检查 |
| `/api/status` | GET | 完整系统状态 |
| `/api/tickers` | GET | 追踪股票列表 |
| `/api/signals` | GET | 交易指令 |
| `/api/kline/<ticker>` | GET | 单只K线数据 |
| `/api/kline/multi` | GET | 批量K线数据 |
| `/api/benchmark_curve` | GET | 基准曲线数据 |
| `/api/backtest_summary` | GET | 回测摘要 |

### 2.2 portfolio.py — 持仓管理 (581行)

**核心类**: `PortfolioManager`, `HoldingPosition`, `TradeRecord`

**关键方法**:
- `execute_buy(ticker, qty, price)` — 执行买入
- `execute_sell(ticker, qty, price)` — 执行卖出
- `execute_short(ticker, qty, price)` — 执行做空
- `update_prices(prices_dict)` — 批量更新持仓价格
- `get_total_equity()` → cash + MV - borrowed
- `get_leverage_ratio()` → (MV + borrowed) / equity
- `get_margin_ratio()` → equity / (MV + borrowed)
- `accrue_interest()` — 每日利息计提

**P&L计算**:
```
unrealized_pnl = quantity × (current_price - avg_cost)
unrealized_pnl_pct = (current_price / avg_cost - 1) × side_sign
realized_pnl = Σ(sell_price - buy_price) × quantity - commission
day_pnl = current_equity - day_start_equity
```

### 2.3 predictor.py — 统计预测引擎 (446行)

**核心类**: `RealtimePredictor`

**预测因子** (7个技术指标):
1. **动量** — 5/10/20周期价格变化率
2. **均值回归** — 价格偏离MA的程度
3. **成交量信号** — 量价背离检测
4. **波动率** — 短期 vs 长期波动率比值
5. **趋势强度** — 线性回归斜率 + R²
6. **RSI** — 超买超卖判断
7. **MACD** — 金叉死叉信号

**综合评分**:
```
score = Σ(factor_i × weight_i)  →  [-1, +1]
direction = 1 if score > 0 else -1
confidence = 0.5 + |score| × 0.5
```

详见 [ALGORITHMS.md](ALGORITHMS.md)。

### 2.4 leverage_engine.py — 动态杠杆 (410行)

**多因子模型**:
```
leverage = Kelly × VolatilityMult × PerfMult × HeatMult
leverage = clamp(leverage, MIN_LEVERAGE, MAX_LEVERAGE)
leverage = min(leverage, DrawdownCap, MarginCap)
```

**四个因子**:
| 因子 | 方法 | 范围 |
|------|------|------|
| Kelly基础 | `_kelly_fraction()` | 0.0 ~ 2.0x |
| 波动率调节 | `_volatility_multiplier()` | 0.5 ~ 1.5x |
| 绩效反馈 | `_performance_multiplier()` | 0.4 ~ 1.3x |
| 组合热度 | `_portfolio_heat_multiplier()` | 0.6 ~ 1.0x |

**反马丁格尔规则**:
- 胜率 > 60% → 加码 (最多+30%)
- 胜率 < 45% → 减仓 (最多-60%)
- 3连败 → 强制上限 0.5x

### 2.5 benchmark.py — 基准对比 (409行)

**核心类**: `BenchmarkTracker`

**追踪指标**:
- 纳指权益曲线 vs 策略权益曲线
- 累计收益率、年化收益率
- 夏普比率、最大回撤
- Alpha、Beta、信息比率
- 超额收益

**数据流**:
```
启动时: fetch_nasdaq_history('6mo') → 纳指历史数据
运行时: update(nasdaq_price, strategy_equity) → 每分钟追加
读取时: _ensure_curves_synced() → dict → Series 同步
API:    /api/benchmark_curve → 最近300个数据点
```

### 2.6 market_clock.py — 市场时钟 (461行)

**市场状态枚举**:
- `CLOSED` — 休市 (20:00-04:00 ET)
- `PRE_MARKET` — 盘前 (04:00-09:30 ET)
- `REGULAR_HOURS` — 正常交易 (09:30-16:00 ET)
- `AFTER_HOURS` — 盘后 (16:00-20:00 ET)
- `EARLY_CLOSE` — 早收盘 (如Black Friday 13:00)

**假期支持**: 2025-2027年完整美股假期列表

### 2.7 state_persistence.py — 状态持久化 (399行)

**保存内容**:
```json
{
  "saved_at": "时间戳",
  "portfolio": {持仓、现金、PnL、杠杆、交易历史},
  "accuracy": {预测准确率统计},
  "benchmark": {纳指曲线、回撤、收益率},
  "globals": {价格、迭代计数、建仓状态等},
  "predictor": {统计预测器状态}
}
```

**保存策略**: 每60秒自动保存 + 进程退出时保存

### 2.8 model_inference.py — ML推理

**职责**: 加载 StockTransformer checkpoint，执行推理

**处理流程**:
1. 加载最新 checkpoint (`.pt` 文件)
2. 构建特征向量 (28 特征 + 4 情感特征)
3. 前向传播 → 方向预测 + 置信度
4. 过滤策略: 特征数不匹配时只保留交集

### 2.9 前端 dashboard.html (646行)

**5个面板**:
| Tab | ID | 内容 |
|-----|-----|------|
| 📊 总览 | panel-positions | 持仓列表 + 最近交易 + 系统状态 |
| 📈 K线图 | panel-charts | 40只股票的K线图 (Lightweight Charts) |
| 📉 分析 | panel-analysis | 策略vs纳指曲线 + 收益统计 |
| 🧠 模型 | panel-model | AI模型信息 + 辅助数据源 |
| 📋 交易 | panel-trades | 完整交易记录 |

**技术细节**:
- 1秒轮询 `/api/status` 获取实时数据
- LightweightCharts 4.1.3 渲染K线和曲线
- `safeResize()` 处理面板切换时的图表尺寸恢复
- 30次重试机制处理容器0宽度问题

## 3. 线程模型

```
主线程 (Flask)
  ├── HTTP请求处理 (threaded=True, 多线程)
  │   ├── GET /api/status      → build_status_data()
  │   ├── GET /api/kline/*     → fetch_kline_data()
  │   └── GET /api/benchmark_curve → _benchmark数据
  │
  └── engine_thread (daemon)
      └── engine_loop()
          └── while _engine_running:
              ├── tick_engine()        # 价格 + 交易 + 基准
              ├── trigger_offline_training()  # 休市时训练
              └── save_state()         # 状态持久化
              └── time.sleep(1)
```

**线程安全**: CPython GIL 保护单个操作，但迭代中修改 dict 可能导致 RuntimeError。当前依赖轮询的原子性，未加锁。

## 4. 数据存储

| 路径 | 格式 | 内容 |
|------|------|------|
| `data/trading_state.json` | JSON | 完整运行时状态 |
| `models/*.pt` | PyTorch | 训练好的模型权重 |
| `data/processed/*.parquet` | Parquet | 处理后的训练数据 |
| `logs/*.log` | 文本 | 运行日志 |
| `work_logs/*.md` | Markdown | 工作日志 |

---

> 📖 **下一步**: 阅读 [ALGORITHMS.md](ALGORITHMS.md) 了解算法细节
