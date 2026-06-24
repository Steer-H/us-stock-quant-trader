# 系統架構文檔

## 1. 總體架構

```
                          ┌──────────────────────┐
                          │   瀏覽器 (Dashboard)   │
                          │   localhost:8080       │
                          └──────────┬───────────┘
                                     │ HTTP/輪詢 1s
                          ┌──────────▼───────────┐
                          │   Flask Web Server     │
                          │   (threaded=True)      │
                          └──────────┬───────────┘
                                     │
              ┌──────────────────────┼──────────────────────┐
              │                      │                      │
     ┌────────▼────────┐   ┌────────▼────────┐   ┌────────▼────────┐
     │   tick_engine()  │   │   REST API      │   │  state_persist  │
     │   (後臺線程)      │   │   (8個端點)      │   │  (每分鐘)        │
     │   每秒1次循環      │   │                 │   │                 │
     └────────┬────────┘   └────────────────┘   └────────────────┘
              │
    ┌─────────┼─────────────────────────────────────────┐
    │         │         數據流 (每tick)                   │
    │         ▼                                          │
    │  ┌─────────────┐                                   │
    │  │ 價格獲取      │  Yahoo Finance v7 API            │
    │  │ fetch_yahoo  │  → 40隻股票 + ^IXIC               │
    │  └──────┬──────┘                                   │
    │         ▼                                          │
    │  ┌─────────────┐                                   │
    │  │ 持倉更新      │  Portfolio.update_prices()        │
    │  │ 市值/PnL     │  → 每隻持倉的浮動盈虧              │
    │  └──────┬──────┘                                   │
    │         ▼                                          │
    │  ┌─────────────┐                                   │
    │  │ ML預測       │  StockTransformer 推理            │
    │  │ + 統計預測   │  RealtimePredictor fallback       │
    │  └──────┬──────┘                                   │
    │         ▼                                          │
    │  ┌─────────────┐                                   │
    │  │ 槓桿計算      │  Kelly × 波動率 × 績效 × 熱度     │
    │  │ LeverageEngine│ → 0.25x ~ 2.0x                   │
    │  └──────┬──────┘                                   │
    │         ▼                                          │
    │  ┌─────────────┐                                   │
    │  │ 交易決策      │  止盈/止損/預測賣出/時間平倉      │
    │  │ tick_engine  │  → execute_buy/execute_sell       │
    │  └──────┬──────┘                                   │
    │         ▼                                          │
    │  ┌─────────────┐                                   │
    │  │ 基準更新      │  Benchmark.update()               │
    │  │ Benchmark    │  → 納指權益曲線 + 策略曲線          │
    │  └─────────────┘                                   │
    └────────────────────────────────────────────────────┘
```

## 2. 核心模塊詳解

### 2.1 web_server.py — 主控模塊 (1326行)

**職責**: Flask 伺服器 + 交易引擎 + 全局狀態管理

**關鍵全局變量**:
| 變量 | 類型 | 說明 |
|------|------|------|
| `_current_prices` | dict | 40+1隻股票最新價格 |
| `_previous_prices` | dict | 前次價格（波動率計算） |
| `_iteration_count` | int | 引擎迭代計數 |
| `_portfolio` | PortfolioManager | 持倉管理器 |
| `_predictor` | RealtimePredictor | 統計預測器 |
| `_benchmark` | BenchmarkTracker | 基準對比 |
| `_leverage_engine` | LeverageEngine | 動態槓桿 |
| `_ml_inference` | ModelInference | ML模型推理 |
| `_ml_ready` | bool | ML模型是否就緒 |

**數據流**:
```
tick_engine() 每秒執行:
  1. fetch_yahoo_prices()     → 獲取實時價格
  2. Portfolio.update_prices() → 更新持倉市值
  3. Predictor.update_price()  → 更新統計特徵
  4. Benchmark.update()        → 更新基準曲線
  5. 交易門控檢查             → is_trading_session?
  6. 價格過期檢查             → _price_is_stale?
  7. 交易信號生成             → 止盈/止損/預測賣出
  8. 每60秒 save_state()      → 持久化
```

**REST API 端點** (詳見 [API_REFERENCE.md](API_REFERENCE.md)):
| 端點 | 方法 | 說明 |
|------|------|------|
| `/` | GET | 儀錶盤HTML |
| `/api/health` | GET | 健康檢查 |
| `/api/status` | GET | 完整系統狀態 |
| `/api/tickers` | GET | 追蹤股票列表 |
| `/api/signals` | GET | 交易指令 |
| `/api/kline/<ticker>` | GET | 單只K線數據 |
| `/api/kline/multi` | GET | 批量K線數據 |
| `/api/benchmark_curve` | GET | 基準曲線數據 |
| `/api/backtest_summary` | GET | 回測摘要 |

### 2.2 portfolio.py — 持倉管理 (581行)

**核心類**: `PortfolioManager`, `HoldingPosition`, `TradeRecord`

**關鍵方法**:
- `execute_buy(ticker, qty, price)` — 執行買入
- `execute_sell(ticker, qty, price)` — 執行賣出
- `execute_short(ticker, qty, price)` — 執行做空
- `update_prices(prices_dict)` — 批量更新持倉價格
- `get_total_equity()` → cash + MV - borrowed
- `get_leverage_ratio()` → (MV + borrowed) / equity
- `get_margin_ratio()` → equity / (MV + borrowed)
- `accrue_interest()` — 每日利息計提

**P&L計算**:
```
unrealized_pnl = quantity × (current_price - avg_cost)
unrealized_pnl_pct = (current_price / avg_cost - 1) × side_sign
realized_pnl = Σ(sell_price - buy_price) × quantity - commission
day_pnl = current_equity - day_start_equity
```

### 2.3 predictor.py — 統計預測引擎 (446行)

**核心類**: `RealtimePredictor`

**預測因子** (7個技術指標):
1. **動量** — 5/10/20周期價格變化率
2. **均值回歸** — 價格偏離MA的程度
3. **成交量信號** — 量價背離檢測
4. **波動率** — 短期 vs 長期波動率比值
5. **趨勢強度** — 線性回歸斜率 + R²
6. **RSI** — 超買超賣判斷
7. **MACD** — 金叉死叉信號

**綜合評分**:
```
score = Σ(factor_i × weight_i)  →  [-1, +1]
direction = 1 if score > 0 else -1
confidence = 0.5 + |score| × 0.5
```

詳見 [ALGORITHMS.md](ALGORITHMS.md)。

### 2.4 leverage_engine.py — 動態槓桿 (410行)

**多因子模型**:
```
leverage = Kelly × VolatilityMult × PerfMult × HeatMult
leverage = clamp(leverage, MIN_LEVERAGE, MAX_LEVERAGE)
leverage = min(leverage, DrawdownCap, MarginCap)
```

**四個因子**:
| 因子 | 方法 | 範圍 |
|------|------|------|
| Kelly基礎 | `_kelly_fraction()` | 0.0 ~ 2.0x |
| 波動率調節 | `_volatility_multiplier()` | 0.5 ~ 1.5x |
| 績效反饋 | `_performance_multiplier()` | 0.4 ~ 1.3x |
| 組合熱度 | `_portfolio_heat_multiplier()` | 0.6 ~ 1.0x |

**反馬丁格爾規則**:
- 勝率 > 60% → 加碼 (最多+30%)
- 勝率 < 45% → 減倉 (最多-60%)
- 3連敗 → 強制上限 0.5x

### 2.5 benchmark.py — 基準對比 (409行)

**核心類**: `BenchmarkTracker`

**追蹤指標**:
- 納指權益曲線 vs 策略權益曲線
- 累計收益率、年化收益率
- 夏普比率、最大回撤
- Alpha、Beta、信息比率
- 超額收益

**數據流**:
```
啟動時: fetch_nasdaq_history('6mo') → 納指歷史數據
運行時: update(nasdaq_price, strategy_equity) → 每分鐘追加
讀取時: _ensure_curves_synced() → dict → Series 同步
API:    /api/benchmark_curve → 最近300個數據點
```

### 2.6 market_clock.py — 市場時鐘 (461行)

**市場狀態枚舉**:
- `CLOSED` — 休市 (20:00-04:00 ET)
- `PRE_MARKET` — 盤前 (04:00-09:30 ET)
- `REGULAR_HOURS` — 正常交易 (09:30-16:00 ET)
- `AFTER_HOURS` — 盤後 (16:00-20:00 ET)
- `EARLY_CLOSE` — 早收盤 (如Black Friday 13:00)

**假期支持**: 2025-2027年完整美股假期列表

### 2.7 state_persistence.py — 狀態持久化 (399行)

**保存內容**:
```json
{
  "saved_at": "時間戳",
  "portfolio": {持倉、現金、PnL、槓桿、交易歷史},
  "accuracy": {預測準確率統計},
  "benchmark": {納指曲線、回撤、收益率},
  "globals": {價格、迭代計數、建倉狀態等},
  "predictor": {統計預測器狀態}
}
```

**保存策略**: 每60秒自動保存 + 進程退出時保存

### 2.8 model_inference.py — ML推理

**職責**: 加載 StockTransformer checkpoint，執行推理

**處理流程**:
1. 加載最新 checkpoint (`.pt` 文件)
2. 構建特徵向量 (28 特徵 + 4 情感特徵)
3. 前向傳播 → 方向預測 + 置信度
4. 過濾策略: 特徵數不匹配時只保留交集

### 2.9 前端 dashboard.html (646行)

**5個面板**:
| Tab | ID | 內容 |
|-----|-----|------|
| 📊 總覽 | panel-positions | 持倉列表 + 最近交易 + 系統狀態 |
| 📈 K線圖 | panel-charts | 40隻股票的K線圖 (Lightweight Charts) |
| 📉 分析 | panel-analysis | 策略vs納指曲線 + 收益統計 |
| 🧠 模型 | panel-model | AI模型信息 + 輔助數據源 |
| 📋 交易 | panel-trades | 完整交易記錄 |

**技術細節**:
- 1秒輪詢 `/api/status` 獲取實時數據
- LightweightCharts 4.1.3 渲染K線和曲線
- `safeResize()` 處理面板切換時的圖表尺寸恢復
- 30次重試機制處理容器0寬度問題

## 3. 線程模型

```
主線程 (Flask)
  ├── HTTP請求處理 (threaded=True, 多線程)
  │   ├── GET /api/status      → build_status_data()
  │   ├── GET /api/kline/*     → fetch_kline_data()
  │   └── GET /api/benchmark_curve → _benchmark數據
  │
  └── engine_thread (daemon)
      └── engine_loop()
          └── while _engine_running:
              ├── tick_engine()        # 價格 + 交易 + 基準
              ├── trigger_offline_training()  # 休市時訓練
              └── save_state()         # 狀態持久化
              └── time.sleep(1)
```

**線程安全**: CPython GIL 保護單個操作，但迭代中修改 dict 可能導致 RuntimeError。當前依賴輪詢的原子性，未加鎖。

## 4. 數據存儲

| 路徑 | 格式 | 內容 |
|------|------|------|
| `data/trading_state.json` | JSON | 完整運行時狀態 |
| `models/*.pt` | PyTorch | 訓練好的模型權重 |
| `data/processed/*.parquet` | Parquet | 處理後的訓練數據 |
| `logs/*.log` | 文本 | 運行日誌 |
| `work_logs/*.md` | Markdown | 工作日誌 |

---

> 📖 **下一步**: 閱讀 [ALGORITHMS.md](ALGORITHMS.md) 了解算法細節
