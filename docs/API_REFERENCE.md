# API 參考文檔

## 基礎信息

- **Base URL**: `http://localhost:8080`
- **協議**: HTTP/1.1
- **格式**: JSON (Content-Type: application/json)
- **字符編碼**: UTF-8

---

## 端點列表

### 1. 健康檢查

```
GET /api/health
```

**響應**:
```json
{
  "status": "ok",
  "timestamp": "2026-06-21 14:30:00"
}
```

---

### 2. 完整系統狀態

```
GET /api/status
```

**響應結構**:
```json
{
  "timestamp": "2026-06-21 14:30:00",
  "iteration": 9001,
  "waiting_for_open": false,
  "positions_initialized": true,
  "ml_ready": true,

  "market": {
    "status": "REGULAR_HOURS",
    "status_desc": "正常交易",
    "is_open": true,
    "is_active": true,
    "countdown": "距閉市 2h 30m 15s",
    "current_et": "14:30:00",
    "next_trading_day": "2026-06-23"
  },

  "account": {
    "initial_capital": 100000.0,
    "cash": 20000.0,
    "total_equity": 105000.0,
    "total_market_value": 85000.0,
    "position_count": 8,
    "borrowed": 0.0,
    "margin_ratio": 100.0,
    "leverage": 1.0,
    "leverage_ratio": 0.85,
    "margin_call_risk": false,
    "total_interest": 0.0
  },

  "pnl": {
    "net_pnl": 5000.0,
    "net_pnl_pct": 5.0,
    "realized_pnl": 1200.0,
    "unrealized_pnl": 3800.0,
    "total_commission": 45.0,
    "day_pnl": 500.0,
    "day_pnl_pct": 0.48,
    "max_drawdown_pct": -2.5
  },

  "positions": [
    {
      "ticker": "AAPL",
      "quantity": 50,
      "avg_cost": 210.0,
      "current_price": 220.0,
      "market_value": 11000.0,
      "cost_basis": 10500.0,
      "unrealized_pnl": 500.0,
      "unrealized_pnl_pct": 4.76,
      "day_change_pct": 1.5,
      "weight": 10.5
    }
  ],

  "accuracy": {
    "direction_accuracy": 55.0,
    "recent_50_accuracy": 58.0,
    "rmse": 0.0234,
    "mae": 0.0185,
    "total_predictions": 120,
    "confirmed_predictions": 100,
    "correct_long": 35,
    "total_long": 60,
    "correct_short": 25,
    "total_short": 40,
    "is_acceptable": true,
    "trend": "improving"
  },

  "benchmark": {
    "nasdaq_price": 18500.0,
    "strategy_return_pct": 5.0,
    "nasdaq_return_pct": 2.5,
    "excess_return": 2.5,
    "strategy_annual_return": 30.0,
    "nasdaq_annual_return": 15.0,
    "strategy_sharpe": 1.5,
    "nasdaq_sharpe": 0.8,
    "strategy_max_drawdown": -2.5,
    "nasdaq_max_drawdown": -3.0,
    "alpha": 0.0012,
    "beta": 0.85,
    "information_ratio": 0.9,
    "outperformance_pct": 2.5
  },

  "model_info": {
    "feature_count": 28,
    "sentiment_features": ["news_sentiment_3d", "news_sentiment_7d", ...],
    "has_sentiment": true,
    "d_model": 192,
    "model_size_mb": 6.0
  },

  "pending_signals": [...],
  "recent_trades": [...],

  "data_quality": {
    "source": "Yahoo Finance (實時)",
    "last_update": "2026-06-21 14:30:00",
    "latency_ms": 120.5,
    "update_count": 500,
    "is_real_time": true,
    "data_age_s": 2,
    "is_stale": false
  }
}
```

---

### 3. 追蹤股票列表

```
GET /api/tickers
```

**響應**:
```json
{
  "tickers": ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", ...],
  "count": 40
}
```

**40隻追蹤股票**: 科技七巨頭 + 軟體/SaaS + 金融 + 消費 + 晶片/半導體 + 光通信 + 存儲 + 數據中心 + 半導體設備 + EDA

---

### 4. 交易指令

```
GET /api/signals
```

**響應**:
```json
{
  "signals_md": "*[BUY] AAPL 50股 @ $210.00 — 開盤建倉*  \n...",
  "signals": [
    {
      "time": "14:30:00",
      "action": "BUY",
      "ticker": "AAPL",
      "qty": 50,
      "price": 210.0,
      "reason": "開盤建倉(Yahoo價格)"
    }
  ]
}
```

---

### 5. K線數據 (單只)

```
GET /api/kline/<ticker>
```

**參數**:
| 參數 | 類型 | 默認值 | 說明 |
|------|------|--------|------|
| `period` | string | `1d` | 時間範圍 (1d/5d/1mo/3mo) |
| `interval` | string | `5m` | K線周期 (1m/5m/15m/30m/1h/1d) |

**示例**:
```
GET /api/kline/AAPL?period=5d&interval=15m
```

**響應**:
```json
{
  "ticker": "AAPL",
  "candles": [
    {
      "time": 1781271000,
      "open": 296.08,
      "high": 297.14,
      "low": 291.88,
      "close": 292.39,
      "volume": 3957276
    }
  ],
  "count": 130
}
```

**time欄位**: Unix 時間戳 (秒)，LightweightCharts 直接兼容。

---

### 6. K線數據 (批量)

```
GET /api/kline/multi
```

**參數**:
| 參數 | 類型 | 默認值 | 說明 |
|------|------|--------|------|
| `tickers` | string | `AAPL,NVDA,MSFT,GOOGL` | 逗號分隔的股票代碼 |
| `period` | string | `1d` | 時間範圍 |
| `interval` | string | `15m` | K線周期 |

**限制**: 最多返回6隻股票的數據。

**示例**:
```
GET /api/kline/multi?tickers=AAPL,NVDA,MSFT&period=5d&interval=15m
```

**響應**:
```json
{
  "data": {
    "AAPL": [{...}, ...],
    "NVDA": [{...}, ...],
    "MSFT": [{...}, ...]
  },
  "count": 3
}
```

---

### 7. 基準曲線

```
GET /api/benchmark_curve
```

**響應**:
```json
{
  "points": [
    {
      "time": 1766102400,
      "nasdaq": 0.0,
      "strategy": 0.0
    }
  ],
  "count": 124,
  "nasdaq_current": 13.7737,
  "strategy_current": 0.0,
  "initial_capital": 100000.0
}
```

**time欄位**: Unix 時間戳 (秒)
**nasdaq/strategy**: 百分比收益率 (如 13.77 表示 +13.77%)

---

### 8. 回測摘要

```
GET /api/backtest_summary
```

**響應**:
```json
{
  "available": false,
  "message": "運行 python main.py backtest 生成回測結果",
  "summary": {}
}
```

---

### 9. 儀錶盤 (HTML)

```
GET /
```

返回完整的 Web 儀錶盤 HTML 頁面。

---

## 錯誤處理

所有端點返回標準 JSON 錯誤：

```json
{
  "error": "錯誤描述",
  "points": [],
  "count": 0
}
```

HTTP 狀態碼:
- `200` — 成功
- `500` — 伺服器內部錯誤

---

## 輪詢建議

前端以 1 秒間隔輪詢 `/api/status`，渲染所有面板數據。

```javascript
setInterval(async () => {
  const data = await fetch('/api/status').then(r => r.json());
  render(data);  // 更新所有面板
}, 1000);
```

---

> 📖 **下一步**: 閱讀 [CODE_STRUCTURE.md](CODE_STRUCTURE.md) 了解代碼組織
