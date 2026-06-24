# 美股量化交易系統 — 完整技術文檔

> **版本**: 2.0 | **最後更新**: 2026-06-21 | **Python**: 3.9+ | **平臺**: macOS / Linux

---

## 目錄

1. [系統概述](#1-系統概述)
2. [文檔導航](#2-文檔導航)
3. [快速開始 (5分鐘)](#3-快速開始)
4. [系統架構概覽](#4-系統架構概覽)
5. [核心功能](#5-核心功能)
6. [技術棧](#6-技術棧)
7. [目錄結構](#7-目錄結構)

---

## 1. 系統概述

本系統是一套**美股量化實盤交易系統**，具備以下能力：

- 🔄 **實時行情抓取**：Yahoo Finance v7 API 批量獲取 40 只美股實時價格
- 🧠 **AI 預測模型**：基於 Transformer 架構的時序預測模型（StockTransformer）
- 📊 **統計預測引擎**：輕量級 RealtimePredictor 作為 ML 模型的 fallback
- ⚖️ **動態槓桿引擎**：多因子 Kelly 公式 + 波動率調節 + 績效反饋
- 📈 **實時儀錶盤**：K線圖、基準對比、持倉監控、交易記錄
- 🛡️ **風險控制**：止盈止損、最大回撤約束、保證金監控、連敗強制減倉
- 💾 **狀態持久化**：每分鐘自動保存，重啟後完整恢復

**核心理念**：反馬丁格爾策略 — 勝率高時加碼，連敗時減倉。

---

## 2. 文檔導航

| 文檔 | 說明 |
|------|------|
| **[README.md](README.md)** | 📖 本文檔 — 系統概述與快速開始 |
| **[ARCHITECTURE.md](ARCHITECTURE.md)** | 🏗️ 系統架構、數據流、模塊交互 |
| **[ALGORITHMS.md](ALGORITHMS.md)** | 🧮 算法詳解：Transformer、Kelly、預測器 |
| **[DEPLOYMENT.md](DEPLOYMENT.md)** | 🚀 一鍵部署、新機配置、launchd 服務 |
| **[setup.sh](setup.sh)** | ⚡ 一鍵環境配置腳本 |
| **[API_REFERENCE.md](API_REFERENCE.md)** | 🔌 全部 REST API 端點文檔 |
| **[CODE_STRUCTURE.md](CODE_STRUCTURE.md)** | 📁 每個文件的職責與依賴 |
| **[CONFIGURATION.md](CONFIGURATION.md)** | ⚙️ 全部配置項詳解 |

---

## 3. 快速開始

### 前置條件

- macOS 或 Linux (推薦 Apple Silicon / x86_64)
- Python 3.9+
- Git

### 3.1 一鍵部署

```bash
# 克隆項目
git clone <repo-url> 美股量化交易
cd 美股量化交易

# 一鍵配置環境
bash docs/setup.sh

# 啟動交易系統
screen -dmS trading python3 -u live_trading/web_server.py

# 驗證
curl http://localhost:8080/api/status
```

### 3.2 手動部署

```bash
# 創建虛擬環境
python3 -m venv .venv
source .venv/bin/activate

# 安裝依賴
pip install -r requirements.txt

# 啟動
screen -dmS trading python3 -u live_trading/web_server.py
```

### 3.3 訪問儀錶盤

瀏覽器打開 `http://localhost:8080`

---

## 4. 系統架構概覽

```
┌─────────────────────────────────────────────────────────┐
│                    Web 儀錶盤 (Flask)                     │
│  localhost:8080  →  dashboard.html  →  實時輪詢 /api/*   │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────┼──────────────────────────────────┐
│              live_trading/web_server.py                  │
│  ┌─────────────┐  ┌──────────┐  ┌───────────────────┐  │
│  │ tick_engine  │  │ REST API │  │ state_persistence │  │
│  │ (每秒1次)    │  │ (8端點)  │  │ (每分鐘保存)      │  │
│  └──────┬───────┘  └──────────┘  └───────────────────┘  │
│         │                                                │
│  ┌──────┼──────────────────────────────────────────┐    │
│  │      ▼ 數據流                                    │    │
│  │  Yahoo Finance ─→ 價格 ─→ 持倉更新 ─→ 交易決策  │    │
│  │                      │                           │    │
│  │                      ▼                           │    │
│  │               ML模型預測 ─→ 槓桿計算 ─→ 執行     │    │
│  └─────────────────────────────────────────────────┘    │
├─────────────────────────────────────────────────────────┤
│  Portfolio │ Predictor │ LeverageEngine │ Benchmark     │
│  MarketClock │ AccuracyTracker │ ModelInference        │
└─────────────────────────────────────────────────────────┘
```

---

## 5. 核心功能

### 實時交易循環 (每秒)

1. **價格更新** — Yahoo Finance v7 批量 API (12-60秒間隔)
2. **持倉估值** — 計算市值、PnL、權重
3. **ML 預測** — StockTransformer 輸出方向 + 置信度
4. **槓桿計算** — Kelly × 波動率 × 績效 × 熱度 × 回撤約束
5. **交易決策** — 止盈/止損/預測賣出/時間平倉
6. **狀態保存** — 每60秒持久化到 `data/trading_state.json`

### 風險控制體系

| 層級 | 機制 | 參數 |
|------|------|------|
| 倉位 | 單只最大佔比 | 8% (槓桿模式12%) |
| 止損 | 硬止損閾值 | -4% |
| 止盈 | 獲利了結 | +3% |
| 槓桿 | Kelly動態 | 0.25x-2.0x |
| 回撤 | 整體回撤約束 | 動態cap |
| 連敗 | 反馬丁格爾 | 3連敗→0.5x上限 |

---

## 6. 技術棧

| 層級 | 技術 |
|------|------|
| 語言 | Python 3.9+ |
| Web框架 | Flask (threaded) |
| 深度學習 | PyTorch 2.0+ |
| 數據處理 | NumPy, Pandas, SciPy |
| 數據源 | Yahoo Finance (yfinance) |
| 存儲 | JSON (狀態), Parquet (訓練數據) |
| 前端 | Vanilla JS + Lightweight Charts |
| 進程管理 | screen / launchd |
| 系統監控 | psutil |

---

## 7. 目錄結構

```
美股量化交易（New）/
├── live_trading/           # 核心交易系統
│   ├── web_server.py       # Flask伺服器 + 交易引擎 (1326行)
│   ├── portfolio.py        # 持倉管理 + P&L計算 (581行)
│   ├── predictor.py        # 統計預測引擎 (446行)
│   ├── benchmark.py        # 納指基準對比 (409行)
│   ├── leverage_engine.py  # 動態槓桿引擎 (410行)
│   ├── market_clock.py     # 美股交易時鐘 (461行)
│   ├── state_persistence.py # 狀態持久化 (399行)
│   ├── accuracy_tracker.py # 預測準確率追蹤
│   ├── model_inference.py  # ML模型推理
│   ├── templates/
│   │   └── dashboard.html  # Web儀錶盤 (646行)
│   └── static/             # 靜態資源
├── ml_model/               # ML訓練
│   ├── transformer.py      # StockTransformer模型 (739行)
│   ├── trainer.py          # 訓練器 (1017行)
│   └── data_loader.py      # 數據加載器
├── config/
│   └── settings.py         # 全局配置 (399行)
├── backtesting/            # 回測引擎
├── data_pipeline/          # 數據管道
├── crawler/                # 新聞爬蟲
├── scripts/                # 工具腳本
│   ├── quick_train.py      # 快速訓練
│   ├── train_100ep_robust.py # 100輪訓練
│   ├── build_sentiment_features.py # 情感特徵構建
│   └── codex_monitor.py    # 系統監控
├── docs/                   # 📖 技術文檔
├── work_logs/              # 工作日誌
├── data/                   # 運行時數據
├── models/                 # 訓練好的模型
├── logs/                   # 運行日誌
├── requirements.txt        # Python依賴
├── AGENTS.md               # AI工作指引
└── GUARDRAILS.md           # 錯誤清單 + 禁區
```

---

> 📖 **下一步**: 閱讀 [ARCHITECTURE.md](ARCHITECTURE.md) 了解系統架構細節
