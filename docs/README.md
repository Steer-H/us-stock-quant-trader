# 美股量化交易系统 — 完整技术文档

> **版本**: 2.0 | **最后更新**: 2026-06-21 | **Python**: 3.9+ | **平台**: macOS / Linux

---

## 目录

1. [系统概述](#1-系统概述)
2. [文档导航](#2-文档导航)
3. [快速开始 (5分钟)](#3-快速开始)
4. [系统架构概览](#4-系统架构概览)
5. [核心功能](#5-核心功能)
6. [技术栈](#6-技术栈)
7. [目录结构](#7-目录结构)

---

## 1. 系统概述

本系统是一套**美股量化实盘交易系统**，具备以下能力：

- 🔄 **实时行情抓取**：Yahoo Finance v7 API 批量获取 40 只美股实时价格
- 🧠 **AI 预测模型**：基于 Transformer 架构的时序预测模型（StockTransformer）
- 📊 **统计预测引擎**：轻量级 RealtimePredictor 作为 ML 模型的 fallback
- ⚖️ **动态杠杆引擎**：多因子 Kelly 公式 + 波动率调节 + 绩效反馈
- 📈 **实时仪表盘**：K线图、基准对比、持仓监控、交易记录
- 🛡️ **风险控制**：止盈止损、最大回撤约束、保证金监控、连败强制减仓
- 💾 **状态持久化**：每分钟自动保存，重启后完整恢复

**核心理念**：反马丁格尔策略 — 胜率高时加码，连败时减仓。

---

## 2. 文档导航

| 文档 | 说明 |
|------|------|
| **[README.md](README.md)** | 📖 本文档 — 系统概述与快速开始 |
| **[ARCHITECTURE.md](ARCHITECTURE.md)** | 🏗️ 系统架构、数据流、模块交互 |
| **[ALGORITHMS.md](ALGORITHMS.md)** | 🧮 算法详解：Transformer、Kelly、预测器 |
| **[DEPLOYMENT.md](DEPLOYMENT.md)** | 🚀 一键部署、新机配置、launchd 服务 |
| **[setup.sh](setup.sh)** | ⚡ 一键环境配置脚本 |
| **[API_REFERENCE.md](API_REFERENCE.md)** | 🔌 全部 REST API 端点文档 |
| **[CODE_STRUCTURE.md](CODE_STRUCTURE.md)** | 📁 每个文件的职责与依赖 |
| **[CONFIGURATION.md](CONFIGURATION.md)** | ⚙️ 全部配置项详解 |

---

## 3. 快速开始

### 前置条件

- macOS 或 Linux (推荐 Apple Silicon / x86_64)
- Python 3.9+
- Git

### 3.1 一键部署

```bash
# 克隆项目
git clone <repo-url> 美股量化交易
cd 美股量化交易

# 一键配置环境
bash docs/setup.sh

# 启动交易系统
screen -dmS trading python3 -u live_trading/web_server.py

# 验证
curl http://localhost:8080/api/status
```

### 3.2 手动部署

```bash
# 创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 启动
screen -dmS trading python3 -u live_trading/web_server.py
```

### 3.3 访问仪表盘

浏览器打开 `http://localhost:8080`

---

## 4. 系统架构概览

```
┌─────────────────────────────────────────────────────────┐
│                    Web 仪表盘 (Flask)                     │
│  localhost:8080  →  dashboard.html  →  实时轮询 /api/*   │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────┼──────────────────────────────────┐
│              live_trading/web_server.py                  │
│  ┌─────────────┐  ┌──────────┐  ┌───────────────────┐  │
│  │ tick_engine  │  │ REST API │  │ state_persistence │  │
│  │ (每秒1次)    │  │ (8端点)  │  │ (每分钟保存)      │  │
│  └──────┬───────┘  └──────────┘  └───────────────────┘  │
│         │                                                │
│  ┌──────┼──────────────────────────────────────────┐    │
│  │      ▼ 数据流                                    │    │
│  │  Yahoo Finance ─→ 价格 ─→ 持仓更新 ─→ 交易决策  │    │
│  │                      │                           │    │
│  │                      ▼                           │    │
│  │               ML模型预测 ─→ 杠杆计算 ─→ 执行     │    │
│  └─────────────────────────────────────────────────┘    │
├─────────────────────────────────────────────────────────┤
│  Portfolio │ Predictor │ LeverageEngine │ Benchmark     │
│  MarketClock │ AccuracyTracker │ ModelInference        │
└─────────────────────────────────────────────────────────┘
```

---

## 5. 核心功能

### 实时交易循环 (每秒)

1. **价格更新** — Yahoo Finance v7 批量 API (12-60秒间隔)
2. **持仓估值** — 计算市值、PnL、权重
3. **ML 预测** — StockTransformer 输出方向 + 置信度
4. **杠杆计算** — Kelly × 波动率 × 绩效 × 热度 × 回撤约束
5. **交易决策** — 止盈/止损/预测卖出/时间平仓
6. **状态保存** — 每60秒持久化到 `data/trading_state.json`

### 风险控制体系

| 层级 | 机制 | 参数 |
|------|------|------|
| 仓位 | 单只最大占比 | 8% (杠杆模式12%) |
| 止损 | 硬止损阈值 | -4% |
| 止盈 | 获利了结 | +3% |
| 杠杆 | Kelly动态 | 0.25x-2.0x |
| 回撤 | 整体回撤约束 | 动态cap |
| 连败 | 反马丁格尔 | 3连败→0.5x上限 |

---

## 6. 技术栈

| 层级 | 技术 |
|------|------|
| 语言 | Python 3.9+ |
| Web框架 | Flask (threaded) |
| 深度学习 | PyTorch 2.0+ |
| 数据处理 | NumPy, Pandas, SciPy |
| 数据源 | Yahoo Finance (yfinance) |
| 存储 | JSON (状态), Parquet (训练数据) |
| 前端 | Vanilla JS + Lightweight Charts |
| 进程管理 | screen / launchd |
| 系统监控 | psutil |

---

## 7. 目录结构

```
美股量化交易（New）/
├── live_trading/           # 核心交易系统
│   ├── web_server.py       # Flask服务器 + 交易引擎 (1326行)
│   ├── portfolio.py        # 持仓管理 + P&L计算 (581行)
│   ├── predictor.py        # 统计预测引擎 (446行)
│   ├── benchmark.py        # 纳指基准对比 (409行)
│   ├── leverage_engine.py  # 动态杠杆引擎 (410行)
│   ├── market_clock.py     # 美股交易时钟 (461行)
│   ├── state_persistence.py # 状态持久化 (399行)
│   ├── accuracy_tracker.py # 预测准确率追踪
│   ├── model_inference.py  # ML模型推理
│   ├── templates/
│   │   └── dashboard.html  # Web仪表盘 (646行)
│   └── static/             # 静态资源
├── ml_model/               # ML训练
│   ├── transformer.py      # StockTransformer模型 (739行)
│   ├── trainer.py          # 训练器 (1017行)
│   └── data_loader.py      # 数据加载器
├── config/
│   └── settings.py         # 全局配置 (399行)
├── backtesting/            # 回测引擎
├── data_pipeline/          # 数据管道
├── crawler/                # 新闻爬虫
├── scripts/                # 工具脚本
│   ├── quick_train.py      # 快速训练
│   ├── train_100ep_robust.py # 100轮训练
│   ├── build_sentiment_features.py # 情感特征构建
│   └── codex_monitor.py    # 系统监控
├── docs/                   # 📖 技术文档
├── work_logs/              # 工作日志
├── data/                   # 运行时数据
├── models/                 # 训练好的模型
├── logs/                   # 运行日志
├── requirements.txt        # Python依赖
├── AGENTS.md               # AI工作指引
└── GUARDRAILS.md           # 错误清单 + 禁区
```

---

> 📖 **下一步**: 阅读 [ARCHITECTURE.md](ARCHITECTURE.md) 了解系统架构细节
