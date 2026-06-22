# 美股量化交易系统 v2.0

> 基于 Transformer 深度学习 + 动态杠杆引擎的 40 股美股量化交易系统  
> 最后更新: 2026-06-17 | 版本: v2.0 (40-stock + Dynamic Leverage)

---

## 目录

1. [系统概览](#1-系统概览)
2. [核心架构](#2-核心架构)
3. [运作原理](#3-运作原理)
4. [文件结构](#4-文件结构)
5. [环境要求](#5-环境要求)
6. [快速部署](#6-快速部署)
7. [配置说明](#7-配置说明)
8. [仪表盘使用](#8-仪表盘使用)
9. [训练新模型](#9-训练新模型)
10. [故障排查](#10-故障排查)
11. [与旧版区别](#11-与旧版区别)

---

## 1. 系统概览

本系统对美国股市 40 只股票进行实时量化交易（模拟），核心由三层组成：

```
┌─────────────────────────────────────────────────┐
│              浏览器仪表盘 (:8080)                  │
│         实时K线 · 持仓 · 盈亏 · 杠杆              │
└──────────────────┬──────────────────────────────┘
                   │ HTTP/轮询 (1s)
┌──────────────────▼──────────────────────────────┐
│           Flask Web 服务器 (web_server.py)        │
│   tick_engine 每秒运行：价格更新 → 交易决策        │
│   ┌──────────┬──────────┬──────────┬──────────┐  │
│   │ 止盈止损 │ ML预测   │ 超时平仓  │ 动态杠杆  │  │
│   └──────────┴──────────┴──────────┴──────────┘  │
└──────┬──────────────┬──────────────┬─────────────┘
       │              │              │
┌──────▼──────┐ ┌─────▼──────┐ ┌────▼─────────────┐
│ Yahoo Finance│ │ Transformer │ │  Leverage Engine │
│  实时价格     │ │  ML 推理    │ │  凯利+波动率+绩效 │
│  30s/次      │ │  40股预测   │ │  四因子动态杠杆   │
└─────────────┘ └────────────┘ └──────────────────┘
```

**关键数字**:
- 40 只美股（科技/芯片/光模块/存储/金融/消费/EDA）
- 100% 真实 Yahoo Finance 数据（非模拟）
- Transformer 模型 24 特征 × 256 维度
- 动态杠杆 0.25x ~ 2.0x（凯利公式 + 三因子调节）

---

## 2. 核心架构

### 2.1 交易引擎 (tick_engine)

每秒运行一次的核心循环：

```
tick_engine() 每秒:
  ├── Yahoo Finance 抓取实时价格 (30s间隔)
  ├── 开盘检测 → 自动建仓 40 只股票
  ├── 更新持仓市值 & 基准
  ├── 利息结算 (杠杆借款日息)
  │
  ├── [仅交易时段] 每60秒:
  │   ├── 卖出检查: 止盈(+3%) 止损(-2.5%/-4%) ML预测 超时(30min)
  │   └── 买入检查: ML信号筛选 → 动态杠杆计算 → 开仓
  │
  ├── 每30秒: 统计预测 & 准确率追踪
  └── 每60秒: 自动保存状态
```

### 2.2 ML 模型

| 属性 | 值 |
|------|-----|
| 架构 | StockTransformer (Pre-LN Encoder-only) |
| 特征 | 24个技术指标 (SMA/EMA/RSI/MACD/Bollinger/ATR等) |
| 维度 | d_model=256, 4层, 8头 |
| 输入 | 60天历史特征窗口 |
| 输出 | 5日方向(涨跌) + 5日收益率 |
| 数据 | 40只股票, 113,182训练样本 |
| 设备 | Apple Silicon MPS (GPU加速) |

### 2.3 动态杠杆引擎

替代传统静态杠杆映射，基于四因子实时计算：

```
杠杆 = 凯利基础 × 波动率乘数 × 绩效乘数 × 热度乘数
     → clamp(0.25x ~ 2.0x) → min(熔断上限)
```

| 因子 | 说明 | 示例 |
|------|------|------|
| 凯利公式 | f* = (p×b−q)/b, Half-Kelly | p=0.72 → 1.56x |
| 波动率 | 年化波动率估算 | 高波动→0.6x, 极端→0.25x |
| 绩效反馈 | 近20笔胜率 | 连胜+30%, 连败-60% |
| 组合热度 | 持仓/容量比 | 满仓→杠杆打4折 |

**五级渐进熔断**: 回撤 5%→1.5x, 10%→1.0x, 15%→0.5x, 20%→停仓

### 2.4 40 只股票

| 类别 | 股票 |
|------|------|
| 科技七巨头 | AAPL MSFT NVDA GOOGL AMZN META TSLA |
| 软件/SaaS | NFLX ADBE CRM NOW ORCL |
| 金融 | JPM V MA BAC |
| 消费 | WMT HD NKE SBUX UBER |
| 芯片/半导体 | AVGO AMD INTC QCOM TXN |
| 光模块/光通信 | AAOI COHR LITE FN |
| 存储 | WDC STX NTAP |
| 数据中心芯片 | MRVL MU |
| 半导体设备 | LRCX AMAT KLAC |
| EDA软件 | SNPS CDNS |

---

## 3. 运作原理

### 3.1 开盘自动建仓

当美东时间 9:30，系统检测到 `REGULAR_HOURS`：
1. 从 Yahoo Finance 批量获取 40 只股票实时价格
2. 每只分配约 $2,000（总资金 $100,000 的 80%）
3. 按实时价格计算股数，扣除佣金后建仓
4. 建仓完成后开始交易循环

### 3.2 买卖决策

**卖出条件**（任一触发）:
1. 止盈: 盈利 ≥ 3%，80% 概率卖出（确定性哈希）
2. 止损: 杠杆持仓 -2.5%，普通 -4%，90% 概率卖出
3. ML 预测: Transformer 预测下跌 + 置信度 > 55%
4. 超时: 持仓超过 30 分钟强制平仓

**买入条件**:
1. 现金 > $5,000
2. 候选股票未持仓且不在冷却期
3. ML 模型预测上涨
4. 动态杠杆引擎计算杠杆 ≥ 0.5x（机会足够好）

### 3.3 风控体系

| 层级 | 机制 |
|------|------|
| 仓位 | 单股 ≤ 8% 总权益，杠杆 ≤ 12% |
| 止损 | 杠杆 -2.5%，普通 -4% |
| 回撤 | 5级渐进熔断 (5%/10%/15%/20%) |
| 保证金 | > 60%→1.5x, > 80%→1.0x, > 90%→停仓 |
| 杠杆借款 | 年化 5% 利息，最大 2x |
| 状态持久化 | 每 60 秒自动保存，崩溃可恢复 |

---

## 4. 文件结构

```
美股量化交易（New）/
├── main.py                    # 入口：web/backtest/crawl/train
├── requirements.txt           # Python 依赖
├── README.md                  # 本文档
│
├── config/                    # 配置
│   └── settings.py            # 模型/交易/数据源配置
│
├── live_trading/              # ★ 核心交易系统
│   ├── web_server.py          # Flask 服务器 + tick_engine
│   ├── portfolio.py           # 持仓管理 + 杠杆借款
│   ├── model_inference.py     # Transformer 实时推理
│   ├── predictor.py           # 统计预测器（ML后备）
│   ├── leverage_engine.py     # ★ 动态杠杆引擎 (v2.0新增)
│   ├── market_clock.py        # 美股交易时钟
│   ├── benchmark.py           # 基准对比（vs 纳指）
│   ├── accuracy_tracker.py    # 预测准确率追踪
│   ├── state_persistence.py   # 状态保存/恢复
│   └── templates/
│       └── dashboard.html     # Web 仪表盘前端
│
├── ml_model/                  # ML 模型
│   ├── transformer.py         # StockTransformer 架构
│   ├── trainer.py             # 训练器 + 超参调优
│   └── data_loader.py         # 数据加载 + 标准化
│
├── data_pipeline/             # 数据管道
│   ├── fetcher.py             # Yahoo Finance 抓取
│   ├── cleaner.py             # 数据清洗
│   ├── indicators.py          # 技术指标计算
│   └── storage.py             # Parquet 存储
│
├── risk/                      # 风控
│   └── manager.py             # 风险检查
│
├── crawler/                   # 爬虫
│   └── stock_crawler.py       # S&P 500 批量爬取
│
├── backtesting/               # 回测
│   ├── engine.py              # 回测引擎
│   └── performance.py         # 绩效分析
│
├── data/                      # 数据目录
│   ├── processed/             # 40只股票特征数据 (*.parquet)
│   ├── models/                # 训练好的模型 (*.pt)
│   └── trading_state.json     # 交易状态持久化
│
├── logs/                      # 运行日志
└── work_logs/                 # 开发文档/变更日志
```

---

## 5. 环境要求

| 组件 | 最低要求 | 推荐 |
|------|---------|------|
| Python | 3.9+ | 3.11+ |
| PyTorch | 2.0+ | 2.3+ (MPS支持) |
| macOS | 12.0+ | 14.0+ (Apple Silicon) |
| 内存 | 8 GB | 16 GB |
| 网络 | 可访问 Yahoo Finance | 稳定连接 |
| 浏览器 | Chrome/Safari/Firefox | 最新版 |

**Python 依赖** (`requirements.txt`):
```
flask, yfinance, pandas, numpy, torch, scikit-learn, 
lightweight-charts (CDN), requests, beautifulsoup4, pyarrow
```

---

## 6. 快速部署

### 6.1 一键启动

```bash
cd /Users/oujianli/Documents/美股量化交易（New）

# 安装依赖
pip3 install -r requirements.txt

# 启动交易服务器（screen 后台运行）
screen -dmS trading python3 -u live_trading/web_server.py

# 打开浏览器
open http://localhost:8080
```

### 6.2 训练模型（首次或数据更新后）

```bash
# 40只股票完整训练（MPS加速，约90分钟）
python3 << 'PYEOF'
import sys; sys.path.insert(0, '.')
from ml_model.trainer import ModelTrainer
from ml_model.data_loader import prepare_data
from config.settings import model_config

train_loader, val_loader, test_loader, scaler = prepare_data(config=model_config)
trainer = ModelTrainer(model_config)
trainer.train(train_loader, val_loader, epochs=30)
result = trainer.evaluate(test_loader)
print(f"方向准确率: {result.direction_accuracy:.2%}")
trainer.save_model('transformer_stock_latest')
PYEOF
```

### 6.3 爬取历史数据（如需要）

```bash
python3 main.py crawl
```

### 6.4 重启服务器

```bash
# 杀掉旧进程
screen -S trading -X quit
pkill -f web_server.py

# 等待端口释放
sleep 3

# 启动新服务
screen -dmS trading python3 -u live_trading/web_server.py

# 验证
curl http://localhost:8080/api/health
```

---

## 7. 配置说明

核心配置在 `config/settings.py`：

### 模型配置 (`ModelConfig`)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| d_model | 256 | Transformer 隐藏维度 |
| n_heads | 8 | 注意力头数 |
| n_encoder_layers | 4 | Encoder 层数 |
| dropout | 0.15 | Dropout 率 |
| lookback_window | 60 | 输入窗口(天) |
| prediction_horizon | 5 | 预测窗口(天) |
| learning_rate | 1e-4 | 学习率 |
| batch_size | 64 | 批大小 |
| min_direction_accuracy | 0.55 | 最低方向准确率 |
| max_rmse | 0.06 | 最大 RMSE |

### 交易配置 (`TradingConfig`)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| initial_capital | 100,000 | 初始资金(美元) |
| commission_per_share | 0.005 | 每股佣金 |
| commission_min | 1.0 | 最低佣金 |
| max_leverage | 2.0 | 最大杠杆 |
| max_position_pct | 0.20 | 单股最大仓位 |
| max_drawdown_pct | 0.25 | 最大回撤(触及停交易) |

### 杠杆引擎常量 (`leverage_engine.py`)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| MAX_LEVERAGE | 2.0 | 全局杠杆上限 |
| MIN_LEVERAGE | 0.25 | 全局杠杆下限 |
| HALF_KELLY_FRACTION | 0.5 | Half-Kelly 系数 |
| PERF_WINDOW | 20 | 绩效追踪窗口(笔) |
| DRAWDOWN_CAPS | 5级 | 回撤熔断阈值 |

---

## 8. 仪表盘使用

访问 `http://localhost:8080` 打开仪表盘，包含四个标签页：

### 📊 概览 (Overview)
- 账户总览：总权益、现金、市值、净盈亏
- 杠杆状态：当前杠杆、历史平均、保证金率、借款金额
- 市场状态：开盘/盘前/盘后/闭市，倒计时

### 📈 持仓 (Positions)
- 40只股票实时持仓表（盈亏、权重、日变动）
- 最近10笔交易记录
- 交易信号日志

### 📉 K线图 (Charts)
- TradingView 专业K线图（支持 1m/5m/15m/1h/1d）
- 40只股票切换按钮，按行业分组
- 成交量副图

### 📊 分析 (Analysis)
- 策略 vs 纳指对比曲线
- Alpha/Beta/夏普/信息比率
- 预测准确率（方向 + RMSE）
- 数据质量（来源、延迟、更新次数）
- ML 模型状态

---

## 9. 训练新模型

当数据更新或需要改进模型时：

### 完整训练流程

```bash
cd /Users/oujianli/Documents/美股量化交易（New）

# 1. 爬取最新数据（可选）
python3 main.py crawl

# 2. 处理数据（计算技术指标）
python3 -c "
from data_pipeline.storage import ParquetStorage
from data_pipeline.indicators import compute_all_indicators
# ... 处理逻辑见 data_pipeline/
"

# 3. 训练模型
python3 << 'PYEOF'
import sys; sys.path.insert(0, '.')
from ml_model.trainer import ModelTrainer
from ml_model.data_loader import prepare_data
from config.settings import model_config

train_loader, val_loader, test_loader, scaler = prepare_data(config=model_config)
trainer = ModelTrainer(model_config)
trainer.train(train_loader, val_loader, epochs=30)
result = trainer.evaluate(test_loader)
trainer.save_model('transformer_stock_latest')
PYEOF

# 4. 重启服务器加载新模型
screen -S trading -X quit; pkill -f web_server.py; sleep 3
screen -dmS trading python3 -u live_trading/web_server.py
```

### 监控训练

```bash
tail -f logs/train_40stocks.log
```

---

## 10. 故障排查

### 服务器不响应

```bash
# 检查进程
pgrep -fl web_server.py
lsof -i :8080

# 查看错误日志
tail -50 logs/launchd_stderr.log

# 重启
screen -S trading -X quit; pkill -f web_server.py
sleep 3; screen -dmS trading python3 -u live_trading/web_server.py
```

### ML 模型未加载 (ml_ready=False)

```bash
# 检查模型文件
ls -la data/models/transformer_stock_latest.pt

# 手动测试加载
python3 -c "
import sys; sys.path.insert(0,'.')
from live_trading.model_inference import ModelInference
mi = ModelInference()
print('Load:', mi.load())
print('Ready:', mi.is_ready())
"
```

### Yahoo Finance 无数据

- Yahoo Finance 对中国大陆 IP 有限制，需代理/VPN
- 系统会自动降级：3次失败后标记数据过期，暂停交易
- 检查：`curl -s http://localhost:8080/api/status | python3 -c "import sys,json;d=json.load(sys.stdin);print(d['data_quality'])"`

### 数据显示过期 (Stale=True)

- 等待 30 秒后 Yahoo 自动重试
- 连续 3 次失败才会标记过期
- 过期期间系统暂停交易，仅显示价格

### 端口被占用

```bash
lsof -ti :8080 | xargs kill -9
```

---

## 11. 与旧版区别

### v2.0 vs v1.0 重大变更

| 维度 | v1.0 (旧版) | v2.0 (当前) |
|------|------------|------------|
| 股票数量 | 8 → 20 → 26 | **40** (芯片/光模块/存储/EDA全覆盖) |
| 杠杆系统 | 无 | **凯利四因子动态杠杆** (0.25x~2.0x) |
| 数据新鲜度 | 简单二值 stale | **多级判定 + 数据年龄追踪** |
| 模型推理 | 模拟随机 | **Transformer 实时推理** (40股全预测) |
| 止损 | -8% 单一 | **-2.5%杠杆 / -4%普通 分级** |
| 回撤保护 | 无 | **五级渐进熔断** (5%/10%/15%/20%) |
| 前端 | 20只K线按钮 | **40只按行业分组** |
| 后端稳定性 | Werkzeug kqueue 报错 | **threaded=True 修复** |
| 状态持久化 | 部分 | **完整(含杠杆引擎/预测器)** |
| 数据过期误报 | 经常误报 | **仅连续失败3次才标记** |

### 新增文件

- `live_trading/leverage_engine.py` — 动态杠杆引擎 (410行)
- `work_logs/2026-06-17_pre-market-final-check.md` — 开盘前最终检查
- `work_logs/2026-06-17_comprehensive-audit-v2.md` — 全面审计报告

### 训练历史

| 版本 | 数据 | 股票数 | 准确率 | RMSE |
|------|------|--------|--------|------|
| v1 合成 | GBM合成 | 8 | 61.72% | 0.053 |
| v3 锚点 | 锚点约束合成 | 26 | 67.26% | 0.037 |
| v4 真实 | Yahoo真实 | 40 | 52.94% | 0.056 |

> 注: v4 使用真实 Yahoo 数据，准确率略低但更接近实战。系统有多层安全网（杠杆引擎 + 止损 + 熔断）弥补模型精度。

---

## 许可证与声明

本项目仅用于个人学习和研究目的。不构成投资建议。美股交易有风险，实盘请谨慎。

---

> **维护者**: Codex AI  
> **项目路径**: `/Users/oujianli/Documents/美股量化交易（New）`  
> **最后更新**: 2026-06-17 盘前
