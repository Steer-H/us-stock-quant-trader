# 配置参考

## 1. ModelConfig — 模型配置 (`config/settings.py`)

### 架构参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `d_model` | 192 | Transformer 隐藏维度 |
| `n_heads` | 8 | 多头注意力头数 |
| `n_layers` | 4 | Encoder 层数 |
| `d_ff` | 768 | Feed-Forward 中间维度 (4×d_model) |
| `dropout` | 0.15 | 通用 Dropout 概率 |
| `attn_dropout` | 0.1 | 注意力 Dropout |
| `drop_path_rate` | 0.1 | Stochastic Depth 概率 |
| `activation` | `'gelu'` | 激活函数 (gelu/relu) |

### 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `batch_size` | 64 | 训练批次大小 |
| `epochs` | 100 | 训练轮数 |
| `learning_rate` | 1e-4 | 初始学习率 |
| `min_lr` | 1e-6 | 最小学习率 |
| `weight_decay` | 1e-4 | L2 正则化系数 |
| `grad_clip` | 1.0 | 梯度裁剪阈值 |
| `patience` | 10 | 早停耐心值 |
| `label_smoothing` | 0.1 | 标签平滑因子 |
| `focal_gamma` | 2.0 | Focal Loss γ |
| `direction_reward_weight` | 0.6 | 回归损失权重 (MSE) |
| `magnitude_reward_weight` | 0.4 | 分类损失权重 (BCE) |

### 数据参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `seq_len` | 60 | 输入序列长度 |
| `feature_columns` | 28 | 技术指标数量 |
| `prediction_horizon` | 1 | 预测步长 |
| `use_sentiment` | True | 是否使用情感特征 |

### 硬件参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `device` | `'mps'` (Apple Silicon) | 训练设备 (mps/cpu/cuda) |
| `use_amp` | True | 混合精度训练 |
| `num_workers` | 0 | DataLoader 工作线程 |

### 路径配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `data_dir` | `Path('data')` | 数据目录 |
| `model_dir` | `Path('models')` | 模型保存目录 |
| `log_dir` | `Path('logs')` | 日志目录 |

---

## 2. 交易参数 (`live_trading/web_server.py`)

### 跟踪股票 (40只)

```python
TRACKED_TICKERS = [
    # 科技七巨头
    'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'META', 'TSLA',
    # 软件/SaaS
    'NFLX', 'ADBE', 'CRM', 'NOW', 'ORCL',
    # 金融
    'JPM', 'V', 'MA', 'BAC',
    # 消费
    'WMT', 'HD', 'NKE', 'SBUX', 'UBER',
    # 芯片/半导体
    'AVGO', 'AMD', 'INTC', 'QCOM', 'TXN',
    # 光通信 (高波动)
    'AAOI', 'COHR', 'LITE', 'FN',
    # 存储 (高波动)
    'WDC', 'STX', 'NTAP',
    # 数据中心芯片
    'MRVL', 'MU',
    # 半导体设备
    'LRCX', 'AMAT', 'KLAC',
    # EDA软件
    'SNPS', 'CDNS',
]
```

### 风险控制

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `PROFIT_TAKE_THRESHOLD` | 0.03 (3%) | 止盈阈值 |
| `STOP_LOSS_THRESHOLD` | -0.04 (4%) | 止损阈值 |
| `MAX_POSITION_HOLD_TIME` | 30 (分钟) | 最大持仓时间 |
| `REENTRY_COOLDOWN` | 5 (分钟) | 卖出后冷却时间 |
| `POSITION_MAX_PCT` | 0.08 (8%) | 单只最大仓位 |
| `PREDICTIVE_SELL_THRESHOLD` | 0.55 | 预测卖出触发准确率 |

### 杠杆参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `MAX_LEVERAGE` | 2.0 | 最大杠杆倍数 |
| `MIN_LEVERAGE` | 0.25 | 最小杠杆倍数 |
| `MAX_POSITION_PCT_LEVERAGED` | 0.12 (12%) | 杠杆模式最大仓位 |
| `LEVERAGE_STOP_LOSS` | -0.025 (2.5%) | 杠杆止损 |
| `DRAWDOWN_DELEVERAGE` | 0.10 (10%) | 回撤去杠杆触发 |

### 杠杆引擎内部常量 (`leverage_engine.py`)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `MAX_LEVERAGE` | 2.0 | 全局最大杠杆 |
| `MIN_LEVERAGE` | 0.25 | 全局最小杠杆 |
| `DEFAULT_WIN_LOSS_RATIO` | 1.5 | 默认盈亏比 |
| `MAX_WIN_LOSS_RATIO` | 3.0 | 最大盈亏比 |
| `HALF_KELLY_FRACTION` | 0.5 | Half-Kelly系数 |
| `PERF_BOOST_WINRATE` | 0.60 | 加码触发胜率 |
| `PERF_REDUCE_WINRATE` | 0.45 | 减仓触发胜率 |

### Yahoo Finance 抓取

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `YAHOO_INTERVALS['REGULAR']` | 12 (秒) | 正常交易时段抓取间隔 |
| `YAHOO_INTERVALS['PRE_MARKET']` | 60 (秒) | 盘前抓取间隔 |
| `YAHOO_INTERVALS['AFTER_HOURS']` | 60 (秒) | 盘后抓取间隔 |
| `YAHOO_INTERVALS['CLOSED']` | 60 (秒) | 闭市抓取间隔 |

---

## 3. 市场时间 (`market_clock.py`)

```python
MARKET_TIMES = {
    'pre_market_start':   time(4, 0),     # 04:00 ET
    'regular_start':      time(9, 30),    # 09:30 ET
    'regular_end':        time(16, 0),    # 16:00 ET
    'early_close_end':    time(13, 0),    # 13:00 ET (Black Friday)
    'after_hours_end':    time(20, 0),    # 20:00 ET
}
```

**假期数据**: 2025-2027年完整美股假期 (Martin Luther King Jr. Day, Presidents Day, Good Friday, Memorial Day, Juneteenth, Independence Day, Labor Day, Thanksgiving, Christmas, New Year's Day + 浮动调整)

---

## 4. 资源配置建议

### 开发/测试环境
```python
ModelConfig:
    d_model = 128
    n_layers = 2
    n_heads = 4
    batch_size = 32
    epochs = 30
    device = 'cpu'
```

### 生产环境 (Apple Silicon)
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

### 生产环境 (NVIDIA GPU)
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

## 5. 修改配置后

⚠️ 修改 `ModelConfig` 的参数（尤其是架构参数如 `d_model`、`n_layers`）会导致旧 checkpoint 不兼容。需要：

1. 删除旧模型: `rm models/*.pt`
2. 重新训练: `python3 scripts/quick_train.py`
3. 重启服务

---

> 📖 **返回**: [README.md](README.md) 文档首页
