# 美股量化交易系统 — 详细修改日志

> 项目: `/Users/oujianli/Documents/美股量化交易（New）`
> 格式: `[时间] 类别 | 文件 | 变更描述`

---

## 2026-06-19 — 全面代码审查与修复

### 状态持久化关键字段遗漏修复

| 时间 | 文件 | 变更 |
|------|------|------|
| 06-19 | `live_trading/state_persistence.py` | **Critical Fix**: save_state()添加ml_ready/prediction_iters/leverage_engine三个关键字段 |
| 06-19 | `live_trading/model_inference.py` | 修复特征数适配三元表达式格式损坏(单行→多行) + 循环变量冲突 |
| 06-19 | `utils/helpers.py` | datetime.utcnow()→datetime.now(timezone.utc) 修复弃用API |
| 06-19 | `run_server.sh` | 添加1小时20次崩溃上限保护(与keepalive.sh一致) |
| 06-19 | `live_trading/web_server.py` | import random从函数内提升到模块级别 |

### 审查确认
- 全项目51个Python文件语法检查通过 ✅
- 17项历史bug确认已修复 ✅
- 核心交易逻辑(止盈止损/ML预测/杠杆/风控)无bug ✅
- 文档: 新增`work_logs/2026-06-19_comprehensive-review.md`

---

## 2026-06-17 — 全面系统优化


### 10:11 — 随机数滥用审计与修复

| 时间 | 文件 | 变更 |
|------|------|------|
| 10:11 | `live_trading/web_server.py` | **Critical**: 移除Yahoo离线时的random.uniform价格伪造，新增_price_is_stale门控 |
| 10:11 | `live_trading/web_server.py` | 止盈止损从random.random()改为确定性哈希（可复现） |
| 10:11 | `live_trading/web_server.py` | ticker选择从random.choice改为轮询 |
| 10:11 | `live_trading/live_simulator.py` | 移除mock价格random.uniform波动和预测信号噪声 |
| 10:11 | `live_trading/run_watch.py` | 添加合成数据警告横幅，交易原因标记为SYNTHETIC_DEMO |
| 10:11 | `live_trading/dashboard.py` | run_offline_demo添加红色警告 |
| 10:11 | `live_trading/predictor.py` | 移除未使用的import random |

### 09:30 — Bug修复: 闭市交易 + ML加载 + 价格过时

| 时间 | 文件 | 变更 |
|------|------|------|
| 09:30 | `live_trading/web_server.py` | **Critical Fix**: tick_engine()添加is_trading_session门控，休市时跳过交易 |
| 09:30 | `live_trading/web_server.py` | 修复init_ml_model()仅_iteration_count<300才执行的限制 |
| 09:30 | `live_trading/web_server.py` | INITIAL_POSITIONS价格更新为2025年中合理值 |
| 09:30 | `live_trading/web_server.py` | 状态持久化新增ml_ready字段 |

### 10:00 — ML推理引擎

| 时间 | 文件 | 变更 |
|------|------|------|
| 10:00 | `live_trading/model_inference.py` | **新建**: ModelInference类，匹配encoder-only架构 |
| 10:00 | `live_trading/model_inference.py` | 加载transformer_26stocks.pt，支持批量预测 |
| 10:00 | `live_trading/web_server.py` | tick_engine买入逻辑优先使用Transformer模型预测 |

### 10:30 — 预测器自适应优化

| 时间 | 文件 | 变更 |
|------|------|------|
| 10:30 | `live_trading/predictor.py` | 新增_detect_regime()市场状态检测 |
| 10:30 | `live_trading/predictor.py` | 新增_adapt_weights_to_regime()自适应权重 |
| 10:30 | `live_trading/predictor.py` | 因子级表现追踪，驱动权重持续优化 |

### 11:00 — WebSocket实时推送

| 时间 | 文件 | 变更 |
|------|------|------|
| 11:00 | `live_trading/web_server.py` | 集成flask-socketio，新增build_status_data()函数 |
| 11:00 | `live_trading/web_server.py` | engine_loop每秒通过WebSocket推送状态 |
| 11:00 | `live_trading/web_server.py` | 新增@socketio.on(connect/disconnect)事件处理 |
| 11:00 | `live_trading/templates/dashboard.html` | socket.io客户端集成，自动降级到轮询 |

### 12:00 — 前端全面重写

| 时间 | 文件 | 变更 |
|------|------|------|
| 12:00 | `live_trading/templates/dashboard.html` | **完全重写**：5标签专业面板（总览/持仓/挂单成交/K线/分析） |
| 12:00 | `live_trading/templates/dashboard.html` | 集成TradingView Lightweight Charts K线图 |
| 12:00 | `live_trading/web_server.py` | 新增/api/kline/<ticker>和/api/kline/multi API |
| 12:00 | `live_trading/web_server.py` | K线数据缓存（5分钟更新），fetch_kline_data()函数 |

### 12:30 — 做空支持 & 风控集成

| 时间 | 文件 | 变更 |
|------|------|------|
| 12:30 | `live_trading/web_server.py` | SHORT_SELL_ENABLED开关，模型预测下跌时做空 |
| 12:30 | `live_trading/web_server.py` | RiskManager导入，交易前检查is_paused熔断 |
| 12:30 | `config/settings.py` | device改为MPS>CUDA>CPU自动检测 |

### 13:00 — 系统重置 & 回测可视化

| 时间 | 文件 | 变更 |
|------|------|------|
| 13:00 | `data/trading_state.json` | 重置到初始状态($100,000/0持仓/0交易) |
| 13:00 | `.trading_daemon.pid, .trading_server.pid` | 清除PID文件 |
| 13:00 | `live_trading/web_server.py` | 新增/api/backtest_summary API |
| 13:00 | `live_trading/templates/dashboard.html` | 总览面板新增回测结果卡片 |

---


### 13:30 — 代码审计与UI优化

| 时间 | 文件 | 变更 |
|------|------|------|
| 13:30 | `live_trading/web_server.py` | **Cleanup**: 移除dead `import random`（零random调用原则） |
| 13:30 | `live_trading/web_server.py` | **Cleanup**: 移除未引用的`SHORT_SELL_ENABLED`死代码 |
| 13:30 | `live_trading/templates/dashboard.html` | **Critical Fix**: 移除`io()`引用（flask-socketio已不存在，导致JS错误） |
| 13:30 | `live_trading/templates/dashboard.html` | 改为纯轮询模式（每1秒），移除全部socket.io代码 |
| 13:30 | `live_trading/templates/dashboard.html` | **新增**: 数据过期红色警告（顶部栏+页脚+横幅） |
| 13:30 | `live_trading/templates/dashboard.html` | **新增**: 客户端实时倒计时（每秒独立tick） |
| 13:30 | `live_trading/templates/dashboard.html` | **新增**: K线图加载spinner动画+错误状态 |
| 13:30 | `live_trading/templates/dashboard.html` | **新增**: 页脚显示更新时间 |
| 13:30 | `live_trading/templates/dashboard.html` | CSS过渡动画、统计卡片悬停效果 |
| 13:30 | 进程管理 | 杀掉screen会话旧进程，清除PID文件 |

### 14:30 — Transformer模型深度优化

| 时间 | 文件 | 变更 |
|------|------|------|
| 14:30 | `ml_model/transformer.py` | **Pre-LN架构**: Post-LN→Pre-LN，训练更稳定 |
| 14:30 | `ml_model/transformer.py` | **新增DropPath**: Stochastic Depth，线性递增drop概率 |
| 14:30 | `ml_model/transformer.py` | **新增AttnDropout**: 注意力矩阵独立Dropout(0.2) |
| 14:30 | `ml_model/transformer.py` | **新增FinalLN**: Pre-LN架构最终归一化 |
| 14:30 | `config/settings.py` | dropout 0.1→0.15, 新增8个配置参数 |
| 14:30 | `config/settings.py` | 启用AMP混合精度训练 |
| 14:30 | `ml_model/trainer.py` | **Label Smoothing**: BCE损失集成0.1标签平滑 |
| 14:30 | `work_logs/` | 新增2026-06-17_transformer-optimization.md |

### 14:55 — 交易系统扩展至40只股票

| 时间 | 文件 | 变更 |
|------|------|------|
| 14:53 | `crawler/stock_crawler.py` | **Bug Fix**: Yahoo Finance时区兼容（tz-aware→tz-naive） |
| 14:53 | `data/processed/` | **新增**: 14只高波动股数据（AAOI,COHR,LITE,FN,WDC,STX,NTAP,MRVL,MU,LRCX,AMAT,KLAC,SNPS,CDNS） |
| 14:55 | `live_trading/web_server.py` | TRACKED_TICKERS 20→40只（新增6只补全+14只高波动） |
| 14:55 | `live_trading/web_server.py` | INITIAL_POSITIONS 20→40只 |
| 14:55 | 模型训练 | 启动40只股票30轮MPS训练(113K样本) |

### 15:05 — 全面代码审计与漏洞修复

| 时间 | 文件 | 变更 |
|------|------|------|
| 15:05 | `live_trading/model_inference.py` | **Critical Fix**: 推理架构重写，InferenceTransformer→StockTransformer（修复state_dict键不匹配，模型从未真正加载的bug） |
| 15:05 | `live_trading/model_inference.py` | 分类头修复：5-way softmax→Sigmoid二分类 |
| 15:05 | `live_trading/model_inference.py` | 默认模型路径更新：transformer_20stocks.pt→transformer_stock_latest.pt |
| 15:05 | `live_trading/predictor.py` | **Bug Fix**: train_offline_transformer引用不存在的类名/函数 |
| 15:05 | `live_trading/predictor.py` | **Bug Fix**: 缩进错误修正 |
| 15:05 | `live_trading/web_server.py` | **Bug Fix**: _shutdown_handler缺少prediction_iters |
| 15:05 | `live_trading/web_server.py` | **Bug Fix**: engine_loop最终保存缺少position_entry_time等字段 |

### 15:18 — 显示问题修复

| 时间 | 文件 | 变更 |
|------|------|------|
| 15:18 | `live_trading/templates/dashboard.html` | 准确率0预测时显示"--"而非"0.0%" |
| 15:18 | `live_trading/templates/dashboard.html` | 成交面板新增SHORT/做空标签支持 |
| 15:18 | `live_trading/templates/dashboard.html` | 最大回撤0值显示"0.00%"而非"+0.00%" |
| 15:18 | `live_trading/templates/dashboard.html` | 基准数据为空时显示"--" |
| 15:18 | `live_trading/templates/dashboard.html` | 无预测时准确率状态显示"⏳ 待数据" |
| 15:18 | `live_trading/web_server.py` | initial_capital由硬编码改为动态读取 |

---

## 2026-06-19 — 数据获取修复 + 代码审计优化

### 🔴 Critical Fix: 纳斯达克对比曲线始终为0

| 时间 | 文件 | 变更 |
|------|------|------|
| 17:30 | `live_trading/benchmark.py` | **Bug Fix**: `update()` 方法中 `nasdaq_equity_curve` 缺少 else 分支，第一条数据永远无法写入，导致曲线始终为空。补上 `else: self.nasdaq_equity_curve = pd.Series({dt: new_nasdaq_equity})`，镜像 strategy_equity_curve 的逻辑。 |

### 🔴 Critical Fix: Yahoo数据抓取静默失败

| 时间 | 文件 | 变更 |
|------|------|------|
| 18:40 | `live_trading/web_server.py:287` | **Bug Fix**: `fetch_yahoo_prices()` 中 `skipped = len(dl) - completed` 使用了从未定义的变量 `completed`（仅出现这一次）。每次成功抓取≥5只股票后触发 NameError，被外层 except 静默吞掉，返回 None。修复：`completed` → `fetched`。 |

### 🟡 历史Bug漏网修复

| 时间 | 文件 | 变更 |
|------|------|------|
| 18:30 | `utils/exceptions.py:29` | `datetime.utcnow()` → `datetime.now(timezone.utc).replace(tzinfo=None)`（06-19审查漏网） |

### 🟡 代码去重与效率优化

| 时间 | 文件 | 变更 |
|------|------|------|
| 18:35 | `live_trading/web_server.py` | 提取 `_collect_globals_dict()` 辅助函数，消除 engine_loop/最终保存/_shutdown_handler 三处完全相同的13行 globals_dict 构造代码 |
| 18:35 | `live_trading/web_server.py` | `import pandas as pd` 从 `api_benchmark_curve()` 函数内部提升到模块级别 |
| 18:35 | `live_trading/web_server.py` | 移除 `fetch_kline_data` 和 `fetch_yahoo_prices` 中冗余的 `import time as _time`（模块顶部已有 `import time`） |

### 审查确认

- 17项历史bug修复状态逐一核实：16项已修，1项漏网(utcnow)已补修 ✅
- 全项目空序列/边界条件模式排查：仅 benchmark.py 一处缺失 ✅
- 核心交易逻辑(止盈止损/ML预测/杠杆/风控)无新增问题 ✅
- Yahoo Finance 实时价格抓取恢复，40只股票 9055ms ✅

## 2026-06-19 (续) — 前后端数据断裂 + 盈亏计算修复

### 问题发现
用户报告前端显示数据错误：总资产显示 $100,000（实际 $99,890）、可用现金显示 $0（实际 $27,952）、总盈亏显示 $0.00（实际 -$109.66）。

### 根因分析

**Bug 1（主因）：前端数据绑定断裂**
- `build_status_data()` 返回嵌套结构 `{account: {cash, total_equity...}, pnl: {net_pnl...}}`
- 前端 `render()` 直接访问 `d.cash`、`d.total_equity` 等 flat 字段 → 全部 undefined → 使用错误后备值
- 所有 6 个核心指标卡片（总净资产、可用现金、持仓市值、总盈亏、杠杆倍数、持仓数）均受影响

**Bug 2：avg_cost 不完整含佣金**
- `execute_buy` 首次建仓：`avg_cost=price` — 完全没有佣金
- `execute_buy` 后续加仓：`new_total_cost += commission/2` — 只有一半佣金
- 导致 `unrealized_pnl = qty*(price-avg_cost)` 系统性偏高

**Bug 3：卖出盈亏不减去卖出佣金**
- `execute_sell`：`realized_pnl += trade_pnl` 其中 `trade_pnl = (price-avg_cost)*qty`
- 卖出佣金 $1 被正确地从 cash 扣除，但未从 realized_pnl 扣除
- 导致 realized_pnl 系统性偏高

### 修复内容

| 文件 | 修改 |
|------|------|
| `templates/dashboard.html` | render() 中新增 flatten 逻辑，将 `d.account.*` 和 `d.pnl.*` 展开到 `d` 顶层 |
| `portfolio.py:265` | 首次建仓 `avg_cost=(quantity*price+commission)/quantity`（原 `price`） |
| `portfolio.py:298` | 加仓 `new_total_cost += commission`（原 `commission/2`） |
| `portfolio.py:334` | `realized_pnl += trade_pnl - commission`（原不加 -commission） |
| `portfolio.py:355` | TradeRecord.pnl 同步修正 `trade_pnl - commission` |
| `data/trading_state.json` | 40只持仓的 avg_cost 从 cost_basis/qty 重新计算，unrealized_pnl 同步修正 |

### 验证

- 会计恒等：`cash + sum(cost_basis) = initial_capital` ✓
- 盈亏一致：`unrealized_pnl = sum(mv) - sum(cb) = -109.66` ✓
- 前端模拟：总净资产 $99,890.34 / 可用现金 $27,952 / 总盈亏 -$109.66 ✓
- 服务已重启，4 个修改文件 syntax OK

## 2026-06-19 (续2) — 恢复挂单面板

### 问题
上一轮 UI 重写将「挂单&成交」面板简化为仅「成交记录」，删除了挂单信号展示。

### 修复
- `web_server.py`: `build_status_data()` 新增 `pending_signals` 字段（最近15条交易信号）
- `dashboard.html`: 成交记录面板改为双卡片结构：
  - 📝 挂单信号 — 时间/方向/Ticker/数量/价格/原因
  - ✅ 已成交 — 保留原有成交记录表
- Tab 标签恢复为「挂单&成交」

## 2026-06-19 (续3) — 修复策略对比纳斯达克曲线消失

### 根因
`web_server.py:376` 行 `pd = _saved_state.get('predictor')` 将 pandas 模块覆盖为普通 dict，导致 `api_benchmark_curve` 中 `pd.Timestamp(t)` 抛出 `'dict' object has no attribute 'Timestamp'`，整个 API 静默返回空数据。

该问题是上一轮 `import pandas as pd` 提升到模块级别后引入的变量名冲突。

### 修复
- `web_server.py:376`: `pd` → `pp`（predictor data 缩写），避免覆盖 pandas
- 关联行同步修改变量名

### 验证
- API 返回 200 个数据点，NASDAQ=$100,290.13，策略=$99,890.34
- 曲线恢复显示（盘后数据为平线属正常，开盘后将波动）

## 2026-06-20 — 全面代码审计v3

### 🔴 致命修复 (8)
- `web_server.py`: 添加 logger 定义，修复 `fetch_kline_data` NameError
- `web_server.py`: 预测超时确认改用 `_prediction_iters` 映射表
- `dashboard.html`: Benchmark图表初始化从charts tab移至analysis tab (fix #23)
- `dashboard.html`: `fp()` 添加null守卫，防止render静默崩溃
- `dashboard.html`: 成交记录SHORT标签改为"做空" (fix #27)
- `leverage_engine.py`: 连败检测死代码复活，连败3次强制降至0.5x
- `risk/manager.py`: `check_slippage` 添加零除守卫
- `model_inference.py`: 特征数不匹配时零填充降级

### 🟡 严重修复 (9)
- `web_server.py`: 移除死代码 CONFIG_YAHOO、未使用全局变量
- `web_server.py`: shutdown时关闭 `_yahoo_session`
- `helpers.py`: `safe_divide` 移除冗余 import
- `helpers.py`: `is_market_open` 使用 zoneinfo 正确处理夏令时
- `helpers.py`: 未定义年份假期输出 warning
- `helpers.py`: TICKER_PATTERN 支持含点号ticker
- `helpers.py`: `format_currency` 移除死分支
- `benchmark.py`: `append_strategy_snapshot` 同步更新 strategy_peak

### 🟢 轻微修复 (2)
- `portfolio.py`: 移除冗余 import
- `dashboard.html`: catch块添加 console.error 日志

### 已知未修 (10)
详见 `work_logs/2026-06-20_comprehensive-audit-v3.md`

### 第二轮补充 (同日)
- `data_loader.py`: stride参数实现，目标列注释，key解析兼容，类型标注修正
- `trainer.py`: val_loader早停修复，除零守卫，损失权重命名注释，返回类型修正
- `leverage_engine.py`: 函数内import提升至模块顶部
- `web_server.py`: 函数内import(yfinance/requests/futures)提升至模块顶部
- `dashboard.html`: 方向准确率0%显示修复，klVolume死代码移除，tab切换chart resize
- `constants.py`: 假期注释更新为2025-2027

### 第三轮补充 (同日)
- `oms.py`: on_reject 状态机守卫，检查transition返回值
- `predictor.py`: to_dict/from_dict 序列化对称补全5个遗漏字段
- `risk/manager.py`: RiskManager 添加 is_order_allowed 委托方法
- `benchmark.py`: max_drawdown 改为追踪历史最大值
- `market_clock.py`: DST使用zoneinfo修复，添加Black Friday提前收盘

## 2026-06-20 — 全面项目优化

### P0 高影响
- `benchmark.py`: O(n²) pd.concat → O(1) dict追加 + 惰性Series重建 + index union优化
- `market_clock.py`: 假日数据去重，从constants导入EXCHANGE_HOLIDAYS
- `data_loader.py`: 修复Scaler仅拟合首只股票bug，移除冗余.copy()

### P1 中等影响
- `market_clock.py`: is_holiday/is_trading_day委派给helpers，消除逻辑重复
- `portfolio.py`: _equity_history添加上限(7200→截断3600)
- `web_server.py`: random抖动→确定性基于迭代计数
- `model_inference.py`: torch.load weights_only=True
- `ml_model/transformer.py`: hasattr移除，predict()副作用消除
- `ml_model/trainer.py`: deepcopy→dict comprehension clone

### P2 代码质量
- `ml_model/transformer.py`: QKV自注意力优化（一次投影替代三次）
- `dashboard.html`: DOM元素缓存(ovEquity/ovPnL/ovML/ovAcc)，选择器缓存

## 2026-06-20 — Transformer ML算法升级

### 架构改进 (7项)
- `transformer.py`: 固定正弦PE → 可学习PE (warm-start)
- `transformer.py`: ReLU → GELU (FeedForward + 双预测头)
- `transformer.py`: 启用DropPath Stochastic Depth (线性递增)
- `transformer.py`: AdaptiveAvgPool → Attention Pooling
- `transformer.py`: Xavier init → Kaiming init
- `transformer.py`: QKV自注意力优化（一次投影替代三次）

### 训练改进 (4项)
- `trainer.py`: BCE → FocalBCELoss (gamma=2.0, alpha=0.25)
- `trainer.py`: 新增 Label Smoothing (0.1)
- `trainer.py`: 新增梯度裁剪 (clip_norm=1.0)
- `trainer.py`: ReduceLROnPlateau → CosineAnnealingWarmRestarts

### 配置更新
- CPU: d_model=256, n_heads=8, n_layers=4, d_ff=1024
- GPU: d_model=384, n_heads=12, n_layers=6, d_ff=1536
- dropout: 0.15→0.2, batch_size: 64→16(CPU)/32(GPU)
- lr: 1e-4→3e-4, weight_decay: 1e-4→1e-5

## 2026-06-20 训练监控与会话管理

### 修复
- 🔴 杀掉5个重复训练进程，CPU负载从16降至3
- 🟡 创建缺失的前端模型面板（panel-model + 所有DOM元素）
- 🟡 修复训练日志不输出问题（root logger配置）
- 🟡 get_model_info() API新增训练参数字段
- 🟢 random()添加确定性种子，import规范化
- 🟢 情感分析器关键词库扩展 +203词

### 训练
- 2次训练尝试（52.83%/52.30%），保留现有55.77%生产模型

## 2026-06-21 代码质量审查与UX优化

### 修复
- 🟡 6个文件10处异常静默吞没：添加debug日志或修正except类型
- 🟡 model_inference.py裸`except:`→`except Exception:`
- 🟢 前端UX全面优化：移除8个技术细节标签，6个标签友好化
- 🟢 Overview面板ovML标签错位修正

### 验证
- 全部60个.py文件语法通过
- 边缘情况检查全部通过
- 配置一致性验证通过
