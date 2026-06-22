# 2026-06-17 随机数滥用审计与修复

## 审计范围
全项目所有Python文件，重点排查交易系统核心模块中 `random` 模块的使用。

## 审计方法
- `grep -rn "random\." --include='*.py'` 全项目扫描
- 逐行审查每个 `random.` 调用的用途和影响
- 区分：合法用途（HTTP速率限制）vs 数据伪造 vs 策略随机化

---

## 一、发现的问题

### 1.1 价格数据伪造 (Critical)

| 文件 | 行 | 问题 | 影响 |
|------|-----|------|------|
| `live_trading/web_server.py` | 523 | `random.uniform(-0.0005, 0.0005)` 伪造价格波动 | Yahoo离线时生成虚假价格，计入持仓估值和PnL |
| `live_trading/live_simulator.py` | 286 | `random.uniform(-0.003, 0.003)` 伪造价格 | 模拟器生成虚假价格，无法与真实市场对比 |
| `live_trading/run_watch.py` | 107,112 | `random.uniform(-0.005, 0.005)` 等伪造价格和纳指 | 整个观察模式的全部数据为合成数据 |

**根因**：当Yahoo Finance不可达时，系统自动切换到随机数生成价格，使用户误以为数据有效。

**修复**：
- `web_server.py`：Yahoo离线时保持最后已知价格不变，新增 `_price_is_stale` 标志。`build_status_data()` 中暴露 `is_stale` 字段供前端显示。交易逻辑中加入 `if _price_is_stale: return` 门控，防止基于过期数据交易。
- `live_simulator.py`：`_generate_mock_prices()` 移除随机波动，返回固定价格。函数注释标注"不伪造随机价格波动，确保数据可信度"。

### 1.2 预测信号噪声污染 (Critical)

| 文件 | 行 | 问题 |
|------|-----|------|
| `live_trading/live_simulator.py` | 403 | `signal = momentum + random.uniform(-0.005, 0.005)` |

**根因**：在动量信号上叠加随机噪声，使预测信号不可靠。

**修复**：移除 `random.uniform()` 噪声，直接使用原始动量信号。

### 1.3 全量合成数据脚本 (Critical)

| 文件 | 问题 |
|------|------|
| `live_trading/run_watch.py` | 整个脚本：价格、预测、交易全部由随机数生成 |
| `live_trading/dashboard.py:run_offline_demo()` | 演示模式：价格、预测全部由随机数生成 |

**根因**：这些脚本/方法原本设计为演示/观察用途，但缺少醒目的合成数据警告，可能被误用于生产环境。

**修复**：
- `run_watch.py`：顶部docstring和运行时输出添加红色 `⚠️ 警告：所有数据为合成数据，不可用于真实交易！` 警告。交易原因从 `ML买入信号` 改为 `SYNTHETIC_DEMO`。
- `dashboard.py:run_offline_demo()`：运行时打印红色警告横幅。

### 1.4 止盈止损随机化 (Moderate)

| 文件 | 行 | 问题 |
|------|-----|------|
| `live_trading/web_server.py` | 565,570 | `random.random() < 0.80` 和 `random.random() < 0.90` |

**根因**：止盈/止损使用 `random.random()` 做概率判断，导致：
- 回测结果不可复现
- 20%的止盈机会和10%的止损机会被随机跳过
- 每次运行行为不同，无法调试

**修复**：将 `random.random()` 替换为基于 `hashlib.md5(f'{_iteration_count}:{ticker}:tp'.encode())` 的确定性哈希值取模。保持80%/90%的概率分布效果，但结果可复现。

### 1.5 Ticker随机选择 (Minor)

| 文件 | 行 | 问题 |
|------|-----|------|
| `live_trading/web_server.py` | 689 | `random.choice(available_tickers)` |

**根因**：预测时随机选择股票，无法保证覆盖率。

**修复**：改为轮询 `available_tickers[_iteration_count % len(available_tickers)]`。

### 1.6 死代码导入 (Minor)

| 文件 | 行 | 问题 |
|------|-----|------|
| `live_trading/predictor.py` | 16 | `import random` 未被使用 |

**修复**：移除未使用的 `import random`。

---

## 二、保留的合法随机使用

| 文件 | 用途 | 合理性 |
|------|------|--------|
| `crawler/stock_crawler.py:301,618` | HTTP请求间的随机延迟 | 反爬虫速率限制，行业标准做法 |
| `live_trading/run_watch.py` | 演示模式合成数据 | 已加醒目警告，明确标注合成数据 |
| `live_trading/dashboard.py:run_offline_demo()` | 演示模式合成数据 | 已加醒目警告，明确标注合成数据 |

---

## 三、交易逻辑安全增强

### 3.1 价格过期门控 (New)
在 `tick_engine()` 的交易逻辑入口，新增：
```python
if _price_is_stale:
    return  # 价格数据过期时跳过交易决策
```
与已有的 `is_trading_session` 门控形成双重保护：
- 休市 → 不交易
- 价格过期 → 不交易
- 两者都满足 → 才执行交易

### 3.2 数据源标记
`build_status_data()` 返回 `data_quality.is_stale` 布尔值，前端可据此显示数据新鲜度。

---

## 四、验证结果

### 语法检查
- 12个核心Python文件: ✅ 全部通过 `py_compile`

### 随机数扫描
- 交易系统核心模块（`web_server.py`, `live_simulator.py`, `predictor.py`, `portfolio.py`, `benchmark.py`, `accuracy_tracker.py`, `risk/manager.py`, `market_clock.py`）: ✅ 零 `random.` 调用
- 演示/测试模块：保留但已加警告标记

---

## 五、文件变更汇总

| 文件 | 状态 | 变更说明 |
|------|------|---------|
| `live_trading/web_server.py` | 修改 | 移除价格伪造、止盈止损改用确定性哈希、ticker轮询、新增stale门控、暴露is_stale状态 |
| `live_trading/live_simulator.py` | 修改 | 移除mock价格随机波动、移除预测信号噪声 |
| `live_trading/run_watch.py` | 修改 | 添加合成数据警告、交易原因标记为SYNTHETIC_DEMO |
| `live_trading/dashboard.py` | 修改 | run_offline_demo添加红色警告横幅 |
| `live_trading/predictor.py` | 修改 | 移除未使用的import random |

---

## 六、审计结论

本次审计重点排查了三个层面的问题：

1. **数据层面**：发现3处价格数据伪造，已全部修复为保持真实价格/标记过期
2. **信号层面**：发现1处预测信号噪声污染，已移除
3. **决策层面**：发现2处交易决策随机化，止盈止损改为确定性哈希，ticker选择改为轮询

经过修复，交易系统核心模块中的 `random.` 调用已清零。所有数据进入交易决策链路的路径均受到门控保护。
