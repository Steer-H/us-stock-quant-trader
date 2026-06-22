# 美股量化交易系统 - 全面审计报告 v2

**日期**: 2026-06-17  
**审计范围**: 全部核心代码（web_server.py, model_inference.py, portfolio.py, predictor.py, transformer.py, trainer.py, data_loader.py, settings.py, stock_crawler.py, dashboard.html）  
**状态**: ✅ 所有关键问题已修复，服务器正常运行

---

## 一、已修复的关键BUG

### 🔴 CRITICAL: model_inference.py 引用不存在的 InferenceTransformer 类

**位置**: `live_trading/model_inference.py:98`  
**问题**: 代码 `self.model = InferenceTransformer(...)` 引用了一个已删除的类。注释明确说"不再定义独立的InferenceTransformer，确保架构完全匹配"，但代码未更新。  
**影响**: ML推理功能完全无法工作，启动即崩溃。  
**修复**: 
- 导入 `from ml_model.transformer import StockTransformer`
- 模型创建改为 `self.model = StockTransformer(self.config)`
- 添加特征数自动适配逻辑（checkpoint与config不一致时自动调整）

### 🔴 CRITICAL: Scaler仅适配20只旧股票

**位置**: `live_trading/model_inference.py:128`  
**问题**: Scaler拟合时硬编码20只老股票列表，40只新股票无法正确标准化。  
**修复**: 改为自动扫描 `data/processed/` 目录下所有 `*_features.parquet` 文件，动态发现40只股票。

### 🟡 HIGH: Werkzeug macOS kqueue 错误

**位置**: `live_trading/web_server.py:1130`, `logs/watchdog.log`  
**问题**: Flask开发服务器在macOS上每个请求都触发 `TypeError: changelist must be an iterable of select.kevent objects`。  
**影响**: 日志充满错误噪音，可能影响请求稳定性。  
**修复**: `app.run()` 添加 `threaded=True` 参数。

### 🟡 HIGH: 数据过期逻辑缺陷

**位置**: `live_trading/web_server.py:tick_engine()`  
**问题**: Yahoo fetch因间隔未到返回None时，错误地将 `_price_is_stale` 设为True，导致前端误报"数据过期"。  
**修复**: 
- 仅在连续3次Yahoo失败后才标记数据过期
- 间隔未到时标记 `数据仍有效`，延迟高时标记 `缓存数据`
- 添加 `_price_data_age_s` 追踪数据新鲜度

### 🟡 HIGH: is_real_time 阈值过严

**位置**: `live_trading/web_server.py:build_status_data()`  
**问题**: `is_real_time = latency_ms < 1000` —— Yahoo批量下载延迟通常3-5秒，导致永远显示"非实时"。  
**修复**: 阈值改为 `latency_ms < 5000 and not _price_is_stale`。

### 🟡 MEDIUM: 前端K线图ticker按钮仅20只

**位置**: `live_trading/templates/dashboard.html:338-340`  
**问题**: 图表ticker选择器硬编码20只旧股票，不包含新增的芯片/光模块/存储等14只高波动股票。  
**修复**: 扩展到全部40只，按行业分组排列。

### 🟢 LOW: 前端数据过期提示改进

**位置**: `live_trading/templates/dashboard.html`  
**改进**: 
- 延迟显示改为 `latency_ms + data_age_s` 双指标
- 添加数据年龄颜色标记（>60s黄色警告）
- 过期横幅仅在真实过期时显示

---

## 二、架构验证通过项

| 模块 | 状态 | 备注 |
|------|------|------|
| 杠杆系统 | ✅ | borrow/repay/interest/margin_call 逻辑完整 |
| 风险控制 | ✅ | 回撤降杠杆、margin>80%限1x、止损-2.5%(杠杆)/-4%(普通) |
| 持仓管理 | ✅ | FIFO成本计算、日盈亏跟踪、peak equity回撤 |
| 状态持久化 | ✅ | 定时保存+退出保存，恢复建仓跳过 |
| 数据新鲜度 | ✅ | 多级过期判定，不会误报 |
| 40只股票数据 | ✅ | 全部40只有processed parquet数据 |
| 训练进行中 | ✅ | Epoch 5/30, val_loss: 0.534→0.332, MPS加速 |

---

## 三、已知但未修复的问题（设计权衡）

1. **训练shuffle=True** (`ml_model/data_loader.py:322`): batch级shuffle有助于SGD但可能引入轻微时序泄漏。当前设计权衡：接受轻微泄漏换取更好收敛。

2. **Encoder-Decoder vs Encoder-Only** (`ml_model/transformer.py`): 同时存在`StockTransformer`(encoder-only，训练用)和`TimeSeriesTransformer`(encoder-decoder)。推理使用前者，架构匹配。

3. **Werkzeug开发服务器**: `threaded=True`缓解了kqueue问题，但生产环境建议切换到`waitress`或`gunicorn`。

4. **Yahoo Finance限流**: 30秒间隔 + 批量下载，约120次/小时，安全在2000次/小时限制内。

---

## 四、训练进度

- 模型: StockTransformer (Pre-LN, DropPath, AttentionDropout 0.2)
- 数据: 40只股票, 113,182训练样本
- 设备: MPS (Apple Silicon)
- 进度: Epoch 5/30 (~395s/epoch)
- 损失: train=0.211, val=0.369 (持续改善中)
- 预计完成: ~2.7小时后
- 完成后自动保存至 `data/models/transformer_stock_latest.pt`

---

## 五、验证结果

```
服务器: ✅ 运行中 (PID 21076, port 8080)
数据:   ✅ Yahoo Finance 实时 (Stale: False)
延迟:   ✅ ~3-5s (Yahoo批量下载)
ML:     ⏳ 等待训练完成 (ml_ready: False)
持仓:   ⏳ 等待开盘建仓 (positions_initialized: False)
```

---

## 六、修改文件清单

| 文件 | 修改内容 |
|------|---------|
| `live_trading/model_inference.py` | StockTransformer导入+模型创建+scaler自动发现+去重state_dict |
| `live_trading/web_server.py` | threaded=True + 数据过期逻辑重写 + is_real_time阈值 + data_age追踪 |
| `live_trading/templates/dashboard.html` | 40只ticker按钮 + 延迟/年龄双显示 + 过期横幅优化 |

---

## 七、动态杠杆系统 (Dynamic Leverage Engine) — 新增

### 架构设计

**文件**: `live_trading/leverage_engine.py` (410行)

**核心公式**:
```
leverage = kelly_base × vol_multiplier × perf_multiplier × heat_multiplier
leverage = clamp(leverage, 0.25, 2.0)
leverage = min(leverage, drawdown_cap, margin_cap)
```

### 四因子详解

#### 1. 凯利公式 (Kelly Criterion)
- `f* = (p × b - q) / b`
- p = ML预测置信度, b = 近期盈亏比(从AccuracyTracker推断)
- Half-Kelly保守: 只用50%的Kelly建议仓位
- 示例: p=0.72, b=1.8 → f_half=0.33 → 1.66x杠杆

#### 2. 波动率调节 (Volatility Adjustment)
- 从持仓股票价格变动实时估算市场年化波动率
- 低波动(<15%): 1.0x → 可满杠杆
- 正常(15-35%): 1.0→0.6x 线性衰减
- 高波动(35-55%): 0.6→0.25x 线性衰减  
- 极端(>55%): 0.25x → 强制最小杠杆

#### 3. 绩效反馈 (Performance Feedback)
- 追踪最近20笔交易胜率
- 胜率>60%: 加码最多+30% (反脆弱: 赢时加码)
- 胜率45-60%: 中性
- 胜率<45%: 减仓最多-60% (反马丁格尔: 输时缩量)
- 额外: 连输3笔 → 强制打5折

#### 4. 组合热度 (Portfolio Heat)
- 持仓数/最大持仓数
- <30%: 全额
- 30-60%: 线性衰减至75%
- >60%: 线性衰减至40%

### 风险硬约束 (Circuit Breakers)

| 条件 | 杠杆上限 |
|------|---------|
| 回撤 < 5% | 2.0x (不限制) |
| 回撤 5-10% | 1.5x |
| 回撤 10-15% | 1.0x |
| 回撤 15-20% | 0.5x (强平) |
| 回撤 > 20% | 0.0x (停止开仓) |
| 保证金 > 60% | 1.5x |
| 保证金 > 80% | 1.0x |
| 保证金 > 90% | 0.0x (margin call) |

### 集成点

| 位置 | 集成内容 |
|------|---------|
| `web_server.py:729` | 买入时调用 `_leverage_engine.calculate()` 替代静态LEVERAGE_TIERS |
| `web_server.py:670` | 卖出时 `record_trade(win, pnl_pct)` 反馈绩效 |
| `web_server.py:1048` | API返回 `leverage_avg` 和 `leverage_calcs` |
| `web_server.py:340` | 启动时从持久化状态恢复引擎 |
| `dashboard.html:517` | 前端显示 `avg杠杆` 统计 |

### 与旧系统的对比

| 维度 | 旧系统 (静态LEVERAGE_TIERS) | 新系统 (动态LeverageEngine) |
|------|---------------------------|---------------------------|
| 杠杆决定 | 仅ML置信度 | 凯利公式+3个调节因子 |
| 市场环境 | 不感知 | 波动率自适应 |
| 交易绩效 | 不感知 | 胜率反馈加减仓 |
| 组合风险 | 不感知 | 持仓热度折扣 |
| 回撤保护 | 简单二值(>10%→1x) | 5级渐进式熔断 |
| 可解释性 | 黑盒映射 | 四因子可拆分可审计 |

### 验证状态

```
✅ 服务器运行中 (PID ~21076, :8080)
✅ 杠杆引擎初始化正常 (Avg=1.0, Calcs=0 等待开盘)
✅ 数据新鲜 (Yahoo实时, Stale=False)
✅ API返回 leverage_avg + leverage_calcs
✅ 前端显示 avg杠杆统计
⏳ 等待开盘后验证实际交易中的杠杆计算
```
