# 2026-06-19 代码清理与稳定化审计

**时间**: 2026-06-19 20:45-21:00 (UTC+8)  
**类型**: 全项目巡视 — 按GUARDRAILS.md规则系统性审查

---

## 一、巡视范围

| 维度 | 方法 | 结果 |
|------|------|------|
| global声明遗漏 | AST扫描所有函数 vs 模块级变量 | 发现1处 |
| 序列化对称性 | `_collect_globals_dict` vs `save_state` globals | ✅ 一致 |
| 未使用import | AST Name节点追踪 | 发现22处 |
| 前后端字段一致性 | API return dict vs 前端 d.xxx引用 | ✅ 一致 |
| 死代码/冗余 | 重复定义扫描 | 无 |

---

## 二、发现并修复的问题

### 🔴 BUG: `init_positions()` 缺少 `global _last_price_update`

**文件**: `live_trading/web_server.py:465`  
**问题**: `init_positions()` 声明了 `global _positions_initialized, _portfolio, _current_prices, _data_source` 但遗漏了 `_last_price_update`。导致函数内对 `_last_price_update` 的赋值创建了局部变量，全局变量保持空字符串。

**修复**: 添加 `_last_price_update` 到 global 声明。

---

### 🟢 清理：22 个未使用的 import

| 文件 | 移除的import | 数量 |
|------|-------------|------|
| `ml_model/trainer.py` | `json`, `Callable`, `F`(torch.nn.functional), `system_config`, `prepare_data`, `Timer`, `LowAccuracyError` | 7 |
| `ml_model/data_loader.py` | `Dict`, `Union`, `np`(numpy), `RobustScaler`, `MinMaxScaler`, `ConfigurationError` | 6 |
| `live_trading/web_server.py` | `os`, `json`, `get_inference`, `AccountSnapshot`, `PreTradeRisk` | 5 |
| `data_pipeline/cleaner.py` | `Set`, `Tuple`, `clip_values`, `DataAlignmentError` | 4 |
| `data_pipeline/indicators.py` | `Optional`, `Tuple` | 2 |

**验证方法**: AST遍历所有Name/Attribute节点，精确确认每个import无任何使用。

---

## 三、确认无问题的区域

| 检查项 | 状态 |
|--------|------|
| `save_state` / `load_state` 序列化对称 | ✅ 12字段完全一致 |
| `tick_engine` 交易门控 | ✅ `is_trading_session`、价格过期、建仓逻辑均正确 |
| 前端 `d.xxx` 引用 vs API字段 | ✅ 无孤悬引用 |
| `StockTransformer` forward签名 | ✅ 未被修改 |
| `_shutdown_handler` 无重复定义 | ✅ 仅1处 |
| `TimeSeriesTransformer` 死代码 | 保留（GUARDRAILS设计权衡） |

---

## 四、修改文件清单

| 文件 | 修改 | 严重度 |
|------|------|--------|
| `live_trading/web_server.py` | `init_positions()` 添加 `global _last_price_update` | 🔴 Bug |
| `live_trading/web_server.py` | 移除 5 个未使用 import | 🟢 清理 |
| `ml_model/trainer.py` | 移除 7 个未使用 import | 🟢 清理 |
| `ml_model/data_loader.py` | 移除 6 个未使用 import | 🟢 清理 |
| `data_pipeline/cleaner.py` | 移除 4 个未使用 import | 🟢 清理 |
| `data_pipeline/indicators.py` | 移除 2 个未使用 import | 🟢 清理 |

---

## 五、功能验证

- 全项目 40+ Python 文件语法检查 ✅
- 服务重启后 API 正常返回 ✅
- 数据源: `Yahoo Finance (缓存 [v7])` ✅
- 40只持仓、纳指 $26517.93、超额 -0.4% ✅

---

> **结论**: 项目稳定。1个global声明bug已修，22个冗余import已清理。无功能影响。
