# 2026-06-19 数据获取修复 & 代码审计

**日期**: 2026-06-19  
**触发**: 用户反馈纳斯达克对比曲线显示为0

---

## 一、纳斯达克曲线为0的根因

**文件**: `live_trading/benchmark.py:205`

`update()` 方法中，纳斯达克权益曲线（`nasdaq_equity_curve`）的写入逻辑只有追加分支，没有初始创建分支：

```python
# 旧代码（bug）
if not self.nasdaq_equity_curve.empty:
    self.nasdaq_equity_curve = pd.concat([...])  # 只能追加
# ← 无 else，第一条数据永远写不进去

# 策略曲线有完整的 if/else（正确）
if not self.strategy_equity_curve.empty:
    self.strategy_equity_curve = pd.concat([...])
else:
    self.strategy_equity_curve = pd.Series({dt: strategy_equity})  # 第一条
```

`nasdaq_equity_curve` 初始化为空 `pd.Series`，第一条数据因缺少 `else` 无法写入，曲线永远为空，前端拿到全部是0。

---

## 二、Yahoo数据抓取静默失败

**文件**: `live_trading/web_server.py:287`

```python
fetched = len(prices)
skipped = len(dl) - completed   # ← NameError: 'completed' 从未定义
```

变量 `completed` 在整个函数中只出现这一次。每次 Yahoo 成功抓取 ≥5 只股票后，代码走到此行即抛 `NameError`，被外层 `except Exception` 捕获：
- `_yahoo_error_count += 1`
- 返回 `None`
- 前端标记数据过期

**成功抓取被伪装成了失败**。修复：`completed` → `fetched`。

验证结果：40只股票 9055ms 恢复正常。

---

## 三、历史Bug补漏

| 文件 | 问题 | 状态 |
|------|------|------|
| `utils/exceptions.py:29` | `datetime.utcnow()` 未替换 | 06-19审查漏网，已修复 |

---

## 四、代码去重

| 文件 | 改动 | 说明 |
|------|------|------|
| `web_server.py` | 提取 `_collect_globals_dict()` | 消除3处重复的13行字典构造 |
| `web_server.py` | `import pandas as pd` 提升 | 从函数内移到模块顶部 |
| `web_server.py` | 移除 `import time as _time`×2 | 模块顶部已有 `import time` |

---

## 五、全部51个Python文件语法验证通过 ✅
