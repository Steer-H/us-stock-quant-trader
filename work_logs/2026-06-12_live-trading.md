# 工作日志 - 2026-06-12 在线模拟交易系统

## 概述
在初始项目基础上，新增完整的在线模拟交易系统，实现实时行情同步、持仓展示、盈亏追踪、模型准确率监控和纳指基准对比。

## 新增模块: live_trading/ (7个文件)

### market_clock.py — 市场时钟
- 美股四大时段检测（盘前/正常/盘后/闭市）
- 精确倒计时至开市/闭市（时:分:秒格式）
- 夏令时(EDT)与冬令时(EST)自动切换
- 美股假期日历（2025-2026完整列表）
- 全局便捷函数: `get_market_status()`, `countdown_to_market()`

### portfolio.py — 持仓与盈亏管理
- 初始资金 $100,000
- 买入/卖出操作（FIFO成本计算，佣金模拟）
- 实时市值更新（按最新行情）
- 详细持仓明细（股票代码/数量/成本价/现价/市值/盈亏金额/盈亏%/权重）
- PortfolioSnapshot 完整账户快照
- TradeRecord 交易审计记录

### benchmark.py — 纳指基准对比
- NASDAQ(^IXIC)实时追踪
- 策略 vs 纳指多维度对比（累计收益/年化收益/夏普比率/最大回撤）
- Alpha/Beta分析（OLS回归）
- 信息比率和跟踪误差
- 超额收益评价（跑赢/持平/跑输）

### accuracy_tracker.py — 模型准确率追踪
- 每笔预测记录（方向/收益率/置信度）
- 方向准确率 + 最近50次滚动准确率
- RMSE/MAE 误差统计
- 模型性能退化检测
- 按股票分组准确率统计

### live_simulator.py — 核心引擎
- 每分钟同步真实行情（yfinance）
- 交易时段自动生成预测信号并执行模拟交易
- 非交易时段显示倒计时
- 仓位管理（Kelly准则简化）
- 止盈止损逻辑

### dashboard.py — 实时仪表盘
- 专业终端UI（ANSI颜色/格式化表格）
- 7大展示区块: 头部状态/账户概览/盈亏统计/持仓明细/模型准确率/纳指对比/最近交易
- 自动刷新（每分钟）
- 离线演示模式（无需网络）

## Bug修复

### LULD frozen dataclass mutable default
- 文件: `utils/constants.py`
- 问题: frozen dataclass中Dict字段使用`= {}`导致ValueError
- 修复: 改用 `field(default_factory=lambda: {...})`
- 同时将TIER1_BANDS和TIER2_BANDS的内联初始化移到default_factory中

## 文件变更
- 新增: `live_trading/__init__.py`
- 新增: `live_trading/market_clock.py`
- 新增: `live_trading/portfolio.py`
- 新增: `live_trading/benchmark.py`
- 新增: `live_trading/accuracy_tracker.py`
- 新增: `live_trading/live_simulator.py`
- 新增: `live_trading/dashboard.py`
- 修改: `main.py` — 更新cmd_live，新增cmd_demo
- 修改: `live_trading/__init__.py` — 导出launch_dashboard
- 修复: `utils/constants.py` — LULD类mutable default修复

## 使用方式
```bash
python main.py demo     # 离线演示模式
python main.py live     # 在线模拟交易（真实行情）
```
