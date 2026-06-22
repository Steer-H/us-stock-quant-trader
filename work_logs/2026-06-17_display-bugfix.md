# 2026-06-17 交易系统显示问题修复

## 审计范围
前端Dashboard HTML + 后端API数据 + 数据流完整性

---

## 一、发现的显示/逻辑bug

### 1.1 准确率显示 "0.0%" (Misleading)
- **文件**: `live_trading/templates/dashboard.html:566-568`
- **问题**: 当预测总数为0时（系统刚启动），方向准确率显示 "0.0%"，误导用户以为准确率很差
- **修复**: 预测数>0时显示百分比，否则显示 "--"
- **影响**: 分析面板的"方向准确率"、"最近50次"、"做多/做空准确"

### 1.2 成交面板缺少做空标签 (Missing Feature)
- **文件**: `live_trading/templates/dashboard.html:554`
- **问题**: 已成交列表只支持"买入"/"卖出"标签，做空交易显示为"卖出"
- **修复**: 添加 SHORT → "做空" 映射 + short CSS样式
- **对比**: 总览面板的最近交易已正确支持SHORT，成交面板遗漏

### 1.3 最大回撤显示 "+0.00%" (Misleading)
- **文件**: `live_trading/templates/dashboard.html:505-506`
- **问题**: fp()函数对非负值添加"+"前缀，新账户最大回撤0%显示为"+0.00%"
- **修复**: 回撤单独处理，0值时显示"0.00%"，颜色判定也修正
- **关联bug**: cls(-max_drawdown) 对正值显示绿色——回撤越大越"绿"，完全反向

### 1.4 初始资金硬编码 (Data Integrity)
- **文件**: `live_trading/web_server.py:975`
- **问题**: API返回的 `initial_capital` 硬编码为 `100000.0`
- **修复**: 改为 `_portfolio.initial_capital`（从状态恢复后可能不同）

### 1.5 基准数据为空时的显示
- **文件**: `live_trading/templates/dashboard.html:581-586`
- **问题**: 纳指价格为0时（系统新启动），策略/纳指收益显示 "+0.00%"
- **修复**: 纳指价格>0时显示百分比，否则显示 "--"

### 1.6 准确率状态为空时显示
- **文件**: `live_trading/templates/dashboard.html:577-578`
- **问题**: 无预测时仍显示 "✅ 可接受" 或 "⚠️ 需调优"
- **修复**: 无预测时显示 "⏳ 待数据"

---

## 二、文件变更

| 文件 | 变更 |
|------|------|
| `live_trading/templates/dashboard.html` | 6处显示bug修复 |
| `live_trading/web_server.py` | initial_capital动态化 |

## 三、验证结果

- 后端语法: ✅
- 前端HTML: ✅ 所有修复模式已生效
- 服务器状态: 🟢 运行正常
- 数据新鲜度: 🟢 过期=False
