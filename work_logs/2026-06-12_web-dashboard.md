# 工作日志 - 2026-06-12 Web仪表盘

## 概述
将终端版交易面板升级为 Web 网页版仪表盘，通过浏览器即可实时观察交易状态。

## 新增/修改文件

### live_trading/web_server.py (新增)
- Flask Web 服务器，提供 RESTful API
- `GET /` — 仪表盘HTML页面
- `GET /api/status` — 完整系统状态 JSON API
- `GET /api/refresh` — 手动刷新引擎
- 后台线程每分钟自动执行模拟交易
- 引擎与前端完全解耦，API可独立调用

### live_trading/templates/dashboard.html (新增)
- 暗色主题专业交易面板（GitHub风格配色）
- 响应式布局（桌面/平板自适应）
- JavaScript 自动轮询（60秒刷新）
- 7大展示区块：
  1. 顶部状态栏（市场状态+倒计时）
  2. 账户概览（总净资产/总盈亏/模型准确率 三大卡片）
  3. 持仓明细表格（代码/数量/成本/现价/市值/盈亏/权重）
  4. 账户详情（现金/市值/已实现/未实现/当日/回撤/杠杆）
  5. 纳指基准对比（可视化柱状图+Alpha/Beta/信息比率）
  6. 准确率详情（涨跌分别统计/趋势/状态）
  7. 最近交易列表

### main.py (修改)
- 新增 `python main.py web` 命令
- 新增 `cmd_web()` 函数

### requirements.txt (依赖)
- flask>=3.0 已安装

## 技术细节
- 端口: 8080 (避免macOS AirPlay占用5000)
- 刷新频率: 60秒自动轮询
- 所有数值实时计算，支持正负颜色标记
- 圆形准确率仪表盘（绿色>=55%, 黄色50-55%, 红色<50%）

## 使用方式
```bash
python main.py web
# 浏览器打开 http://localhost:8080
```

## 更新 (同日)

### 改动1: 交易记录添加时间戳
- `web_server.py`: API 返回的 `recent_trades` 新增 `time` 字段（HH:MM:SS格式）
- `dashboard.html`: 交易列表每行右侧显示成交时间

### 改动2: 刷新频率改为每秒
- `web_server.py`: 后台引擎循环从 60s 改为 1s
- `dashboard.html`: JavaScript `setInterval` 从 60000ms 改为 1000ms
- 价格波动幅度从 ±0.5%/60s 调整为 ±0.1%/1s（保持波动率一致）
- 交易频率从每5轮改为每300轮（约5分钟）
- 预测频率从每3轮改为每180轮（约3分钟）
- 页面标题显示 "⚡ 实时" 和实时刷新编号

### 改动3: 开盘后才开始交易
- `web_server.py`: 新增 `_positions_initialized` 和 `_market_opened` 标志
- `tick_engine()`: 检测到市场状态变为 REGULAR_HOURS 时才执行 `init_positions()` 建仓
- API 返回 `waiting_for_open` 和 `positions_initialized` 字段
- `dashboard.html`: 开盘前显示大倒计时横幅（"等待美股开盘"），隐藏主面板
- 所有之前交易数据已清空，从零开始

### Bug修复: 盘前倒计时为空
- `market_clock.py`: `get_trading_session_info()` 中盘前/盘后/闭市都提供 `countdown_to_open`
- `web_server.py`: 倒计时逻辑简化为"正常盘→距闭市，其他→距开市"

### 改动4: 等待阶段也显示完整面板
- `dashboard.html`: 移除等待时隐藏主面板的逻辑
- 等待横幅改为紧凑横条（不遮挡下方数据）
- 始终显示账户概览/盈亏/持仓/纳指对比/准确率面板
- 建仓前：显示 $100,000 现金、0 持仓、完整对比数据
- 建仓后：横幅消失，面板正常显示实时数据

### 改动5: 进程守护系统

**新增文件:**
- `live_trading/daemon.py` — Python守护进程
- `live_trading/watchdog.py` — 完整Watchdog（健康检查+自动重启+内存监控）
- `live_trading/keepalive.sh` — Bash自动重启脚本
- `run_server.sh` — 简化版守护脚本
- `com.trading.dashboard.plist` — macOS launchd 配置

**修改:**
- `web_server.py` — 新增 `/api/health` 端点
- `start.sh` — 完整重写，支持 start/stop/restart/status

**启动方式:**
```bash
cd /Users/oujianli/Documents/美股量化交易（New）
bash start.sh start     # 启动（nohup 后台运行）
bash start.sh status    # 查看状态
bash start.sh stop      # 停止
bash start.sh restart   # 重启
```

**注意事项:** 由于Codex CLI会话管理机制，通过工具启动的后台进程会在命令结束后被清理。
用户需在自己的终端中执行 `bash start.sh start` 来持久运行。
