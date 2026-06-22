# 部署指南 — 从零到运行

## 1. 环境要求

| 项目 | 最低要求 | 推荐 |
|------|---------|------|
| 操作系统 | macOS 12+ | macOS 14+ (Apple Silicon) |
| CPU | x86_64 / ARM64 | Apple M1/M2/M3 |
| 内存 | 8 GB | 16 GB+ |
| Python | 3.9+ | 3.11+ |
| 磁盘 | 2 GB (代码) + 模型/数据 | 10 GB+ |
| 网络 | 稳定互联网连接 | — |

## 2. 新机部署 (5分钟)

### 2.1 克隆项目

```bash
git clone <repo-url> 美股量化交易
cd 美股量化交易
```

### 2.2 一键配置

```bash
bash docs/setup.sh
```

脚本自动完成 8 个步骤：
1. ✅ 检查 Python 3.9+ (未安装则提示安装方法)
2. ✅ 创建虚拟环境 `.venv/`
3. ✅ 安装所有 Python 依赖 (numpy, pandas, torch, flask, yfinance...)
4. ✅ 检测 MPS/CUDA 硬件加速
5. ✅ 创建目录结构 (data/, logs/, models/, work_logs/)
6. ✅ 全项目 Python 语法检查
7. ✅ 配置 macOS launchd 服务 (可选)
8. ✅ 生成 start.sh / stop.sh / restart.sh

### 2.3 启动系统

```bash
bash start.sh
```

浏览器打开 `http://localhost:8080`

### 2.4 验证

```bash
curl http://localhost:8080/api/status
```

预期响应包含:
- `"iteration"`: 大于 0
- `"data_quality": {"source": "Yahoo Finance..."}`
- `"ml_ready"`: true/false

## 3. 首次运行说明

### ⚡ 无需任何准备即可运行

系统设计为**零依赖启动**：
- 没有 ML 模型 → 自动使用统计预测器 (RealtimePredictor)
- 没有训练数据 → 仅影响离线训练，不影响实时交易
- 没有情感特征 → ML 模型自动使用 28 个技术指标

### 🔧 可选增强 (提升预测准确率)

```bash
# 激活虚拟环境
source .venv/bin/activate

# 1. 训练 ML 模型 (约30分钟, 强烈推荐)
python3 scripts/quick_train.py

# 2. 构建情感特征 (约5分钟, 推荐)
python3 scripts/build_sentiment_features.py
```

训练完成后重启服务:
```bash
bash restart.sh
```

## 4. macOS 开机自启 (launchd)

```bash
# 安装后自动配置, 手动启动:
launchctl load ~/Library/LaunchAgents/com.trading.dashboard.plist
launchctl load ~/Library/LaunchAgents/com.trading.watchdog.plist

# 查看状态
launchctl list | grep trading

# 停止
launchctl unload ~/Library/LaunchAgents/com.trading.dashboard.plist
```

## 5. 手动部署 (不使用 setup.sh)

```bash
# 1. Python 环境
python3 -m venv .venv
source .venv/bin/activate

# 2. 安装依赖
pip install -r requirements.txt

# 3. 创建目录
mkdir -p data logs models work_logs

# 4. 启动
screen -dmS trading python3 -u live_trading/web_server.py
```

## 6. 系统管理

```bash
# 查看实时日志
screen -r trading
# 按 Ctrl+A 然后 D 退出 screen

# 查看系统状态
curl -s http://localhost:8080/api/status | python3 -m json.tool | head -20

# 停止系统
bash stop.sh

# 重启系统
bash restart.sh
```

## 7. 常见问题

### Q: "command not found: python3"
macOS 默认不带 Python 3。安装方法:
```bash
# 方法1: Homebrew (推荐)
brew install python@3.11

# 方法2: 官方安装包
# 访问 https://www.python.org/downloads/
```

### Q: "No module named 'torch'"
```bash
source .venv/bin/activate
pip install torch
```

### Q: 端口 8080 被占用
```bash
lsof -ti tcp:8080 | xargs kill -9
```

### Q: Yahoo Finance 无法连接
- 检查网络连接 (需要访问 Yahoo Finance API)
- 等待几分钟后重试 (Yahoo 有频率限制)
- 系统会自动标记数据源状态并重试

### Q: Apple Silicon MPS 不可用
```bash
# 确认 macOS >= 12.3
sw_vers -productVersion

# 确认 torch 版本 >= 2.0
python -c "import torch; print(torch.__version__)"
```

### Q: 模型加载失败
- 首次运行无模型是正常现象
- 系统自动回退到统计预测器
- 运行 `python3 scripts/quick_train.py` 训练模型

## 8. 更新与维护

```bash
# 拉取最新代码
git pull

# 如有依赖变更
source .venv/bin/activate
pip install -r requirements.txt

# 重启
bash restart.sh
```

---

> 📖 **返回**: [README.md](README.md) 文档首页
