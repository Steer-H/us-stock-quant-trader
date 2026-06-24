# 部署指南 — 從零到運行

## 1. 環境要求

| 項目 | 最低要求 | 推薦 |
|------|---------|------|
| 作業系統 | macOS 12+ | macOS 14+ (Apple Silicon) |
| CPU | x86_64 / ARM64 | Apple M1/M2/M3 |
| 內存 | 8 GB | 16 GB+ |
| Python | 3.9+ | 3.11+ |
| 磁碟 | 2 GB (代碼) + 模型/數據 | 10 GB+ |
| 網絡 | 穩定網際網路連接 | — |

## 2. 新機部署 (5分鐘)

### 2.1 克隆項目

```bash
git clone <repo-url> 美股量化交易
cd 美股量化交易
```

### 2.2 一鍵配置

```bash
bash docs/setup.sh
```

腳本自動完成 8 個步驟：
1. ✅ 檢查 Python 3.9+ (未安裝則提示安裝方法)
2. ✅ 創建虛擬環境 `.venv/`
3. ✅ 安裝所有 Python 依賴 (numpy, pandas, torch, flask, yfinance...)
4. ✅ 檢測 MPS/CUDA 硬體加速
5. ✅ 創建目錄結構 (data/, logs/, models/, work_logs/)
6. ✅ 全項目 Python 語法檢查
7. ✅ 配置 macOS launchd 服務 (可選)
8. ✅ 生成 start.sh / stop.sh / restart.sh

### 2.3 啟動系統

```bash
bash start.sh
```

瀏覽器打開 `http://localhost:8080`

### 2.4 驗證

```bash
curl http://localhost:8080/api/status
```

預期響應包含:
- `"iteration"`: 大於 0
- `"data_quality": {"source": "Yahoo Finance..."}`
- `"ml_ready"`: true/false

## 3. 首次運行說明

### ⚡ 無需任何準備即可運行

系統設計為**零依賴啟動**：
- 沒有 ML 模型 → 自動使用統計預測器 (RealtimePredictor)
- 沒有訓練數據 → 僅影響離線訓練，不影響實時交易
- 沒有情感特徵 → ML 模型自動使用 28 個技術指標

### 🔧 可選增強 (提升預測準確率)

```bash
# 激活虛擬環境
source .venv/bin/activate

# 1. 訓練 ML 模型 (約30分鐘, 強烈推薦)
python3 scripts/quick_train.py

# 2. 構建情感特徵 (約5分鐘, 推薦)
python3 scripts/build_sentiment_features.py
```

訓練完成後重啟服務:
```bash
bash restart.sh
```

## 4. macOS 開機自啟 (launchd)

```bash
# 安裝後自動配置, 手動啟動:
launchctl load ~/Library/LaunchAgents/com.trading.dashboard.plist
launchctl load ~/Library/LaunchAgents/com.trading.watchdog.plist

# 查看狀態
launchctl list | grep trading

# 停止
launchctl unload ~/Library/LaunchAgents/com.trading.dashboard.plist
```

## 5. 手動部署 (不使用 setup.sh)

```bash
# 1. Python 環境
python3 -m venv .venv
source .venv/bin/activate

# 2. 安裝依賴
pip install -r requirements.txt

# 3. 創建目錄
mkdir -p data logs models work_logs

# 4. 啟動
screen -dmS trading python3 -u live_trading/web_server.py
```

## 6. 系統管理

```bash
# 查看實時日誌
screen -r trading
# 按 Ctrl+A 然後 D 退出 screen

# 查看系統狀態
curl -s http://localhost:8080/api/status | python3 -m json.tool | head -20

# 停止系統
bash stop.sh

# 重啟系統
bash restart.sh
```

## 7. 常見問題

### Q: "command not found: python3"
macOS 默認不帶 Python 3。安裝方法:
```bash
# 方法1: Homebrew (推薦)
brew install python@3.11

# 方法2: 官方安裝包
# 訪問 https://www.python.org/downloads/
```

### Q: "No module named 'torch'"
```bash
source .venv/bin/activate
pip install torch
```

### Q: 埠 8080 被佔用
```bash
lsof -ti tcp:8080 | xargs kill -9
```

### Q: Yahoo Finance 無法連接
- 檢查網絡連接 (需要訪問 Yahoo Finance API)
- 等待幾分鐘後重試 (Yahoo 有頻率限制)
- 系統會自動標記數據源狀態並重試

### Q: Apple Silicon MPS 不可用
```bash
# 確認 macOS >= 12.3
sw_vers -productVersion

# 確認 torch 版本 >= 2.0
python -c "import torch; print(torch.__version__)"
```

### Q: 模型加載失敗
- 首次運行無模型是正常現象
- 系統自動回退到統計預測器
- 運行 `python3 scripts/quick_train.py` 訓練模型

## 8. 更新與維護

```bash
# 拉取最新代碼
git pull

# 如有依賴變更
source .venv/bin/activate
pip install -r requirements.txt

# 重啟
bash restart.sh
```

---

> 📖 **返回**: [README.md](README.md) 文檔首頁
