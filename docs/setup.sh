#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# 美股量化交易系統 — 一鍵環境配置腳本 v2.0
# ═══════════════════════════════════════════════════════════════
# 用法: bash docs/setup.sh
# 支持: macOS 12+ (Apple Silicon / Intel)
# ═══════════════════════════════════════════════════════════════

set -e
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'

# 自動檢測項目根目錄
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

echo ""
echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║     美股量化交易系統 — 一鍵環境配置 v2.0                     ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
echo -e "  ${BLUE}項目路徑:${NC} $PROJECT_DIR"
echo ""

# ═══════════════════════════════════════════════════════════════
# 步驟1: 檢查 Python
# ═══════════════════════════════════════════════════════════════
echo -e "${YELLOW}[1/8]${NC} 檢查 Python 環境..."

PYTHON=""
for py in python3.12 python3.11 python3.10 python3.9 python3; do
    if command -v $py &>/dev/null; then
        VER=$($py -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null)
        MAJOR=$(echo $VER | cut -d. -f1)
        MINOR=$(echo $VER | cut -d. -f2)
        if [ -n "$MAJOR" ] && [ "$MAJOR" -ge 3 ] && [ "$MINOR" -ge 9 ] 2>/dev/null; then
            PYTHON=$py
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    echo -e "${RED}❌ 未找到 Python 3.9+${NC}"
    echo ""
    echo -e "  ${YELLOW}請先安裝 Python:${NC}"
    echo -e "  ${BLUE}macOS (推薦):${NC}"
    echo "    brew install python@3.11"
    echo ""
    echo -e "  ${BLUE}或下載官方安裝包:${NC}"
    echo "    https://www.python.org/downloads/"
    echo ""
    echo -e "  ${BLUE}如果沒有 Homebrew, 先安裝:${NC}"
    echo '    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
    exit 1
fi
echo -e "   ${GREEN}✅${NC} $PYTHON ($($PYTHON --version))"

# 檢查 pip
if ! $PYTHON -m pip --version &>/dev/null; then
    echo -e "   ${YELLOW}⚠️${NC}  pip 未安裝，正在安裝..."
    $PYTHON -m ensurepip --upgrade 2>/dev/null || true
fi

# ═══════════════════════════════════════════════════════════════
# 步驟2: 創建虛擬環境
# ═══════════════════════════════════════════════════════════════
echo -e "${YELLOW}[2/8]${NC} 創建虛擬環境..."

if [ -d ".venv" ]; then
    echo -e "   ${BLUE}ℹ️${NC}  虛擬環境已存在，跳過創建"
else
    $PYTHON -m venv .venv
    echo -e "   ${GREEN}✅${NC} 虛擬環境已創建: .venv/"
fi

# 激活虛擬環境
source .venv/bin/activate
python -m pip install --upgrade pip -q 2>&1 | tail -1
echo -e "   ${GREEN}✅${NC} pip 已升級到 $(pip --version | cut -d' ' -f2)"

# ═══════════════════════════════════════════════════════════════
# 步驟3: 安裝依賴
# ═══════════════════════════════════════════════════════════════
echo -e "${YELLOW}[3/8]${NC} 安裝 Python 依賴 (可能需要幾分鐘)..."

if [ -f "requirements.txt" ]; then
    # 先安裝核心依賴
    echo -e "   ${BLUE}→${NC} 安裝 numpy, pandas, torch..."
    pip install numpy pandas torch --quiet 2>&1 | tail -1
    
    # 再安裝其餘依賴
    echo -e "   ${BLUE}→${NC} 安裝其餘依賴..."
    pip install -r requirements.txt --quiet 2>&1 | tail -3
    
    echo -e "   ${GREEN}✅${NC} 依賴安裝完成"
    
    # 驗證關鍵包
    python -c "
import numpy, pandas, torch, yfinance, flask
print(f'   numpy={numpy.__version__}  pandas={pandas.__version__}  torch={torch.__version__}')
" 2>/dev/null && echo -e "   ${GREEN}✅${NC} 關鍵包驗證通過" || echo -e "   ${YELLOW}⚠️${NC} 部分包可能未正確安裝"
else
    echo -e "   ${RED}❌${NC} requirements.txt 不存在"
    exit 1
fi

# ═══════════════════════════════════════════════════════════════
# 步驟4: 檢查 MPS (Apple Silicon GPU)
# ═══════════════════════════════════════════════════════════════
echo -e "${YELLOW}[4/8]${NC} 檢查硬體加速..."

ARCH=$(uname -m)
python -c "
import torch
if torch.backends.mps.is_available():
    print('   ✅ MPS (Apple Silicon GPU) 可用 — 訓練將使用GPU加速')
elif torch.cuda.is_available():
    print('   ✅ CUDA GPU 可用')
else:
    print('   ℹ️  僅 CPU 可用 — 訓練較慢但功能完整')
" 2>/dev/null || echo -e "   ${YELLOW}⚠️${NC} torch 未正確安裝，請檢查"

# ═══════════════════════════════════════════════════════════════
# 步驟5: 創建目錄
# ═══════════════════════════════════════════════════════════════
echo -e "${YELLOW}[5/8]${NC} 創建運行目錄..."

mkdir -p data logs models work_logs logs/archive data/processed
echo -e "   ${GREEN}✅${NC} 目錄結構就緒"

# ═══════════════════════════════════════════════════════════════
# 步驟6: 語法檢查
# ═══════════════════════════════════════════════════════════════
echo -e "${YELLOW}[6/8]${NC} 語法檢查..."

ERRORS=0
for f in $(find live_trading ml_model config utils backtesting -name "*.py" 2>/dev/null); do
    if ! python -c "import ast; ast.parse(open('$f').read())" 2>/dev/null; then
        echo -e "   ${RED}❌${NC} $f"
        ERRORS=$((ERRORS+1))
    fi
done
if [ $ERRORS -eq 0 ]; then
    echo -e "   ${GREEN}✅${NC} 所有 Python 文件語法正確"
else
    echo -e "   ${RED}⚠️${NC}  $ERRORS 個文件有語法錯誤，請檢查"
fi

# ═══════════════════════════════════════════════════════════════
# 步驟7: 安裝 launchd 服務 (僅 macOS)
# ═══════════════════════════════════════════════════════════════
echo -e "${YELLOW}[7/8]${NC} 配置系統服務..."

if [[ "$OSTYPE" == "darwin"* ]]; then
    LAUNCH_DIR="$HOME/Library/LaunchAgents"
    mkdir -p "$LAUNCH_DIR"
    
    INSTALLED=0
    for plist in com.trading.*.plist; do
        if [ -f "$plist" ]; then
            # 用 Python 精確替換 XML 中的硬編碼路徑
            python3 -c "
import re
with open('$plist', 'r') as f:
    c = f.read()
# 替換所有 /Users/xxx/... 路徑為當前項目路徑
c = re.sub(r'/Users/[^<]*/美股量化交易(?:（New）)?', '$PROJECT_DIR', c)
c = re.sub(r'/Users/[^<]*/Documents/美股量化交易(?:（New）)?', '$PROJECT_DIR', c)
# 也替換單獨的 /Users/xxx (home目錄引用)
c = re.sub(r'(?<=<string>)/Users/\w+(?=</string>)', '$HOME', c)
with open('$plist', 'w') as f:
    f.write(c)
"
            cp "$plist" "$LAUNCH_DIR/"
            echo -e "   ${GREEN}✅${NC} 已安裝: $plist → $LAUNCH_DIR/"
            INSTALLED=$((INSTALLED+1))
        fi
    done
    
    if [ $INSTALLED -gt 0 ]; then
        echo ""
        echo -e "   ${BLUE}💡 啟動服務:${NC}"
        echo "      launchctl load ~/Library/LaunchAgents/com.trading.dashboard.plist"
        echo "      launchctl load ~/Library/LaunchAgents/com.trading.watchdog.plist"
        echo ""
        echo -e "   ${BLUE}💡 管理服務:${NC}"
        echo "      launchctl list | grep trading    # 查看狀態"
        echo "      launchctl unload ~/Library/LaunchAgents/com.trading.dashboard.plist  # 停止"
    fi
else
    echo -e "   ${BLUE}ℹ️${NC}  非 macOS，跳過 launchd (請使用 systemd/cron)"
fi

# ═══════════════════════════════════════════════════════════════
# 步驟8: 生成便捷腳本
# ═══════════════════════════════════════════════════════════════
echo -e "${YELLOW}[8/8]${NC} 生成便捷腳本..."

cat > start.sh << 'STARTEOF'
#!/bin/bash
cd "$(dirname "$0")"
source .venv/bin/activate 2>/dev/null || { echo "❌ 請先運行: bash docs/setup.sh"; exit 1; }
lsof -ti tcp:8080 | xargs kill -9 2>/dev/null
screen -S trading -X quit 2>/dev/null
sleep 1
screen -dmS trading python3 -u live_trading/web_server.py
sleep 5
curl -s http://localhost:8080/api/status 2>/dev/null | python3 -c "
import sys,json
d=json.load(sys.stdin)
src=d.get('data_quality',{}).get('source','?')
print(f'✅ 系統已啟動 | 迭代#{d[\"iteration\"]} | {src}')
" 2>/dev/null || echo "⚠️  系統正在啟動中，請稍候訪問 http://localhost:8080"
echo "🌐 儀錶盤: http://localhost:8080"
echo "📋 查看日誌: screen -r trading"
echo "🛑 停止系統: bash stop.sh"
STARTEOF
chmod +x start.sh

cat > stop.sh << 'STOPEOF'
#!/bin/bash
lsof -ti tcp:8080 | xargs kill -9 2>/dev/null
screen -S trading -X quit 2>/dev/null
echo "✅ 系統已停止"
STOPEOF
chmod +x stop.sh

cat > restart.sh << 'RESTARTEOF'
#!/bin/bash
cd "$(dirname "$0")"
bash stop.sh
sleep 2
bash start.sh
RESTARTEOF
chmod +x restart.sh

echo -e "   ${GREEN}✅${NC} 便捷腳本已生成: start.sh stop.sh restart.sh"

# ═══════════════════════════════════════════════════════════════
# 完成
# ═══════════════════════════════════════════════════════════════
echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  🎉 環境配置完成！                                         ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "  ${BLUE}啟動系統:${NC}"
echo -e "    bash start.sh"
echo ""
echo -e "  ${BLUE}儀錶盤:${NC}     http://localhost:8080"
echo -e "  ${BLUE}完整文檔:${NC}   docs/README.md"
echo -e "  ${BLUE}API文檔:${NC}    docs/API_REFERENCE.md"
echo ""
echo -e "  ${BLUE}ℹ️  首次運行說明:${NC}"
echo -e "  • 系統使用統計預測器 (無需ML模型即可運行)"
echo -e "  • ML模型訓練: python3 scripts/quick_train.py (可選, 耗時約30分鐘)"
echo -e "  • 情感特徵: python3 scripts/build_sentiment_features.py (可選)"
echo ""
echo -e "  ${BLUE}⚠️  注意:${NC}"
echo -e "  • 美股交易時間: 美東 09:30-16:00 (北京時間 21:30-04:00 夏令時)"
echo -e "  • 休市時系統正常運行但不執行交易"
echo -e "  • 如需開機自啟，運行上述 launchctl 命令"
echo ""
