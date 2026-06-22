#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# 美股量化交易系统 — 一键环境配置脚本 v2.0
# ═══════════════════════════════════════════════════════════════
# 用法: bash docs/setup.sh
# 支持: macOS 12+ (Apple Silicon / Intel)
# ═══════════════════════════════════════════════════════════════

set -e
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'

# 自动检测项目根目录
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

echo ""
echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║     美股量化交易系统 — 一键环境配置 v2.0                     ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
echo -e "  ${BLUE}项目路径:${NC} $PROJECT_DIR"
echo ""

# ═══════════════════════════════════════════════════════════════
# 步骤1: 检查 Python
# ═══════════════════════════════════════════════════════════════
echo -e "${YELLOW}[1/8]${NC} 检查 Python 环境..."

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
    echo -e "  ${YELLOW}请先安装 Python:${NC}"
    echo -e "  ${BLUE}macOS (推荐):${NC}"
    echo "    brew install python@3.11"
    echo ""
    echo -e "  ${BLUE}或下载官方安装包:${NC}"
    echo "    https://www.python.org/downloads/"
    echo ""
    echo -e "  ${BLUE}如果没有 Homebrew, 先安装:${NC}"
    echo '    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
    exit 1
fi
echo -e "   ${GREEN}✅${NC} $PYTHON ($($PYTHON --version))"

# 检查 pip
if ! $PYTHON -m pip --version &>/dev/null; then
    echo -e "   ${YELLOW}⚠️${NC}  pip 未安装，正在安装..."
    $PYTHON -m ensurepip --upgrade 2>/dev/null || true
fi

# ═══════════════════════════════════════════════════════════════
# 步骤2: 创建虚拟环境
# ═══════════════════════════════════════════════════════════════
echo -e "${YELLOW}[2/8]${NC} 创建虚拟环境..."

if [ -d ".venv" ]; then
    echo -e "   ${BLUE}ℹ️${NC}  虚拟环境已存在，跳过创建"
else
    $PYTHON -m venv .venv
    echo -e "   ${GREEN}✅${NC} 虚拟环境已创建: .venv/"
fi

# 激活虚拟环境
source .venv/bin/activate
python -m pip install --upgrade pip -q 2>&1 | tail -1
echo -e "   ${GREEN}✅${NC} pip 已升级到 $(pip --version | cut -d' ' -f2)"

# ═══════════════════════════════════════════════════════════════
# 步骤3: 安装依赖
# ═══════════════════════════════════════════════════════════════
echo -e "${YELLOW}[3/8]${NC} 安装 Python 依赖 (可能需要几分钟)..."

if [ -f "requirements.txt" ]; then
    # 先安装核心依赖
    echo -e "   ${BLUE}→${NC} 安装 numpy, pandas, torch..."
    pip install numpy pandas torch --quiet 2>&1 | tail -1
    
    # 再安装其余依赖
    echo -e "   ${BLUE}→${NC} 安装其余依赖..."
    pip install -r requirements.txt --quiet 2>&1 | tail -3
    
    echo -e "   ${GREEN}✅${NC} 依赖安装完成"
    
    # 验证关键包
    python -c "
import numpy, pandas, torch, yfinance, flask
print(f'   numpy={numpy.__version__}  pandas={pandas.__version__}  torch={torch.__version__}')
" 2>/dev/null && echo -e "   ${GREEN}✅${NC} 关键包验证通过" || echo -e "   ${YELLOW}⚠️${NC} 部分包可能未正确安装"
else
    echo -e "   ${RED}❌${NC} requirements.txt 不存在"
    exit 1
fi

# ═══════════════════════════════════════════════════════════════
# 步骤4: 检查 MPS (Apple Silicon GPU)
# ═══════════════════════════════════════════════════════════════
echo -e "${YELLOW}[4/8]${NC} 检查硬件加速..."

ARCH=$(uname -m)
python -c "
import torch
if torch.backends.mps.is_available():
    print('   ✅ MPS (Apple Silicon GPU) 可用 — 训练将使用GPU加速')
elif torch.cuda.is_available():
    print('   ✅ CUDA GPU 可用')
else:
    print('   ℹ️  仅 CPU 可用 — 训练较慢但功能完整')
" 2>/dev/null || echo -e "   ${YELLOW}⚠️${NC} torch 未正确安装，请检查"

# ═══════════════════════════════════════════════════════════════
# 步骤5: 创建目录
# ═══════════════════════════════════════════════════════════════
echo -e "${YELLOW}[5/8]${NC} 创建运行目录..."

mkdir -p data logs models work_logs logs/archive data/processed
echo -e "   ${GREEN}✅${NC} 目录结构就绪"

# ═══════════════════════════════════════════════════════════════
# 步骤6: 语法检查
# ═══════════════════════════════════════════════════════════════
echo -e "${YELLOW}[6/8]${NC} 语法检查..."

ERRORS=0
for f in $(find live_trading ml_model config utils backtesting -name "*.py" 2>/dev/null); do
    if ! python -c "import ast; ast.parse(open('$f').read())" 2>/dev/null; then
        echo -e "   ${RED}❌${NC} $f"
        ERRORS=$((ERRORS+1))
    fi
done
if [ $ERRORS -eq 0 ]; then
    echo -e "   ${GREEN}✅${NC} 所有 Python 文件语法正确"
else
    echo -e "   ${RED}⚠️${NC}  $ERRORS 个文件有语法错误，请检查"
fi

# ═══════════════════════════════════════════════════════════════
# 步骤7: 安装 launchd 服务 (仅 macOS)
# ═══════════════════════════════════════════════════════════════
echo -e "${YELLOW}[7/8]${NC} 配置系统服务..."

if [[ "$OSTYPE" == "darwin"* ]]; then
    LAUNCH_DIR="$HOME/Library/LaunchAgents"
    mkdir -p "$LAUNCH_DIR"
    
    INSTALLED=0
    for plist in com.trading.*.plist; do
        if [ -f "$plist" ]; then
            # 用 Python 精确替换 XML 中的硬编码路径
            python3 -c "
import re
with open('$plist', 'r') as f:
    c = f.read()
# 替换所有 /Users/xxx/... 路径为当前项目路径
c = re.sub(r'/Users/[^<]*/美股量化交易(?:（New）)?', '$PROJECT_DIR', c)
c = re.sub(r'/Users/[^<]*/Documents/美股量化交易(?:（New）)?', '$PROJECT_DIR', c)
# 也替换单独的 /Users/xxx (home目录引用)
c = re.sub(r'(?<=<string>)/Users/\w+(?=</string>)', '$HOME', c)
with open('$plist', 'w') as f:
    f.write(c)
"
            cp "$plist" "$LAUNCH_DIR/"
            echo -e "   ${GREEN}✅${NC} 已安装: $plist → $LAUNCH_DIR/"
            INSTALLED=$((INSTALLED+1))
        fi
    done
    
    if [ $INSTALLED -gt 0 ]; then
        echo ""
        echo -e "   ${BLUE}💡 启动服务:${NC}"
        echo "      launchctl load ~/Library/LaunchAgents/com.trading.dashboard.plist"
        echo "      launchctl load ~/Library/LaunchAgents/com.trading.watchdog.plist"
        echo ""
        echo -e "   ${BLUE}💡 管理服务:${NC}"
        echo "      launchctl list | grep trading    # 查看状态"
        echo "      launchctl unload ~/Library/LaunchAgents/com.trading.dashboard.plist  # 停止"
    fi
else
    echo -e "   ${BLUE}ℹ️${NC}  非 macOS，跳过 launchd (请使用 systemd/cron)"
fi

# ═══════════════════════════════════════════════════════════════
# 步骤8: 生成便捷脚本
# ═══════════════════════════════════════════════════════════════
echo -e "${YELLOW}[8/8]${NC} 生成便捷脚本..."

cat > start.sh << 'STARTEOF'
#!/bin/bash
cd "$(dirname "$0")"
source .venv/bin/activate 2>/dev/null || { echo "❌ 请先运行: bash docs/setup.sh"; exit 1; }
lsof -ti tcp:8080 | xargs kill -9 2>/dev/null
screen -S trading -X quit 2>/dev/null
sleep 1
screen -dmS trading python3 -u live_trading/web_server.py
sleep 5
curl -s http://localhost:8080/api/status 2>/dev/null | python3 -c "
import sys,json
d=json.load(sys.stdin)
src=d.get('data_quality',{}).get('source','?')
print(f'✅ 系统已启动 | 迭代#{d[\"iteration\"]} | {src}')
" 2>/dev/null || echo "⚠️  系统正在启动中，请稍候访问 http://localhost:8080"
echo "🌐 仪表盘: http://localhost:8080"
echo "📋 查看日志: screen -r trading"
echo "🛑 停止系统: bash stop.sh"
STARTEOF
chmod +x start.sh

cat > stop.sh << 'STOPEOF'
#!/bin/bash
lsof -ti tcp:8080 | xargs kill -9 2>/dev/null
screen -S trading -X quit 2>/dev/null
echo "✅ 系统已停止"
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

echo -e "   ${GREEN}✅${NC} 便捷脚本已生成: start.sh stop.sh restart.sh"

# ═══════════════════════════════════════════════════════════════
# 完成
# ═══════════════════════════════════════════════════════════════
echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  🎉 环境配置完成！                                         ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "  ${BLUE}启动系统:${NC}"
echo -e "    bash start.sh"
echo ""
echo -e "  ${BLUE}仪表盘:${NC}     http://localhost:8080"
echo -e "  ${BLUE}完整文档:${NC}   docs/README.md"
echo -e "  ${BLUE}API文档:${NC}    docs/API_REFERENCE.md"
echo ""
echo -e "  ${BLUE}ℹ️  首次运行说明:${NC}"
echo -e "  • 系统使用统计预测器 (无需ML模型即可运行)"
echo -e "  • ML模型训练: python3 scripts/quick_train.py (可选, 耗时约30分钟)"
echo -e "  • 情感特征: python3 scripts/build_sentiment_features.py (可选)"
echo ""
echo -e "  ${BLUE}⚠️  注意:${NC}"
echo -e "  • 美股交易时间: 美东 09:30-16:00 (北京时间 21:30-04:00 夏令时)"
echo -e "  • 休市时系统正常运行但不执行交易"
echo -e "  • 如需开机自启，运行上述 launchctl 命令"
echo ""
