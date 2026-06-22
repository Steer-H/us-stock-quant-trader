#!/bin/bash
# ============================================================
#  每小时自动审查脚本
#  - 仅在美股休市时执行（避免干扰交易）
#  - 调用 Codex 进行全项目审查
#  - 输出记录到 logs/hourly_review.log
# ============================================================
set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG_FILE="${PROJECT_DIR}/logs/hourly_review.log"
REVIEW_LOG_DIR="${PROJECT_DIR}/work_logs"

cd "$PROJECT_DIR" || exit 1

# 确保日志目录存在
mkdir -p "$(dirname "$LOG_FILE")"

# ── 检查市场是否休市 ──
MARKET_STATUS=$(/usr/bin/python3 -c "
import sys
sys.path.insert(0, '${PROJECT_DIR}')
from live_trading.market_clock import MarketClock, MarketStatus
clock = MarketClock()
status, desc = clock.get_status()
if status == MarketStatus.REGULAR_HOURS:
    print('OPEN')
elif status in (MarketStatus.PRE_MARKET, MarketStatus.AFTER_HOURS):
    print('EXTENDED')
else:
    print('CLOSED')
" 2>/dev/null)

echo "[$(date '+%Y-%m-%d %H:%M:%S')] 市场状态: ${MARKET_STATUS:-UNKNOWN}" >> "$LOG_FILE"

if [ "$MARKET_STATUS" = "OPEN" ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 市场开盘中，跳过审查" >> "$LOG_FILE"
    exit 0
fi

if [ "$MARKET_STATUS" = "EXTENDED" ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 盘前/盘后时段，跳过审查" >> "$LOG_FILE"
    exit 0
fi

# ── 休市：执行审查 ──
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 休市中，开始执行每小时审查..." >> "$LOG_FILE"

REVIEW_PROMPT='你在休市时间执行每小时自动审查。请完成以下任务：

1. 查看 work_logs/ 目录下最新的审查日志，了解历史问题
2. 阅读 logs/ 目录下最近的运行日志（server_stderr.log, server_stdout.log, watchdog.log等），发现异常
3. 重点审查 ml_model/transformer.py 和 live_trading/ 目录下的交易系统代码
4. 检查前端 dashboard.html 和后端 web_server.py 是否有异常
5. 检查算法是否有可优化的地方，是否有bug
6. 对发现的问题进行修复
7. 将审查结果写入 work_logs/YYYY-MM-DD_hourly-review-N.md（N为自增序号）
8. 最后确认功能没有受到影响

请先思考，再行动。保持简洁。'

# 使用 codex exec 非交互式运行
/Applications/Codex.app/Contents/Resources/codex exec "$REVIEW_PROMPT" \
    --config 'approval_policy="on-failure"' \
    >> "$LOG_FILE" 2>&1

EXIT_CODE=$?
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 审查完成 (exit=${EXIT_CODE})" >> "$LOG_FILE"
echo "---" >> "$LOG_FILE"

exit $EXIT_CODE
