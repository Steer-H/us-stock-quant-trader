#!/bin/bash
# ============================================================
#  每小時自動審查腳本
#  - 僅在美股休市時執行（避免幹擾交易）
#  - 調用 Codex 進行全項目審查
#  - 輸出記錄到 logs/hourly_review.log
# ============================================================
set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG_FILE="${PROJECT_DIR}/logs/hourly_review.log"
REVIEW_LOG_DIR="${PROJECT_DIR}/work_logs"

cd "$PROJECT_DIR" || exit 1

# 確保日誌目錄存在
mkdir -p "$(dirname "$LOG_FILE")"

# ── 檢查市場是否休市 ──
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

echo "[$(date '+%Y-%m-%d %H:%M:%S')] 市場狀態: ${MARKET_STATUS:-UNKNOWN}" >> "$LOG_FILE"

if [ "$MARKET_STATUS" = "OPEN" ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 市場開盤中，跳過審查" >> "$LOG_FILE"
    exit 0
fi

if [ "$MARKET_STATUS" = "EXTENDED" ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 盤前/盤後時段，跳過審查" >> "$LOG_FILE"
    exit 0
fi

# ── 休市：執行審查 ──
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 休市中，開始執行每小時審查..." >> "$LOG_FILE"

REVIEW_PROMPT='你在休市時間執行每小時自動審查。請完成以下任務：

1. 查看 work_logs/ 目錄下最新的審查日誌，了解歷史問題
2. 閱讀 logs/ 目錄下最近的運行日誌（server_stderr.log, server_stdout.log, watchdog.log等），發現異常
3. 重點審查 ml_model/transformer.py 和 live_trading/ 目錄下的交易系統代碼
4. 檢查前端 dashboard.html 和後端 web_server.py 是否有異常
5. 檢查算法是否有可優化的地方，是否有bug
6. 對發現的問題進行修復
7. 將審查結果寫入 work_logs/YYYY-MM-DD_hourly-review-N.md（N為自增序號）
8. 最後確認功能沒有受到影響

請先思考，再行動。保持簡潔。'

# 使用 codex exec 非交互式運行
/Applications/Codex.app/Contents/Resources/codex exec "$REVIEW_PROMPT" \
    --config 'approval_policy="on-failure"' \
    >> "$LOG_FILE" 2>&1

EXIT_CODE=$?
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 審查完成 (exit=${EXIT_CODE})" >> "$LOG_FILE"
echo "---" >> "$LOG_FILE"

exit $EXIT_CODE
