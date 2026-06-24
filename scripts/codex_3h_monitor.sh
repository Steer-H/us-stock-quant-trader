#!/bin/bash
# ============================================================================
# Codex 3小時粗略巡視腳本
# 每3小時運行一次，快速掃描系統關鍵指標，生成簡潔狀態報告
# ============================================================================

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="${PROJECT_DIR}/logs"
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
LOG_FILE="${LOG_DIR}/codex_3h_monitor.log"

cd "$PROJECT_DIR" || exit 1

mkdir -p "$LOG_DIR"

# ------------------------------------------------------------------
# 運行監控腳本，捕獲 JSON 輸出
# ------------------------------------------------------------------
JSON_OUTPUT=$(/usr/bin/python3 scripts/codex_monitor.py --json-only 2>/dev/null)
EXIT_CODE=$?

# ------------------------------------------------------------------
# 從 JSON 提取關鍵欄位
# ------------------------------------------------------------------
OVERALL=$(echo "$JSON_OUTPUT" | /usr/bin/python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    status = d.get('overall_status', 'UNKNOWN')
    market = d.get('market_status', '?')
    proc_alive = d['checks']['server_process']['alive']
    health_ok = d['checks']['health_api']['ok']
    anomalies = len(d.get('anomalies', []))
    criticals = sum(1 for a in d.get('anomalies', []) if a.get('severity') == 'CRITICAL')
    warnings = sum(1 for a in d.get('anomalies', []) if a.get('severity') == 'WARNING')
    print(f'{status}|{market}|{proc_alive}|{health_ok}|{anomalies}|{criticals}|{warnings}')
except Exception as e:
    print(f'ERROR|?|False|False|0|0|0')
")

IFS='|' read -r STATUS MARKET PROC_ALIVE HEALTH_OK ANOMALIES CRITICALS WARNINGS <<< "$OVERALL"

# ------------------------------------------------------------------
# 狀態圖標
# ------------------------------------------------------------------
case "$STATUS" in
    HEALTHY)   ICON="✅" ;;
    WARNING)   ICON="⚠️"  ;;
    CRITICAL)  ICON="🚨" ;;
    *)         ICON="❓" ;;
esac

# ------------------------------------------------------------------
# 輸出單行簡潔報告（寫日誌 + 終端）
# ------------------------------------------------------------------
SUMMARY="[${TIMESTAMP}] ${ICON} ${STATUS} | 市場:${MARKET} | 進程:${PROC_ALIVE} | API:${HEALTH_OK} | 異常:${ANOMALIES} (嚴重:${CRITICALS} 警告:${WARNINGS})"
echo "$SUMMARY" | tee -a "$LOG_FILE"

# ------------------------------------------------------------------
# 如果有關鍵異常，輸出詳細信息
# ------------------------------------------------------------------
if [ "$CRITICALS" -gt 0 ] || [ "$WARNINGS" -gt 0 ]; then
    echo "" | tee -a "$LOG_FILE"
    echo "--- 異常明細 ---" | tee -a "$LOG_FILE"
    echo "$JSON_OUTPUT" | /usr/bin/python3 -c "
import sys, json
d = json.load(sys.stdin)
for a in d.get('anomalies', []):
    sev = a.get('severity', '?')
    msg = a.get('message', '')
    print(f'  [{sev}] {msg}')
" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"

    # 嚴重異常時發送桌面通知
    if [ "$CRITICALS" -gt 0 ]; then
        /usr/bin/python3 scripts/desktop_notify.py \
            "🚨 量化交易系統嚴重異常" \
            "${CRITICALS}項嚴重異常，請立即檢查！" 2>/dev/null || true
    fi
fi

echo "---" | tee -a "$LOG_FILE"

exit $EXIT_CODE
