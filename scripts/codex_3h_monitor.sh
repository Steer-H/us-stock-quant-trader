#!/bin/bash
# ============================================================================
# Codex 3小时粗略巡视脚本
# 每3小时运行一次，快速扫描系统关键指标，生成简洁状态报告
# ============================================================================

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="${PROJECT_DIR}/logs"
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
LOG_FILE="${LOG_DIR}/codex_3h_monitor.log"

cd "$PROJECT_DIR" || exit 1

mkdir -p "$LOG_DIR"

# ------------------------------------------------------------------
# 运行监控脚本，捕获 JSON 输出
# ------------------------------------------------------------------
JSON_OUTPUT=$(/usr/bin/python3 scripts/codex_monitor.py --json-only 2>/dev/null)
EXIT_CODE=$?

# ------------------------------------------------------------------
# 从 JSON 提取关键字段
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
# 状态图标
# ------------------------------------------------------------------
case "$STATUS" in
    HEALTHY)   ICON="✅" ;;
    WARNING)   ICON="⚠️"  ;;
    CRITICAL)  ICON="🚨" ;;
    *)         ICON="❓" ;;
esac

# ------------------------------------------------------------------
# 输出单行简洁报告（写日志 + 终端）
# ------------------------------------------------------------------
SUMMARY="[${TIMESTAMP}] ${ICON} ${STATUS} | 市场:${MARKET} | 进程:${PROC_ALIVE} | API:${HEALTH_OK} | 异常:${ANOMALIES} (严重:${CRITICALS} 警告:${WARNINGS})"
echo "$SUMMARY" | tee -a "$LOG_FILE"

# ------------------------------------------------------------------
# 如果有关键异常，输出详细信息
# ------------------------------------------------------------------
if [ "$CRITICALS" -gt 0 ] || [ "$WARNINGS" -gt 0 ]; then
    echo "" | tee -a "$LOG_FILE"
    echo "--- 异常明细 ---" | tee -a "$LOG_FILE"
    echo "$JSON_OUTPUT" | /usr/bin/python3 -c "
import sys, json
d = json.load(sys.stdin)
for a in d.get('anomalies', []):
    sev = a.get('severity', '?')
    msg = a.get('message', '')
    print(f'  [{sev}] {msg}')
" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"

    # 严重异常时发送桌面通知
    if [ "$CRITICALS" -gt 0 ]; then
        /usr/bin/python3 scripts/desktop_notify.py \
            "🚨 量化交易系统严重异常" \
            "${CRITICALS}项严重异常，请立即检查！" 2>/dev/null || true
    fi
fi

echo "---" | tee -a "$LOG_FILE"

exit $EXIT_CODE
