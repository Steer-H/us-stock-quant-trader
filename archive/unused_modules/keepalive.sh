#!/bin/bash
# ============================================================
#  Web 服务器自动重启守护脚本
#  - 监控 web_server.py，崩溃后自动重启
#  - 1小时内超过20次崩溃则停止
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
RESTART_LOG="$LOG_DIR/server_restart.log"
RESTART_COUNT=0
RESTART_WINDOW_START=$(date +%s)

mkdir -p "$LOG_DIR"

cleanup() {
    # 确保子进程被清理
    kill -TERM "$CHILD_PID" 2>/dev/null || true
    wait "$CHILD_PID" 2>/dev/null || true
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 守护进程退出" >> "$RESTART_LOG"
    exit 0
}

trap cleanup SIGTERM SIGINT

echo "[$(date '+%Y-%m-%d %H:%M:%S')] ===== 守护进程启动 =====" >> "$RESTART_LOG"

while true; do
    # 重置1小时窗口
    NOW=$(date +%s)
    if [ $((NOW - RESTART_WINDOW_START)) -gt 3600 ]; then
        RESTART_COUNT=0
        RESTART_WINDOW_START=$NOW
    fi
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 启动 Web 服务器 (重启 #$RESTART_COUNT)" >> "$RESTART_LOG"
    
    # 启动 web_server 并等待
    python3 "$SCRIPT_DIR/live_trading/web_server.py" \
        >> "$LOG_DIR/server_stdout.log" 2>> "$LOG_DIR/server_stderr.log" &
    CHILD_PID=$!
    
    wait "$CHILD_PID" 2>/dev/null
    EXIT_CODE=$?
    
    RESTART_COUNT=$((RESTART_COUNT + 1))
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 服务器退出 (code: $EXIT_CODE), 重启 #$RESTART_COUNT, 5s后..." >> "$RESTART_LOG"
    
    # 检查是否崩溃太频繁
    if [ $RESTART_COUNT -gt 20 ]; then
        echo "[$(date)] ❌ 1小时内崩溃 $RESTART_COUNT 次，停止自动恢复！请手动检查。" >> "$RESTART_LOG"
        exit 1
    fi
    
    sleep 5
done
