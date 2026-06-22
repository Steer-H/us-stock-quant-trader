#!/bin/bash
# ============================================================
#  Web 服务器自动重启脚本
#  - 监控 web_server.py，崩溃后自动重启
#  - 1小时内超过20次崩溃则停止
# ============================================================
cd "$(dirname "$0")"
COUNT=0
WINDOW_START=$(date +%s)

while true; do
    # 重置1小时窗口
    NOW=$(date +%s)
    if [ $((NOW - WINDOW_START)) -gt 3600 ]; then
        COUNT=0
        WINDOW_START=$NOW
    fi
    
    echo "[$(date +%H:%M:%S)] 启动 #$COUNT" >> logs/restart.log
    python3 live_trading/web_server.py 2>> logs/error.log
    COUNT=$((COUNT+1))
    echo "[$(date +%H:%M:%S)] 退出，5s后重启 (共$COUNT次)" >> logs/restart.log
    
    # 1小时内崩溃超过20次则停止
    if [ $COUNT -gt 20 ]; then
        echo "[$(date)] ❌ 1小时内崩溃 $COUNT 次，停止自动恢复！请手动检查。" >> logs/restart.log
        exit 1
    fi
    
    sleep 5
done
