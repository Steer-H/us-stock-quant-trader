#!/bin/bash
cd "$(dirname "$0")"
SCREEN_NAME="trading_dashboard"
MONITOR_SCREEN="trading_monitor"
CODEX_MONITOR_SCREEN="codex_monitor"
LOG_DIR="logs"
PORT=8080
mkdir -p "$LOG_DIR"

RED='\033[0;31m'; GREEN='\033[0;32m'; CYAN='\033[0;36m'; NC='\033[0m'

start() {
    # 检查 screen session 是否已存在
    if screen -ls 2>/dev/null | grep -q "\.${SCREEN_NAME}"; then
        echo -e "${GREEN}✅ 已在运行 (screen: ${SCREEN_NAME})${NC}"
    else
        # 清理占用端口的僵尸进程
        lsof -ti "tcp:${PORT}" 2>/dev/null | xargs kill -9 2>/dev/null || true
        sleep 1
        
        # 使用 screen 启动(不会因 shell 退出而被杀)
        screen -dmS "$SCREEN_NAME" python3 -u live_trading/web_server.py
        echo -n "启动中"
        for i in $(seq 1 30); do
            curl -s -o /dev/null "http://localhost:${PORT}/api/health" 2>/dev/null && break
            echo -n "."; sleep 1
        done
        if curl -s "http://localhost:${PORT}/api/health" > /dev/null 2>&1; then
            echo -e "\n${GREEN}✅ http://localhost:${PORT}${NC}"
        else
            echo -e "\n${RED}❌ 启动失败,检查日志: tail -50 ${LOG_DIR}/server.log${NC}"
        fi
    fi

    # 启动市场监控守护进程
    if screen -ls 2>/dev/null | grep -q "\.${MONITOR_SCREEN}"; then
        echo -e "${GREEN}✅ 市场监控已在运行 (screen: ${MONITOR_SCREEN})${NC}"
    else
        screen -dmS "$MONITOR_SCREEN" python3 -u scripts/market_monitor.py --daemon
        sleep 1
        if screen -ls 2>/dev/null | grep -q "\.${MONITOR_SCREEN}"; then
            echo -e "${GREEN}✅ 市场监控已启动 (screen: ${MONITOR_SCREEN})${NC}"
        else
            echo -e "${RED}❌ 市场监控启动失败${NC}"
        fi
    fi

    # 启动 Codex 自动化监控（可选，带桌面通知）
    if screen -ls 2>/dev/null | grep -q "\.${CODEX_MONITOR_SCREEN}"; then
        echo -e "${GREEN}✅ Codex 监控已在运行 (screen: ${CODEX_MONITOR_SCREEN})${NC}"
    else
        echo -n "启动 Codex 自动化监控..."
        screen -dmS "$CODEX_MONITOR_SCREEN" bash -c '
            while true; do
                python3 -u scripts/codex_monitor.py --notify
                sleep 120
            done
        '
        sleep 2
        if screen -ls 2>/dev/null | grep -q "\.${CODEX_MONITOR_SCREEN}"; then
            echo -e "\n${GREEN}✅ Codex 监控已启动 (screen: ${CODEX_MONITOR_SCREEN})${NC}"
            echo -e "  ${CYAN}每2分钟自动检查，异常时桌面通知${NC}"
        else
            echo -e "\n${RED}❌ Codex 监控启动失败${NC}"
        fi
    fi

    status
}

stop() {
    screen -S "$SCREEN_NAME" -X quit 2>/dev/null
    screen -S "$MONITOR_SCREEN" -X quit 2>/dev/null
    screen -S "$CODEX_MONITOR_SCREEN" -X quit 2>/dev/null
    lsof -ti "tcp:${PORT}" 2>/dev/null | xargs kill -9 2>/dev/null || true
    sleep 1
    echo -e "${GREEN}已停止${NC}"
}

status() {
    echo -e "\n${CYAN}=== 交易系统 ===${NC}\n"
    H=$(curl -s --max-time 5 "http://localhost:${PORT}/api/health" 2>/dev/null)
    if [ -n "$H" ]; then
        echo "$H" | python3 -c "import sys,json;d=json.load(sys.stdin);print(f'  ✅ 运行中 | 刷新:#{d[\"uptime_iterations\"]} | {d[\"market_status\"]} | 建仓:{d[\"positions_initialized\"]}')" 2>/dev/null
    else
        echo "  ❌ 未运行"
    fi
    if screen -ls 2>/dev/null | grep -q "\.${SCREEN_NAME}"; then
        echo "  📺 screen: ${SCREEN_NAME} (Detached)"
    fi
    echo -e "  ${CYAN}http://localhost:${PORT}${NC}"

    echo -e "\n${CYAN}--- 市场监控 ---${NC}"
    if screen -ls 2>/dev/null | grep -q "\.${MONITOR_SCREEN}"; then
        last_check=$(grep "检查完成" "$LOG_DIR/market_monitor.log" 2>/dev/null | tail -1 || true)
        if [ -n "$last_check" ]; then
            echo -e "  ✅ 运行中 | $(echo "$last_check" | grep -o '系统正常\|异常:[0-9]*' | head -1)"
        else
            echo -e "  ✅ 运行中 (screen: ${MONITOR_SCREEN})"
        fi
    else
        echo -e "  ❌ 未运行"
    fi
    echo -e "  ${CYAN}日志: tail -f logs/market_monitor.log${NC}"
    echo -e "  ${CYAN}异常: tail -f logs/market_anomalies.log${NC}"

    echo -e "\n${CYAN}--- Codex 自动化监控 ---${NC}"
    if screen -ls 2>/dev/null | grep -q "\.${CODEX_MONITOR_SCREEN}"; then
        echo -e "  ✅ 运行中 (screen: ${CODEX_MONITOR_SCREEN})"
        codex_last=$(grep "检查完成" "$LOG_DIR/codex_monitor.log" 2>/dev/null | tail -1 || true)
        if [ -n "$codex_last" ]; then
            echo "  最近: $codex_last"
        fi
    else
        echo -e "  ❌ 未运行"
    fi
    echo -e "  ${CYAN}手动检查: bash start.sh codex-check${NC}\n"
}

case "${1:-start}" in
    start) start ;;
    stop) stop ;;
    restart) stop; sleep 2; start ;;
    status) status ;;
    codex-check)
        python3 scripts/codex_monitor.py --notify
        ;;
    logs) tail -f "$LOG_DIR/server.log" ;;
    monitor-logs) tail -f "$LOG_DIR/market_monitor.log" ;;
    anomaly-logs) tail -f "$LOG_DIR/market_anomalies.log" ;;
    *) echo "用法: $0 {start|stop|restart|status|codex-check|logs|monitor-logs|anomaly-logs}" ;;
esac
