#!/bin/bash
# 市场监控守护进程管理（screen 方式）
# 与项目现有 watchdog/web_server 一致的使用模式

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SCREEN_NAME="trading_monitor"

cd "$PROJECT_DIR"

RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m'

start() {
    if screen -ls 2>/dev/null | grep -q "\.${SCREEN_NAME}"; then
        echo -e "${GREEN}✅ 监控已在运行 (screen: ${SCREEN_NAME})${NC}"
        status
        return 0
    fi

    echo -n "启动市场监控守护进程..."
    screen -dmS "$SCREEN_NAME" python3 -u scripts/market_monitor.py --daemon
    sleep 2

    if screen -ls 2>/dev/null | grep -q "\.${SCREEN_NAME}"; then
        echo -e "\n${GREEN}✅ 监控已启动 (screen: ${SCREEN_NAME})${NC}"
        echo -e "  日志: tail -f logs/market_monitor.log"
        echo -e "  异常: tail -f logs/market_anomalies.log"
    else
        echo -e "\n${RED}❌ 启动失败${NC}"
    fi
}

stop() {
    if screen -ls 2>/dev/null | grep -q "\.${SCREEN_NAME}"; then
        screen -S "$SCREEN_NAME" -X quit 2>/dev/null
        sleep 1
        echo -e "${GREEN}✅ 监控已停止${NC}"
    else
        echo "监控未在运行"
    fi
}

status() {
    echo -e "\n${CYAN}--- 市场监控 ---${NC}"
    if screen -ls 2>/dev/null | grep -q "\.${SCREEN_NAME}"; then
        echo -e "  ${GREEN}✅ 运行中${NC} (screen: ${SCREEN_NAME})"
    else
        echo -e "  ${RED}❌ 未运行${NC}"
    fi

    # 显示最近一次检查结果
    local last_check
    last_check=$(grep "检查完成" "$PROJECT_DIR/logs/market_monitor.log" 2>/dev/null | tail -1 || true)
    if [ -n "$last_check" ]; then
        echo "  最近: $last_check"
    fi

    echo -e "  ${CYAN}日志: tail -f logs/market_monitor.log${NC}"
    echo -e "  ${CYAN}异常: tail -f logs/market_anomalies.log${NC}"
    echo ""
}

restart() {
    stop
    sleep 2
    start
}

case "${1:-start}" in
    start)   start ;;
    stop)    stop ;;
    restart) restart ;;
    status)  status ;;
    *) echo "用法: $0 {start|stop|restart|status}" ;;
esac
