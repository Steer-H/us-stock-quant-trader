#!/bin/bash
# 市场监控守护进程管理
# 使用 screen 方式运行，与项目现有的 web_server/watchdog 一致

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# 委托给 monitor_daemon.sh
exec bash "${PROJECT_DIR}/scripts/monitor_daemon.sh" "$@"
