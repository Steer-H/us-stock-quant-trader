#!/bin/bash
# Codex 监控 launchd 包装脚本
# 在美股交易时段每2分钟运行一次监控，异常时发送桌面通知

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR" || exit 1

exec /usr/bin/python3 scripts/codex_monitor.py --notify
