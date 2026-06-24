#!/bin/bash
# Codex 監控 launchd 包裝腳本
# 在美股交易時段每2分鐘運行一次監控，異常時發送桌面通知

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR" || exit 1

exec /usr/bin/python3 scripts/codex_monitor.py --notify
