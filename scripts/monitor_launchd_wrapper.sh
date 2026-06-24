#!/bin/bash
# launchd 包裝腳本 - 解決 macOS 安全權限問題
# launchd 直接調用 python3 可能因沙箱限制無法讀取腳本文件，
# 通過 shell 包裝可繼承終端的完整文件訪問權限。

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR" || exit 1

exec /usr/bin/python3 scripts/market_monitor.py --once
