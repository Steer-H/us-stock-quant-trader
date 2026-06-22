#!/bin/bash
# launchd 包装脚本 - 解决 macOS 安全权限问题
# launchd 直接调用 python3 可能因沙箱限制无法读取脚本文件，
# 通过 shell 包装可继承终端的完整文件访问权限。

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR" || exit 1

exec /usr/bin/python3 scripts/market_monitor.py --once
