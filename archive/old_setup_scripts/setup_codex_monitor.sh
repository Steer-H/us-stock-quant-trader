#!/bin/bash
# 安装 Codex 自动化监控的 crontab 定时任务
# 仅在美股交易时段（北京时间 16:00-次日10:00 周一至周五）运行

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="$(which python3)"
MONITOR_SCRIPT="${PROJECT_DIR}/scripts/codex_monitor.py"
CRON_MARKER="# 美股量化交易系统-Codex自动化监控"

echo "================================================"
echo "  Codex 自动化监控 - crontab 安装"
echo "================================================"
echo "项目目录: ${PROJECT_DIR}"
echo "Python:    ${PYTHON}"
echo "监控脚本:  ${MONITOR_SCRIPT}"

# 确保脚本可执行
chmod +x "${MONITOR_SCRIPT}"

# 清理旧的 Codex 监控任务
crontab -l 2>/dev/null | grep -v "${CRON_MARKER}" > /tmp/crontab_new 2>/dev/null || true

# 添加新的监控任务
# 美股交易时段（北京时间）：
#   夏令时 EDT(3-11月): 16:00-次日08:00
#   冬令时 EST(11-3月): 17:00-次日09:00
# 取并集覆盖两季: 16:00-次日10:00 周一至周五
# crontab 分 时 日 月 周: 每2分钟
{
    echo ""
    echo "${CRON_MARKER}"
    echo "# 美股交易时段覆盖（北京时间 16:00-次日10:00，覆盖夏令时+冬令时）"
    echo "# 周一至周五 16:00-23:59 每2分钟"
    echo "*/2 16-23 * * 1-5 cd ${PROJECT_DIR} && ${PYTHON} ${MONITOR_SCRIPT} --notify >> ${PROJECT_DIR}/logs/codex_monitor_cron.log 2>&1"
    echo "# 周一至周五 00:00-09:59 每2分钟（跨日覆盖到盘后结束）"
    echo "*/2 0-9 * * 1-5 cd ${PROJECT_DIR} && ${PYTHON} ${MONITOR_SCRIPT} --notify >> ${PROJECT_DIR}/logs/codex_monitor_cron.log 2>&1"
    echo "# 周五晚间延续到周六凌晨（美股周五盘后可能到周六北京时间10点前）"
    echo "*/2 0-9 * * 6 cd ${PROJECT_DIR} && ${PYTHON} ${MONITOR_SCRIPT} --notify >> ${PROJECT_DIR}/logs/codex_monitor_cron.log 2>&1"
} >> /tmp/crontab_new

# 安装
crontab /tmp/crontab_new
rm -f /tmp/crontab_new

echo ""
echo "✅ crontab 已安装"
echo ""
echo "⏰ 监控时段（北京时间）："
echo "   周一 16:00 - 周六 10:00"
echo "   每 2 分钟检查一次"
echo "   脚本内部会再次判断市场是否活跃"
echo ""
echo "📊 查看日志："
echo "   tail -f ${PROJECT_DIR}/logs/codex_monitor_cron.log"
echo "   tail -f ${PROJECT_DIR}/logs/codex_monitor.log"
echo ""
echo "🔔 异常时会收到 macOS 桌面通知"
echo ""
echo "🧪 手动测试："
echo "   cd ${PROJECT_DIR} && python scripts/codex_monitor.py --notify"
echo ""
echo "📋 当前 crontab："
crontab -l | grep -v '^#' | grep -v '^$'
