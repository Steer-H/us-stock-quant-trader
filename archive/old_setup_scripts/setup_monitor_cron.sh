#!/bin/bash
# 安装/更新 market_monitor 的 crontab 定时任务
# 每2分钟执行一次检查，脚本内部会判断市场是否活跃

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="$(which python3)"
MONITOR_SCRIPT="${PROJECT_DIR}/scripts/market_monitor.py"
CRON_MARKER="# 美股量化交易系统-市场监控"

echo "项目目录: ${PROJECT_DIR}"
echo "Python:    ${PYTHON}"
echo "监控脚本:  ${MONITOR_SCRIPT}"

# 确保脚本可执行
chmod +x "${MONITOR_SCRIPT}"

# 清理旧的监控任务
crontab -l 2>/dev/null | grep -v "${CRON_MARKER}" > /tmp/crontab_new 2>/dev/null || true

# 添加新的监控任务（每2分钟）
{
    echo ""
    echo "${CRON_MARKER}"
    echo "*/2 * * * * cd ${PROJECT_DIR} && ${PYTHON} ${MONITOR_SCRIPT} --once >> ${PROJECT_DIR}/logs/monitor_cron.log 2>&1"
} >> /tmp/crontab_new

# 安装
crontab /tmp/crontab_new
rm -f /tmp/crontab_new

echo ""
echo "✅ crontab 已更新"
echo ""
echo "当前 crontab:"
crontab -l | grep -v '^#'
echo ""
echo "查看监控日志: tail -f ${PROJECT_DIR}/logs/market_monitor.log"
echo "查看异常日志: tail -f ${PROJECT_DIR}/logs/market_anomalies.log"
echo "查看cron日志: tail -f ${PROJECT_DIR}/logs/monitor_cron.log"
echo "手动单次检查: python scripts/market_monitor.py --once"
echo "持续监控模式: python scripts/market_monitor.py --daemon"
