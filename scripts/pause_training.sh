#!/bin/bash
# 在开盘前10分钟暂停CPU训练，收盘后恢复
# 美股交易时段: 21:30-04:00 CST

TRAIN_PID=$(pgrep -f "train_cpu_40.py" | head -1)
LOG="$(cd "$(dirname "$0")/.." && pwd)/logs/train_cpu_40stocks.log"
PAUSE_LOG="$(cd "$(dirname "$0")/.." && pwd)/logs/train_pause.log"

echo "[$(date)] 训练暂停/恢复调度启动 PID=$$" >> "$PAUSE_LOG"
echo "  训练进程: $TRAIN_PID" >> "$PAUSE_LOG"

if [ -z "$TRAIN_PID" ]; then
    echo "[$(date)] 错误: 未找到训练进程" >> "$PAUSE_LOG"
    exit 1
fi

# 计算等待秒数到 21:20 CST
H=$(date +%H)
M=$(date +%M)
TARGET_MIN=$((21*60+20))
CURRENT_MIN=$((10#$H*60+10#$M))
WAIT_SEC=$(( (TARGET_MIN - CURRENT_MIN) * 60 ))
if [ $WAIT_SEC -lt 0 ]; then
    WAIT_SEC=0
fi

echo "  将在 $WAIT_SEC 秒后 (21:20 CST) 暂停训练" >> "$PAUSE_LOG"
sleep $WAIT_SEC

# 暂停训练
kill -STOP $TRAIN_PID 2>/dev/null
if [ $? -eq 0 ]; then
    echo "[$(date)] ✅ 训练已暂停 (SIGSTOP → PID $TRAIN_PID)" >> "$PAUSE_LOG"
else
    echo "[$(date)] ⚠️ 暂停失败，进程可能已结束" >> "$PAUSE_LOG"
fi

# 计算等待到 04:00 CST 的秒数（次日）
H=$(date +%H)
M=$(date +%M)
if [ $H -lt 4 ]; then
    WAIT_SEC=$(( (4*60 - 10#$H*60 - 10#$M) * 60 ))
else
    WAIT_SEC=$(( (28*60 - 10#$H*60 - 10#$M) * 60 ))  # next day 04:00
fi

echo "  将在 $WAIT_SEC 秒后 (04:00 CST) 恢复训练" >> "$PAUSE_LOG"
sleep $WAIT_SEC

# 恢复训练
kill -CONT $TRAIN_PID 2>/dev/null
if [ $? -eq 0 ]; then
    echo "[$(date)] ✅ 训练已恢复 (SIGCONT → PID $TRAIN_PID)" >> "$PAUSE_LOG"
else
    echo "[$(date)] ⚠️ 恢复失败，进程可能已结束" >> "$PAUSE_LOG"
fi
