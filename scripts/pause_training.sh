#!/bin/bash
# 在開盤前10分鐘暫停CPU訓練，收盤後恢復
# 美股交易時段: 21:30-04:00 CST

TRAIN_PID=$(pgrep -f "train_cpu_40.py" | head -1)
LOG="$(cd "$(dirname "$0")/.." && pwd)/logs/train_cpu_40stocks.log"
PAUSE_LOG="$(cd "$(dirname "$0")/.." && pwd)/logs/train_pause.log"

echo "[$(date)] 訓練暫停/恢復調度啟動 PID=$$" >> "$PAUSE_LOG"
echo "  訓練進程: $TRAIN_PID" >> "$PAUSE_LOG"

if [ -z "$TRAIN_PID" ]; then
    echo "[$(date)] 錯誤: 未找到訓練進程" >> "$PAUSE_LOG"
    exit 1
fi

# 計算等待秒數到 21:20 CST
H=$(date +%H)
M=$(date +%M)
TARGET_MIN=$((21*60+20))
CURRENT_MIN=$((10#$H*60+10#$M))
WAIT_SEC=$(( (TARGET_MIN - CURRENT_MIN) * 60 ))
if [ $WAIT_SEC -lt 0 ]; then
    WAIT_SEC=0
fi

echo "  將在 $WAIT_SEC 秒後 (21:20 CST) 暫停訓練" >> "$PAUSE_LOG"
sleep $WAIT_SEC

# 暫停訓練
kill -STOP $TRAIN_PID 2>/dev/null
if [ $? -eq 0 ]; then
    echo "[$(date)] ✅ 訓練已暫停 (SIGSTOP → PID $TRAIN_PID)" >> "$PAUSE_LOG"
else
    echo "[$(date)] ⚠️ 暫停失敗，進程可能已結束" >> "$PAUSE_LOG"
fi

# 計算等待到 04:00 CST 的秒數（次日）
H=$(date +%H)
M=$(date +%M)
if [ $H -lt 4 ]; then
    WAIT_SEC=$(( (4*60 - 10#$H*60 - 10#$M) * 60 ))
else
    WAIT_SEC=$(( (28*60 - 10#$H*60 - 10#$M) * 60 ))  # next day 04:00
fi

echo "  將在 $WAIT_SEC 秒後 (04:00 CST) 恢復訓練" >> "$PAUSE_LOG"
sleep $WAIT_SEC

# 恢復訓練
kill -CONT $TRAIN_PID 2>/dev/null
if [ $? -eq 0 ]; then
    echo "[$(date)] ✅ 訓練已恢復 (SIGCONT → PID $TRAIN_PID)" >> "$PAUSE_LOG"
else
    echo "[$(date)] ⚠️ 恢復失敗，進程可能已結束" >> "$PAUSE_LOG"
fi
