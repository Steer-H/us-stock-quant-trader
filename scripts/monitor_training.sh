#!/bin/bash
# 监控CPU训练进程，完成后提取评估结果并自动对比MPS
TRAIN_PID=20781
LOG="$(cd "$(dirname "$0")/.." && pwd)/logs/train_cpu_40stocks.log"
RESULT="$(cd "$(dirname "$0")/.." && pwd)/logs/train_cpu_40_result.txt"
COMPARE_SCRIPT="$(cd "$(dirname "$0")/.." && pwd)/scripts/compare_results.py"

echo "[$(date)] 监控启动 PID=$$" > "$RESULT"
echo "  训练PID: $TRAIN_PID" >> "$RESULT"

while true; do
    STATE=$(ps -p $TRAIN_PID -o state= 2>/dev/null)
    if [ -z "$STATE" ]; then
        echo "[$(date)] 训练进程已结束" >> "$RESULT"
        break
    fi
    sleep 30
done

sleep 5

# 提取评估结果
echo "" >> "$RESULT"
echo "===== 评估结果 =====" >> "$RESULT"
grep -A 10 "评估结果" "$LOG" >> "$RESULT" 2>/dev/null

echo "" >> "$RESULT"
echo "===== 最后30行日志 =====" >> "$RESULT"
tail -30 "$LOG" >> "$RESULT"

# 自动运行对比分析
echo "" >> "$RESULT"
echo "===== MPS vs CPU 对比 =====" >> "$RESULT"
python3 "$COMPARE_SCRIPT" >> "$RESULT" 2>&1

echo "" >> "$RESULT"
echo "[$(date)] 监控完成" >> "$RESULT"
