#!/bin/bash
# 監控CPU訓練進程，完成後提取評估結果並自動對比MPS
TRAIN_PID=20781
LOG="$(cd "$(dirname "$0")/.." && pwd)/logs/train_cpu_40stocks.log"
RESULT="$(cd "$(dirname "$0")/.." && pwd)/logs/train_cpu_40_result.txt"
COMPARE_SCRIPT="$(cd "$(dirname "$0")/.." && pwd)/scripts/compare_results.py"

echo "[$(date)] 監控啟動 PID=$$" > "$RESULT"
echo "  訓練PID: $TRAIN_PID" >> "$RESULT"

while true; do
    STATE=$(ps -p $TRAIN_PID -o state= 2>/dev/null)
    if [ -z "$STATE" ]; then
        echo "[$(date)] 訓練進程已結束" >> "$RESULT"
        break
    fi
    sleep 30
done

sleep 5

# 提取評估結果
echo "" >> "$RESULT"
echo "===== 評估結果 =====" >> "$RESULT"
grep -A 10 "評估結果" "$LOG" >> "$RESULT" 2>/dev/null

echo "" >> "$RESULT"
echo "===== 最後30行日誌 =====" >> "$RESULT"
tail -30 "$LOG" >> "$RESULT"

# 自動運行對比分析
echo "" >> "$RESULT"
echo "===== MPS vs CPU 對比 =====" >> "$RESULT"
python3 "$COMPARE_SCRIPT" >> "$RESULT" 2>&1

echo "" >> "$RESULT"
echo "[$(date)] 監控完成" >> "$RESULT"
