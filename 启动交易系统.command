#!/bin/bash
cd "$(dirname "$0")"

echo "============================================"
echo "  美股量化交易系統 啟動"
echo "============================================"
echo ""

# 清理埠
lsof -ti tcp:8080 2>/dev/null | xargs kill -9 2>/dev/null
sleep 1

# 啟動伺服器（輸出到日誌文件）
python3 live_trading/web_server.py >> logs/server_output.log 2>&1 &
SERVER_PID=$!
echo "Web伺服器 PID: $SERVER_PID"

# 等待啟動
echo -n "啟動中"
for i in $(seq 1 30); do
    curl -s -o /dev/null "http://localhost:8080/api/health" 2>/dev/null && break
    echo -n "."
    sleep 1
done

echo ""
if curl -s "http://localhost:8080/api/health" > /dev/null 2>&1; then
    echo "✅ http://localhost:8080"
    open http://localhost:8080
else
    echo "❌ 啟動失敗，查看日誌:"
    echo "   tail -30 logs/server_output.log"
fi

echo ""
echo "按任意鍵關閉此窗口（伺服器繼續後臺運行）"
read -n 1
