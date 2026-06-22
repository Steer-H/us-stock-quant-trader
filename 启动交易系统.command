#!/bin/bash
cd "$(dirname "$0")"

echo "============================================"
echo "  美股量化交易系统 启动"
echo "============================================"
echo ""

# 清理端口
lsof -ti tcp:8080 2>/dev/null | xargs kill -9 2>/dev/null
sleep 1

# 启动服务器（输出到日志文件）
python3 live_trading/web_server.py >> logs/server_output.log 2>&1 &
SERVER_PID=$!
echo "Web服务器 PID: $SERVER_PID"

# 等待启动
echo -n "启动中"
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
    echo "❌ 启动失败，查看日志:"
    echo "   tail -30 logs/server_output.log"
fi

echo ""
echo "按任意键关闭此窗口（服务器继续后台运行）"
read -n 1
