# AGENTS.md — Codex工作指引

## ⚠️ 开始任何工作前，必须先完整阅读 GUARDRAILS.md

本项目是一个**美股量化实盘交易系统**，代码经过多轮审计才达到稳定。

**强制规则**：
1. 每轮工作开始，第一件事是读取 `GUARDRAILS.md`
2. 修改代码后必须写工作日志到 `work_logs/`
3. 修改代码后必须全项目语法检查
4. 修改代码后必须重启服务并验证 `curl localhost:8080/api/status`
5. 不要触碰 `GUARDRAILS.md` 中列出的"不可改动区域"

## 快速启动

```bash
# 杀旧进程
lsof -ti tcp:8080 | xargs kill -9 2>/dev/null
screen -S trading -X quit 2>/dev/null

# 启动服务
screen -dmS trading python3 -u live_trading/web_server.py

# 验证
curl -s http://localhost:8080/api/status | python3 -c "import sys,json;d=json.load(sys.stdin);print(d.get('data_quality',{}).get('source'))"
```

## 文件结构

- `live_trading/` — 交易系统核心（web_server, predictor, portfolio, benchmark等）
- `ml_model/` — Transformer模型（StockTransformer是唯一使用的模型类）
- `config/` — 全局配置（ModelConfig改参数需注意checkpoint兼容性）
- `work_logs/` — **每次工作必须写日志到此目录**
- `GUARDRAILS.md` — **必读**：错误清单、禁区、工作规则
