# AGENTS.md — Codex工作指引

## ⚠️ 開始任何工作前，必須先完整閱讀 GUARDRAILS.md

本項目是一個**美股量化實盤交易系統**，代碼經過多輪審計才達到穩定。

**強制規則**：
1. 每輪工作開始，第一件事是讀取 `GUARDRAILS.md`
2. 修改代碼後必須寫工作日誌到 `work_logs/`
3. 修改代碼後必須全項目語法檢查
4. 修改代碼後必須重啟服務並驗證 `curl localhost:8080/api/status`
5. 不要觸碰 `GUARDRAILS.md` 中列出的"不可改動區域"

## 快速啟動

```bash
# 殺舊進程
lsof -ti tcp:8080 | xargs kill -9 2>/dev/null
screen -S trading -X quit 2>/dev/null

# 啟動服務
screen -dmS trading python3 -u live_trading/web_server.py

# 驗證
curl -s http://localhost:8080/api/status | python3 -c "import sys,json;d=json.load(sys.stdin);print(d.get('data_quality',{}).get('source'))"
```

## 文件結構

- `live_trading/` — 交易系統核心（web_server, predictor, portfolio, benchmark等）
- `ml_model/` — Transformer模型（StockTransformer是唯一使用的模型類）
- `config/` — 全局配置（ModelConfig改參數需注意checkpoint兼容性）
- `work_logs/` — **每次工作必須寫日誌到此目錄**
- `GUARDRAILS.md` — **必讀**：錯誤清單、禁區、工作規則
