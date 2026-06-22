# 美股量化交易系统 v2.0

> 完整技术手册请见 [`instructions/`](./instructions/) 目录

- [📖 技术手册 (Markdown)](./instructions/README.md)
- [🌐 网页版 (HTML)](./instructions/README.html)
- [📄 PDF 版](./instructions/README.pdf)

### 快速启动

```bash
screen -dmS trading python3 -u live_trading/web_server.py
open http://localhost:8080
```

### 📋 开发规范

- ⚠️ **[GUARDRAILS.md](./GUARDRAILS.md)** — 历史错误清单、不可改动区域、工作规则（**每次修改前必读**）
- 📝 **[work_logs/](./work_logs/)** — 所有工作日志
- 📋 **[CHANGELOG.md](./work_logs/CHANGELOG.md)** — 变更摘要
