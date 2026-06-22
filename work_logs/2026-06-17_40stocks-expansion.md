# 2026-06-17 交易系统扩展至40只股票

## 变更概述

将交易系统从20只扩展到40只，覆盖8大板块，包含高波动芯片/光模块/存储股。

---

## 一、股票扩展明细

### 已有20只（保留）
科技七巨头+软件+金融+消费+芯片

### 补全6只（已有数据，未加入交易系统）
| 代码 | 名称 | 板块 |
|------|------|------|
| BAC | Bank of America | 金融 |
| INTC | Intel | 芯片 |
| ORCL | Oracle | 软件 |
| QCOM | Qualcomm | 芯片 |
| TXN | Texas Instruments | 芯片 |
| UBER | Uber | 消费 |

### 新增14只（爬取+训练）
| 代码 | 名称 | 板块 | 波动特征 |
|------|------|------|---------|
| AAOI | Applied Optoelectronics | 光模块 | 极高波动 |
| COHR | Coherent | 光通信/激光 | 高波动 |
| LITE | Lumentum | 光器件 | 高波动 |
| FN | Fabrinet | 光模块代工 | 中高波动 |
| WDC | Western Digital | 存储 | 高波动 |
| STX | Seagate | 存储 | 中高波动 |
| NTAP | NetApp | 存储 | 中等波动 |
| MRVL | Marvell | 数据中心芯片 | 高波动 |
| MU | Micron | 存储芯片 | 高波动 |
| LRCX | Lam Research | 半导体设备 | 高波动 |
| AMAT | Applied Materials | 半导体设备 | 中高波动 |
| KLAC | KLA Corporation | 检测设备 | 中等波动 |
| SNPS | Synopsys | EDA软件 | 中等波动 |
| CDNS | Cadence | EDA软件 | 中等波动 |

---

## 二、板块分布（40只）

| 板块 | 数量 | 代表股票 |
|------|------|---------|
| 科技七巨头 | 7 | AAPL,MSFT,NVDA,GOOGL,AMZN,META,TSLA |
| 软件/SaaS | 5 | NFLX,ADBE,CRM,NOW,ORCL |
| 金融 | 4 | JPM,V,MA,BAC |
| 消费 | 5 | WMT,HD,NKE,SBUX,UBER |
| 芯片/半导体 | 5 | AVGO,AMD,INTC,QCOM,TXN |
| 光模块/光通信 | 4 | AAOI,COHR,LITE,FN |
| 存储 | 3 | WDC,STX,NTAP |
| 数据中心芯片 | 2 | MRVL,MU |
| 半导体设备 | 3 | LRCX,AMAT,KLAC |
| EDA软件 | 2 | SNPS,CDNS |

---

## 三、数据统计

| 指标 | 26只 | 40只 | 增幅 |
|------|------|------|------|
| 训练样本 | 75,478 | 113,182 | +50% |
| 验证样本 | 16,172 | 24,247 | +50% |
| 测试样本 | 16,198 | 24,285 | +50% |
| 训练批次 | 1,178 | 1,763 | +50% |
| 估计训练时间 | ~95min | ~143min | +50% |

---

## 四、技术修复

### 爬虫时区bug
- **问题**: Yahoo Finance返回tz-aware时间戳，pandas操作时与tz-naive比较报错
- **修复**: 数据获取后立即 `df.index.tz_localize(None)`
- **文件**: `crawler/stock_crawler.py`

### 14只新股数据
- 数据范围: 2010-01-01 ~ 2025-12-31
- 平均每只: ~3,900条日线记录
- 总数据行: ~55,000行

---

## 五、训练配置

```python
股票数: 40只
Epochs: 30
Batch size: 64
学习率: 1e-4
优化器: AdamW
设备: MPS (Apple Silicon)
架构: Pre-LN Transformer (d_model=256, 4层, 8头)
正则化: dropout=0.15, attn_dropout=0.2, drop_path=0.1
损失: MSE + BCEWithLogits(LabelSmoothing=0.1)
```

## 六、文件变更

| 文件 | 变更 |
|------|------|
| `live_trading/web_server.py` | TRACKED_TICKERS 20→40, INITIAL_POSITIONS 20→40 |
| `crawler/stock_crawler.py` | 修复Yahoo Finance时区bug |
| `data/processed/` | 新增14个parquet文件 |
| `data/raw/` | 新增14个CSV文件 |
