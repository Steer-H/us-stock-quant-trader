# 新模型部署到交易系统

**日期**: 2026-06-20
**任务**: 将情感特征模型部署到生产交易系统

## 修改文件

| 文件 | 操作 | 说明 |
|------|------|------|
| `config/settings.py` | ✏️ 更新 | d_model 384→192, heads 12→6, layers 6→3, d_ff 1536→768, lookback 90→60, epochs 100→30 |
| `live_trading/model_inference.py` | ✅ 无修改 | 自动适配28特征(已有兼容逻辑) |
| `data/models/transformer_stock_latest.pt` | 🔄 替换 | 部署情感模型(28特征/d192/5epoch) |

## 部署流程

1. **配置更新**: ModelConfig 默认值改为稳定参数(d192/bs32)
2. **模型部署**: `transformer_sentiment_e5.pt` → `transformer_stock_latest.pt`
3. **自动适配**: model_inference.py 自动检测28特征并加载
4. **服务重启**: 新模型生效，ML Ready: True

## 关键配置

```python
d_model = 192       # (原384)
n_heads = 6         # (原12)
n_layers = 3        # (原6)
d_ff = 768          # (原1536)
lookback = 60       # (原90)
batch_size = 32     # (已为32)
epochs = 30         # (原100)
features = 28       # 24 price + 4 sentiment
```

## 验证

```
✅ 模型加载: 28 features, MPS device
✅ 前向传播: reg(1,5), cls(1,5)
✅ 服务运行: Yahoo Finance (缓存 [v7])
✅ ML Ready: True
✅ 自动重训练将使用新配置
```

## 回滚方案

旧模型备份: `data/models/transformer_stock_latest_backup.pt`
如需回滚: `cp transformer_stock_latest_backup.pt transformer_stock_latest.pt`

## 影响

- 交易系统现使用28特征(含情感)模型
- 下次自动重训练(休市时)将使用优化配置d192/bs32
- 预期准确率提升: +0.8~2.9% vs 旧24特征模型
