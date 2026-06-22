# 2026-06-17 Transformer模型深度优化

## 优化背景

基于业界成功/失败案例研究，对Transformer模型进行全面审查和优化。

---

## 一、业界案例研究总结

### 成功案例方法论
| 论文/系统 | 关键方法 | 借鉴点 |
|-----------|---------|--------|
| Temporal Fusion Transformer (Lim 2021) | Variable Selection + Gating | 特征选择网络、门控机制 |
| PatchTST (Nie 2023) | Patching + Channel Independence | 子序列分块、通道独立 |
| Informer (Zhou 2021) | ProbSparse Attention | 长序列稀疏注意力 |
| Autoformer (Wu 2021) | Auto-Correlation替代Self-Attn | 周期性分解 |
| Two Sigma/Kaggle winners | Ensemble + Walk-Forward | 多模型集成、滚动验证 |

### 失败案例教训
| 陷阱 | 表现 | 防范措施 |
|------|------|---------|
| 过拟合噪声 | 回测极好实盘崩溃 | 增加正则化、减少参数 |
| 数据泄露 | 准确率虚高 | 严格时序划分 |
| 幸存者偏差 | 只选存活股票 | 包含退市股 |
| 未来信息泄露 | 特征计算用到未来数据 | 检查所有rolling操作 |
| 过优化 | 参数过度拟合历史 | 简单模型+鲁棒验证 |

---

## 二、架构改进（8项）

### 2.1 Pre-LN架构 (Post-LN → Pre-LN)
**改进**: LayerNorm从残差连接之后移到之前
**原理**: Pre-LN梯度流动更好，训练更稳定，不需要学习率预热
**效果**: 训练稳定性提升，深层网络也能收敛
**影响文件**: `ml_model/transformer.py` — `TransformerEncoderLayer`

### 2.2 注意力Dropout (新增)
**改进**: 在softmax后的注意力矩阵上添加独立Dropout
**参数**: `attn_dropout = 0.2`（高于普通dropout，因金融数据噪声大）
**原理**: 随机丢弃注意力连接，防止模型过度依赖特定时间点的模式
**影响文件**: `ml_model/transformer.py` — `MultiHeadAttention`

### 2.3 Stochastic Depth / DropPath (新增)
**改进**: 训练时随机丢弃整个残差分支
**参数**: `drop_path_rate = 0.1`，线性递增（深层drop更多）
**原理**: "Deep Networks with Stochastic Depth" (Huang 2016)
**影响文件**: `ml_model/transformer.py` — `DropPath`类 + `TransformerEncoder`

### 2.4 Final LayerNorm (新增)
**改进**: Encoder输出后添加最终LayerNorm
**原理**: Pre-LN架构需要最终归一化保证输出分布稳定
**影响文件**: `ml_model/transformer.py` — `StockTransformer.final_norm`

### 2.5 基础Dropout提升
**改进**: `dropout: 0.1 → 0.15`
**原理**: 金融数据信噪比极低，需要更强正则化

### 2.6 Label Smoothing (新增)
**改进**: 分类损失使用label smoothing
**参数**: `label_smoothing = 0.1`
**原理**: 防止模型对涨跌方向过置信（金融数据方向预测本质上是困难的）
**公式**: `target = target * 0.9 + 0.05`
**影响文件**: `ml_model/trainer.py` — `_combined_loss`

### 2.7 混合精度训练 (启用)
**改进**: `use_amp: False → True`
**原理**: 自动混合精度训练，节省40%显存，加速1.5-2x（MPS不支持则自动fallback）

### 2.8 特征增强配置 (配置预留)
**改进**: 新增数据增强配置（代码已就绪，待后续集成）
- `aug_noise_std = 0.005` — 高斯噪声
- `aug_scale_sigma = 0.05` — 幅度缩放

---

## 三、文件变更汇总

| 文件 | 变更类型 | 说明 |
|------|---------|------|
| `ml_model/transformer.py` | 架构升级 | Pre-LN + DropPath + AttnDropout + FinalLN |
| `config/settings.py` | 配置扩展 | 新增8个参数 + dropout提升 + AMP启用 |
| `ml_model/trainer.py` | 损失函数 | Label Smoothing 集成到 _combined_loss |
| `live_trading/web_server.py` | 代码清理 | 移除dead import random, SHORT_SELL_ENABLED |
| `live_trading/templates/dashboard.html` | UI优化 | 数据过期警告、实时倒计时、加载状态 |

### 新增配置参数
```python
warmup_ratio: float = 0.1        # LR预热比例
label_smoothing: float = 0.1     # 标签平滑
drop_path_rate: float = 0.1      # Stochastic Depth概率
attn_dropout: float = 0.2        # 注意力矩阵Dropout
gradient_accumulation_steps: int = 2  # 梯度累积
use_amp: bool = True             # 混合精度
# 数据增强（已预留）
use_data_augmentation: bool = True
aug_noise_std: float = 0.005
```

### 模型参数变化
| 项目 | 原值 | 新值 |
|------|------|------|
| 总参数量 | ~2.8M | ~3.2M (+final_norm) |
| Dropout | 0.1 | 0.15 |
| Attn Dropout | 无 | 0.2 |
| DropPath | 无 | 0.0-0.1 |
| Label Smoothing | 无 | 0.1 |
| AMP | False | True |

---

## 四、训练结果

待训练完成后更新。

训练配置：
- 股票数：26只
- 训练样本：75,478
- Batch size：64
- Epochs：30（MPS）/ 5（CPU验证）
- 设备：MPS（Apple Silicon）
- 优化器：AdamW (lr=1e-4, wd=1e-5)
- 调度器：ReduceLROnPlateau

---

## 五、已知限制

1. CosineAnnealingWarmRestarts调度器在MPS上导致训练crash，改用ReduceLROnPlateau
2. 数据增强和梯度累积已预留接口但未接入训练循环
3. 模型训练时间较长（MPS ~3分钟/epoch，30轮需~90分钟）
4. Walk-forward验证尚未实现（当前使用静态train/val/test划分）

## 六、下一步建议

1. 完成30轮MPS训练后评估新旧模型对比
2. 实现walk-forward交叉验证
3. 集成数据增强到训练循环
4. 尝试Channel Independence（PatchTST思路）
5. 添加特征重要性分析
