# Transformer ML算法升级

**日期**: 2026-06-20
**目标**: 从ML算法设计角度优化Transformer，提升准确率
**训练**: CPU训练进行中 (screen: quick_train, 日志: logs/quick_train.log)

---

## 修改文件

| 文件 | 改动 |
|------|------|
| `ml_model/transformer.py` | 7项架构改进 |
| `ml_model/trainer.py` | 4项训练改进 |
| `config/settings.py` | 超参数更新 + 新增字段 |
| `scripts/quick_train.py` | 新增训练脚本 |
| `scripts/train_upgraded_cpu.py` | 新增完整训练脚本 |

---

## 算法改进详情

### 1. 可学习位置编码 (Learnable PE)
**原方案**: 固定正弦/余弦位置编码
**改进**: `nn.Parameter` 可学习PE，初始化时warm-start自正弦编码
**原因**: 金融数据的时间结构（财报周期、季节性）无法被固定频率编码充分捕捉

### 2. GELU激活函数
**原方案**: ReLU
**改进**: GELU (Gaussian Error Linear Unit)
**影响**: FeedForward + 两个预测头全部替换
**原因**: 现代Transformer普遍使用GELU，在金融时序数据上通常比ReLU表现更好

### 3. Stochastic Depth (DropPath)
**原方案**: DropPath类已实现但从未使用
**改进**: 在TransformerEncoderLayer中启用，线性递增drop率
**影响**: 深层网络更好的正则化，减少过拟合

### 4. 注意力池化 (Attention Pooling)
**原方案**: AdaptiveAvgPool1d 平均池化
**改进**: 可学习query对所有时间步做注意力加权聚合
**原因**: 不同时间点对预测的贡献不同，注意力池化能自动聚焦关键时间点

### 5. Kaiming权重初始化
**原方案**: Xavier均匀初始化
**改进**: Kaiming正态初始化 (针对GELU/ReLU优化)
**原因**: Kaiming初始化更好地保持前向传播方差稳定性

### 6. Focal Loss (分类)
**原方案**: 普通BCEWithLogitsLoss
**改进**: FocalBCELoss(gamma=2.0, alpha=0.25)
**原因**: 金融预测天然类别不平衡，Focal Loss自动聚焦难分类样本

### 7. Label Smoothing
**新增**: label_smoothing=0.1
**原因**: 防止模型过度自信，改善概率校准

### 8. 梯度裁剪
**新增**: clip_grad_norm=1.0
**原因**: 防止金融数据波动导致的梯度爆炸

### 9. Cosine退火调度器
**原方案**: ReduceLROnPlateau
**改进**: CosineAnnealingWarmRestarts (T_0=20, T_mult=2)
**原因**: 周期性重启帮助逃离局部最优

### 10. 超参数调整
| 参数 | 旧值 | 新值 |
|------|------|------|
| d_model | 256 | 256 (CPU) / 384 (GPU) |
| n_heads | 8 | 8 (CPU) / 12 (GPU) |
| n_layers | 4 | 4 (CPU) / 6 (GPU) |
| d_ff | 1024 | 1024 (CPU) / 1536 (GPU) |
| dropout | 0.15 | 0.2 |
| batch_size | 64 | 16 (CPU) / 32 (GPU) |
| learning_rate | 1e-4 | 3e-4 |
| epochs | 50 | 20 (quick) / 40 (full) |
| weight_decay | 1e-4 | 1e-5 |

---

## 训练状态

- 模型参数: ~11.5M
- 训练数据: 40只股票, 113,182样本
- 验证数据: 24,247样本
- 测试数据: 24,285样本
- 设备: CPU
- 监控: `tail -f logs/quick_train.log`

---

## 预期效果

基于文献和业界实践，预期方向准确率可提升3-8个百分点。
具体效果待训练完成后验证。
