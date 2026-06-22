"""
美股量化交易系统 - Transformer时序预测模型

基于 "Attention Is All You Need" (Vaswani et al., 2017) 的架构，
针对金融时间序列预测进行适配优化。

核心架构：
┌─────────────────────────────────────────┐
│  输入: (batch, seq_len, n_features)      │
│    ↓                                     │
│  输入投影层 (Linear → d_model)           │
│    ↓                                     │
│  位置编码 (Positional Encoding)           │
│    ↓                                     │
│  Transformer Encoder × N layers          │
│    ├── Multi-Head Self-Attention         │
│    ├── Feed-Forward Network              │
│    ├── Layer Normalization               │
│    └── Residual Connections              │
│    ↓                                     │
│  Transformer Decoder × N layers          │
│    ├── Masked Multi-Head Self-Attention  │
│    ├── Cross-Attention (with Encoder)    │
│    └── Feed-Forward Network              │
│    ↓                                     │
│  输出投影层 (d_model → horizon)           │
│    ↓                                     │
│  输出: (batch, horizon) 预测序列          │
└─────────────────────────────────────────┘

针对金融数据的特殊设计：
1. 时间感知位置编码：使用正弦/余弦位置编码，捕捉周期模式
2. 多尺度特征融合：不同周期的指标通过不同注意力头处理
3. 正则化增强：Dropout + LayerNorm + 权重衰减 ，防止过拟合
4. 因果掩码：Decoder使用因果自注意力，确保预测只依赖历史信息

参考文献:
- Vaswani et al., "Attention Is All You Need" (2017)
- Wu et al., "Autoformer: Decomposition Transformers with Auto-Correlation" (2021)
- Lim et al., "Temporal Fusion Transformers" (2021)
"""

import math
import logging
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config.settings import ModelConfig

logger = logging.getLogger(__name__)


# ============================================================================
# 位置编码 (Positional Encoding)
# ============================================================================
class PositionalEncoding(nn.Module):
    """
    可学习位置编码 (Learnable Positional Encoding)
    
    与固定正弦编码不同，可学习位置编码允许模型根据金融数据的
    特性自适应调整位置表示。这对捕捉财报周期、季节性模式等
    非均匀时间结构特别重要。
    
    优势:
    - 自适应: 根据数据分布学习最优位置表示
    - 灵活: 不受固定频率假设限制
    - 兼容: 初始化时可从正弦编码warm-start
    
    时间复杂度: O(1) per forward
    空间复杂度: O(max_len * d_model)
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Xavier-style init for learnable PE with sinusoidal warm-start"""
        with torch.no_grad():
            position = torch.arange(self.pe.size(1)).unsqueeze(1).float()
            div_term = torch.exp(
                torch.arange(0, self.pe.size(2), 2).float() *
                (-math.log(10000.0) / self.pe.size(2))
            )
            init = torch.zeros_like(self.pe[0])
            init[:, 0::2] = torch.sin(position * div_term)
            init[:, 1::2] = torch.cos(position * div_term)
            self.pe[0].copy_(init)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, :x.size(1), :])


# ============================================================================
# 多头自注意力 (Multi-Head Self-Attention)
# ============================================================================

# ============================================================================
# Stochastic Depth (DropPath)
# 参考: "Deep Networks with Stochastic Depth" (Huang et al., 2016)
# 随机丢弃整个残差分支，比普通Dropout更强的正则化
# ============================================================================
class DropPath(nn.Module):
    """Stochastic Depth per sample"""
    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        if self.scale_by_keep:
            x = x.div(keep_prob) * random_tensor
        else:
            x = x * random_tensor
        return x

class MultiHeadAttention(nn.Module):
    """
    多头自注意力机制
    
    将输入投影到多个注意力头，每个头独立计算缩放点积注意力，
    最后拼接并投影回原始维度。
    
    核心公式：
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
    
    多头设计允许模型同时关注不同子空间的信息：
    - 短周期头：关注近期价格变化
    - 长周期头：关注长期趋势
    - 交叉头：关注特征之间的交互
    
    时间复杂度: O(batch * seq_len^2 * d_model)
    空间复杂度: O(batch * n_heads * seq_len^2)  # 注意力矩阵
    
    参数:
        d_model: 模型维度
        n_heads: 注意力头数
        dropout: Attention dropout率
    """
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        
        assert d_model % n_heads == 0, \
            f"d_model({d_model})必须能被n_heads({n_heads})整除"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # 每个头的维度
        
        # Q, K, V的线性投影矩阵
        # 三个合在一起以减少计算开销
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        
        # 输出投影矩阵
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        # 缩放因子: 1/sqrt(d_k)，避免点积过大导致softmax梯度消失
        self.scale = math.sqrt(self.d_k)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播
        
        参数:
            query:  (batch, seq_len_q, d_model)
            key:    (batch, seq_len_k, d_model)
            value:  (batch, seq_len_v, d_model)
            mask:   (batch, seq_len_q, seq_len_k) 或 (seq_len_q, seq_len_k)
                    可选掩码，用于因果掩码或填充掩码
        
        返回:
            注意力输出，形状 (batch, seq_len_q, d_model)
        """
        batch_size = query.size(0)
        seq_len_q = query.size(1)
        seq_len_k = key.size(1)
        
        # 1. 线性投影并分割成多头
        # self-attention: query==key==value, 一次投影即得QKV
        # cross-attention: key/value来自不同输入，需分别投影
        qkv = self.qkv_proj(query).view(
            batch_size, seq_len_q, 3, self.n_heads, self.d_k
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, n_heads, seq, d_k)
        
        if query is key and key is value:
            # self-attention: 所有QKV来自同一次投影
            q, k, v = qkv[0], qkv[1], qkv[2]
        else:
            # cross-attention: K和V来自不同输入
            q = qkv[0]
            k = self.qkv_proj(key).view(
                batch_size, seq_len_k, 3, self.n_heads, self.d_k
            ).permute(2, 0, 3, 1, 4)[1]
            v = self.qkv_proj(value).view(
                batch_size, seq_len_k, 3, self.n_heads, self.d_k
            ).permute(2, 0, 3, 1, 4)[2]
        
        # 2. 计算缩放点积注意力
        # attn_scores: (batch, n_heads, seq_len_q, seq_len_k)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        # 3. 应用掩码（如果有）
        if mask is not None:
            # mask中值为True的位置将被设为 -inf，softmax后为0
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        # 4. Softmax得到注意力权重
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 5. 加权聚合Value
        # (batch, n_heads, seq_len_q, d_k)
        attn_output = torch.matmul(attn_weights, v)
        
        # 6. 合并多头并投影
        # (batch, n_heads, seq_len_q, d_k) → (batch, seq_len_q, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len_q, self.d_model
        )
        
        return self.out_proj(attn_output)


# ============================================================================
# 前馈网络 (Feed-Forward Network)
# ============================================================================
class FeedForward(nn.Module):
    """
    位置独立的前馈网络
    
    结构: Linear → ReLU → Dropout → Linear → Dropout
    
    两个线性层之间使用ReLU激活函数，中间维度通常为d_model的4倍。
    这个膨胀-收缩结构提供了非线性变换能力。
    
    时间复杂度: O(batch * seq_len * d_model * d_ff)
    空间复杂度: O(batch * seq_len * d_ff)
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        """
        参数:
            d_model: 输入/输出维度
            d_ff: 隐藏层维度（通常是d_model的4倍）
            dropout: Dropout率
        """
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数:
            x: (batch, seq_len, d_model)
        
        返回:
            (batch, seq_len, d_model)
        """
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x


# ============================================================================
# Transformer Encoder 层
# ============================================================================
class TransformerEncoderLayer(nn.Module):
    """
    单层Transformer Encoder
    
    结构（Pre-LN设计，更稳定的训练）：
    ┌──────────────────────┐
    │ 输入 x                │
    │   ↓                   │
    │  LayerNorm(x)         │
    │   ↓                   │
    │  MultiHeadAttn(x,x,x) │
    │   ↓                   │
    │  Dropout + 残差连接   │
    │   ↓                   │
    │  LayerNorm(x')        │
    │   ↓                   │
    │  FeedForward(x')      │
    │   ↓                   │
    │  Dropout + 残差连接   │
    │   ↓                   │
    │  输出                  │
    └──────────────────────┘
    
    使用Pre-LN (LayerNorm before sublayer) 而非Post-LN，
    在深层网络中更稳定，不易出现训练不收敛的问题。
    """
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, 
                 dropout: float = 0.1, drop_path: float = 0.0):
        super().__init__()
        
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        # LayerNorm 对各特征维度独立归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        参数:
            x: (batch, seq_len, d_model)
            mask: 自注意力掩码
        
        返回:
            (batch, seq_len, d_model)
        """
        # 自注意力 + 残差连接 + DropPath
        attn_output = self.self_attn(
            self.norm1(x), self.norm1(x), self.norm1(x), mask
        )
        x = x + self.drop_path(self.dropout1(attn_output))
        
        # 前馈网络 + 残差连接 + DropPath
        ff_output = self.feed_forward(self.norm2(x))
        x = x + self.drop_path(self.dropout2(ff_output))
        
        return x


# ============================================================================
# Transformer Encoder（多层堆叠）
# ============================================================================
class TransformerEncoder(nn.Module):
    """
    多层Transformer Encoder堆叠
    
    每层的输出作为下一层的输入，逐层提取更高级的特征。
    浅层捕捉局部模式（如短期趋势），深层捕捉全局模式（如长期周期）。
    """
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, 
                 n_layers: int, dropout: float = 0.1, drop_path_rate: float = 0.0):
        super().__init__()
        # 线性增加的drop_path_rate (从0到drop_path_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers)] if drop_path_rate > 0 else [0.0] * n_layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, dpr[i])
            for i in range(n_layers)
        ])
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """逐层前向传播"""
        for layer in self.layers:
            x = layer(x, mask)
        return x


# ============================================================================
# 完整的 Transformer 时序预测模型
# ============================================================================
class StockTransformer(nn.Module):
    """
    基于Transformer的股票收益预测模型
    
    完整架构：
    1. 输入投影层: 将多特征输入投影到d_model维空间
    2. 位置编码: 添加时间位置信息
    3. Encoder: 提取历史序列的特征表示
    4. 全局池化: 聚合序列维度的信息
    5. 预测头: 多层感知机输出预测结果
    
    输出两个分支：
    - 回归分支: 预测未来N日收益率（连续值）
    - 分类分支: 预测涨跌方向（二分类）
    
    损失函数由两部分组成：
    - MSE Loss (回归): 衡量预测收益率与实际收益率的偏差
    - BCE Loss (分类): 衡量方向预测的准确性
    """
    
    def __init__(self, config: ModelConfig):
        """
        参数:
            config: 模型配置对象
        """
        super().__init__()
        
        self.config = config
        self.d_model = config.d_model
        n_features = len(config.features)
        
        # 1. 输入投影层
        # 将原始特征维度投影到d_model空间
        self.input_proj = nn.Linear(n_features, config.d_model)
        
        # 1b. 特征组权重：情感特征使用较小权重，降低对主信号的干扰
        sentiment_names = getattr(config, 'sentiment_features', [])
        feature_weights = torch.ones(n_features)
        for i, fname in enumerate(config.features):
            if fname in sentiment_names:
                feature_weights[i] = getattr(config, 'news_feature_weight', 0.05)
        self.register_buffer('feature_weights', feature_weights)
        
        # 2. 位置编码
        self.pos_encoding = PositionalEncoding(
            config.d_model, config.max_seq_len, config.dropout
        )
        
        # 3. Encoder (带Stochastic Depth)
        drop_path_rate = getattr(config, 'drop_path_rate', 0.1)
        self.encoder = TransformerEncoder(
            config.d_model, config.n_heads, config.d_ff,
            config.n_encoder_layers, config.dropout, drop_path_rate
        )
        
        # 最终 LayerNorm（Pre-LN 架构需要 Encoder 输出后的归一化）
        self.final_norm = nn.LayerNorm(config.d_model)


        # 4. 注意力池化（比平均池化更好地聚焦关键时间点）
        self.attn_pool_query = nn.Parameter(torch.randn(1, 1, config.d_model) * 0.02)
        self.attn_pool = nn.MultiheadAttention(
            config.d_model, num_heads=1, dropout=config.dropout, batch_first=True
        )
        
        # 5. 预测头
        # 回归分支：预测连续收益率
        self.regression_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model // 2, config.prediction_horizon),
        )
        
        # 分类分支：预测涨跌方向
        self.classification_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model // 2, config.prediction_horizon),
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self) -> None:
        """
        Kaiming初始化 (针对GELU激活函数优化)
        
        Kaiming初始化比Xavier更适合GELU/ReLU激活函数，
        能更好地保持前向传播的方差稳定性。
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播 (Pre-LN架构)
        
        参数:
            x: 输入特征张量
               形状 (batch_size, seq_len, n_features)
               seq_len 通常为 lookback_window (如60天)
            mask: 注意力掩码（可选）
        
        返回:
            regression_output: (batch_size, prediction_horizon)
                              预测的未来收益率
            classification_output: (batch_size, prediction_horizon)
                                  预测的涨跌logits
        """
        # 1. 特征投影（情感特征使用小权重）
        x = x * self.feature_weights.view(1, 1, -1)  # 应用分组权重
        x = self.input_proj(x)  # (batch, seq_len, d_model)
        
        # 2. 位置编码
        x = self.pos_encoding(x)
        
        # 3. Encoder编码 (Pre-LN在每层内部)
        x = self.encoder(x, mask)  # (batch, seq_len, d_model)
        # 最终LayerNorm（Pre-LN架构需要最后的归一化）
        x = self.final_norm(x)
        
        # 4. 注意力池化
        # 用可学习query对所有时间步做注意力加权聚合
        batch_size = x.size(0)
        query = self.attn_pool_query.expand(batch_size, -1, -1)
        x, _ = self.attn_pool(query, x, x)  # (batch, 1, d_model)
        x = x.squeeze(1)  # (batch, d_model)
        
        # 5. 预测
        regression_output = self.regression_head(x)  # (batch, horizon)
        classification_output = self.classification_head(x)  # (batch, horizon)
        
        return regression_output, classification_output
    
    def predict(
        self, x: torch.Tensor, return_probs: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        推理接口
        
        在评估模式下运行，不计算梯度，速度更快。
        
        参数:
            x: 输入特征 (batch, seq_len, n_features)
            return_probs: 是否返回方向概率（否则返回logits）
        
        返回:
            包含 'returns'(预测收益率) 和 'direction'(涨跌概率) 的字典
        """
        self.eval()
        
        with torch.no_grad():
            reg_out, cls_out = self.forward(x)
            
            result = {
                'returns': reg_out.cpu().numpy(),  # 预测收益率
            }
            
            if return_probs:
                # Sigmoid转换为概率
                result['direction'] = torch.sigmoid(cls_out).cpu().numpy()
            else:
                result['direction_logits'] = cls_out.cpu().numpy()
            
            return result


# ============================================================================
# 增强版：Encoder-Decoder Transformer（用于序列到序列预测）
# ============================================================================
class TimeSeriesTransformer(nn.Module):
    """
    Encoder-Decoder 架构的Transformer时序预测模型
    
    相比纯Encoder模型(StockTransformer)的优势：
    - 能直接输出预测序列而非单点值
    - Decoder的交叉注意力能更好对齐历史与未来
    - 适合多步预测（multi-step forecasting）
    
    使用场景：
    - 需要预测未来5-20天的完整走势
    - 需要预测多个目标（收益率+波动率）
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        
        self.config = config
        n_features = len(config.features)
        
        # 编码器输入投影
        self.encoder_proj = nn.Linear(n_features, config.d_model)
        
        # 解码器输入投影（预测目标的占位符）
        self.decoder_proj = nn.Linear(1, config.d_model)
        
        # 位置编码（编码器和解码器各一个）
        self.encoder_pos = PositionalEncoding(
            config.d_model, config.max_seq_len, config.dropout
        )
        self.decoder_pos = PositionalEncoding(
            config.d_model, config.max_seq_len, config.dropout
        )
        
        # Encoder (Pre-LN 架构)
        self.encoder = TransformerEncoder(
            config.d_model, config.n_heads, config.d_ff,
            config.n_encoder_layers, config.dropout
        )
        
        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(
                config.d_model, config.n_heads, config.d_ff, config.dropout
            )
            for _ in range(config.n_decoder_layers)
        ])
        
        # 输出投影
        self.output_proj = nn.Linear(config.d_model, 1)
        
        self._init_weights()
    
    def _init_weights(self) -> None:
        """权重初始化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def _generate_causal_mask(self, seq_len: int) -> torch.Tensor:
        """
        生成因果掩码（上三角矩阵）
        
        确保位置i只能关注位置0到i的信息，
        防止未来信息泄露。
        
        参数:
            seq_len: 序列长度
        
        返回:
            上三角掩码 (seq_len, seq_len)
        """
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        return ~mask  # True表示可以关注
    
    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播
        
        参数:
            src: 编码器输入（历史数据）
                 (batch, src_seq_len, n_features)
            tgt: 解码器输入（目标序列占位符）
                 (batch, tgt_seq_len, 1)
            src_mask: 编码器自注意力掩码
        
        返回:
            预测序列 (batch, tgt_seq_len, 1)
        """
        # 编码器
        src_embedded = self.encoder_proj(src)
        src_embedded = self.encoder_pos(src_embedded)
        encoder_output = self.encoder(src_embedded, src_mask)
        
        # 解码器
        tgt_embedded = self.decoder_proj(tgt)
        tgt_embedded = self.decoder_pos(tgt_embedded)
        
        tgt_seq_len = tgt_embedded.size(1)
        causal_mask = self._generate_causal_mask(tgt_seq_len).to(tgt.device)
        
        decoder_output = tgt_embedded
        for layer in self.decoder_layers:
            decoder_output = layer(
                decoder_output, encoder_output,
                causal_mask, src_mask
            )
        
        # 输出投影
        output = self.output_proj(decoder_output)
        return output


class TransformerDecoderLayer(nn.Module):
    """
    单层Transformer Decoder
    
    包含三个子层：
    1. Masked Self-Attention（因果自注意力）
    2. Cross-Attention（与Encoder输出的交叉注意力）
    3. Feed-Forward Network
    
    每层后都有Layer Norm和残差连接。
    """
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        self_mask: Optional[torch.Tensor] = None,
        cross_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # 1. Causally Masked Self-Attention
        attn_output = self.self_attn(
            self.norm1(x), self.norm1(x), self.norm1(x), self_mask
        )
        x = x + self.dropout1(attn_output)
        
        # 2. Cross-Attention with Encoder
        attn_output = self.cross_attn(
            self.norm2(x), encoder_output, encoder_output, cross_mask
        )
        x = x + self.dropout2(attn_output)
        
        # 3. Feed-Forward
        ff_output = self.feed_forward(self.norm3(x))
        x = x + self.dropout3(ff_output)
        
        return x
