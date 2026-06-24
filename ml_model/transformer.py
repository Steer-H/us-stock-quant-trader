"""Transformer model for stock time series prediction."""

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
# 位置編碼 (Positional Encoding)
# ============================================================================
class PositionalEncoding(nn.Module):
    """
    可學習位置編碼 (Learnable Positional Encoding)
    
    與固定正弦編碼不同，可學習位置編碼允許模型根據金融數據的
    特性自適應調整位置表示。這對捕捉財報周期、季節性模式等
    非均勻時間結構特別重要。
    
    優勢:
    - 自適應: 根據數據分布學習最優位置表示
    - 靈活: 不受固定頻率假設限制
    - 兼容: 初始化時可從正弦編碼warm-start
    
    時間複雜度: O(1) per forward
    空間複雜度: O(max_len * d_model)
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
# 多頭自注意力 (Multi-Head Self-Attention)
# ============================================================================

# ============================================================================
# Stochastic Depth (DropPath)
# 參考: "Deep Networks with Stochastic Depth" (Huang et al., 2016)
# 隨機丟棄整個殘差分支，比普通Dropout更強的正則化
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
    多頭自注意力機制
    
    將輸入投影到多個注意力頭，每個頭獨立計算縮放點積注意力，
    最後拼接並投影回原始維度。
    
    核心公式：
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
    
    多頭設計允許模型同時關注不同子空間的信息：
    - 短周期頭：關注近期價格變化
    - 長周期頭：關注長期趨勢
    - 交叉頭：關注特徵之間的交互
    
    時間複雜度: O(batch * seq_len^2 * d_model)
    空間複雜度: O(batch * n_heads * seq_len^2)  # 注意力矩陣
    
    參數:
        d_model: 模型維度
        n_heads: 注意力頭數
        dropout: Attention dropout率
    """
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        
        assert d_model % n_heads == 0, \
            f"d_model({d_model})必須能被n_heads({n_heads})整除"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # 每個頭的維度
        
        # Q, K, V的線性投影矩陣
        # 三個合在一起以減少計算開銷
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        
        # 輸出投影矩陣
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        # 縮放因子: 1/sqrt(d_k)，避免點積過大導致softmax梯度消失
        self.scale = math.sqrt(self.d_k)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向傳播
        
        參數:
            query:  (batch, seq_len_q, d_model)
            key:    (batch, seq_len_k, d_model)
            value:  (batch, seq_len_v, d_model)
            mask:   (batch, seq_len_q, seq_len_k) 或 (seq_len_q, seq_len_k)
                    可選掩碼，用於因果掩碼或填充掩碼
        
        返回:
            注意力輸出，形狀 (batch, seq_len_q, d_model)
        """
        batch_size = query.size(0)
        seq_len_q = query.size(1)
        seq_len_k = key.size(1)
        
        # 1. 線性投影並分割成多頭
        # self-attention: query==key==value, 一次投影即得QKV
        # cross-attention: key/value來自不同輸入，需分別投影
        qkv = self.qkv_proj(query).view(
            batch_size, seq_len_q, 3, self.n_heads, self.d_k
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, n_heads, seq, d_k)
        
        if query is key and key is value:
            # self-attention: 所有QKV來自同一次投影
            q, k, v = qkv[0], qkv[1], qkv[2]
        else:
            # cross-attention: K和V來自不同輸入
            q = qkv[0]
            k = self.qkv_proj(key).view(
                batch_size, seq_len_k, 3, self.n_heads, self.d_k
            ).permute(2, 0, 3, 1, 4)[1]
            v = self.qkv_proj(value).view(
                batch_size, seq_len_k, 3, self.n_heads, self.d_k
            ).permute(2, 0, 3, 1, 4)[2]
        
        # 2. 計算縮放點積注意力
        # attn_scores: (batch, n_heads, seq_len_q, seq_len_k)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        # 3. 應用掩碼（如果有）
        if mask is not None:
            # mask中值為True的位置將被設為 -inf，softmax後為0
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        # 4. Softmax得到注意力權重
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 5. 加權聚合Value
        # (batch, n_heads, seq_len_q, d_k)
        attn_output = torch.matmul(attn_weights, v)
        
        # 6. 合併多頭並投影
        # (batch, n_heads, seq_len_q, d_k) → (batch, seq_len_q, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len_q, self.d_model
        )
        
        return self.out_proj(attn_output)


# ============================================================================
# 前饋網絡 (Feed-Forward Network)
# ============================================================================
class FeedForward(nn.Module):
    """
    位置獨立的前饋網絡
    
    結構: Linear → ReLU → Dropout → Linear → Dropout
    
    兩個線性層之間使用ReLU激活函數，中間維度通常為d_model的4倍。
    這個膨脹-收縮結構提供了非線性變換能力。
    
    時間複雜度: O(batch * seq_len * d_model * d_ff)
    空間複雜度: O(batch * seq_len * d_ff)
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        """
        參數:
            d_model: 輸入/輸出維度
            d_ff: 隱藏層維度（通常是d_model的4倍）
            dropout: Dropout率
        """
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        參數:
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
# Transformer Encoder 層
# ============================================================================
class TransformerEncoderLayer(nn.Module):
    """
    單層Transformer Encoder
    
    結構（Pre-LN設計，更穩定的訓練）：
    ┌──────────────────────┐
    │ 輸入 x                │
    │   ↓                   │
    │  LayerNorm(x)         │
    │   ↓                   │
    │  MultiHeadAttn(x,x,x) │
    │   ↓                   │
    │  Dropout + 殘差連接   │
    │   ↓                   │
    │  LayerNorm(x')        │
    │   ↓                   │
    │  FeedForward(x')      │
    │   ↓                   │
    │  Dropout + 殘差連接   │
    │   ↓                   │
    │  輸出                  │
    └──────────────────────┘
    
    使用Pre-LN (LayerNorm before sublayer) 而非Post-LN，
    在深層網絡中更穩定，不易出現訓練不收斂的問題。
    """
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, 
                 dropout: float = 0.1, drop_path: float = 0.0):
        super().__init__()
        
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        # LayerNorm 對各特徵維度獨立歸一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        參數:
            x: (batch, seq_len, d_model)
            mask: 自注意力掩碼
        
        返回:
            (batch, seq_len, d_model)
        """
        # 自注意力 + 殘差連接 + DropPath
        attn_output = self.self_attn(
            self.norm1(x), self.norm1(x), self.norm1(x), mask
        )
        x = x + self.drop_path(self.dropout1(attn_output))
        
        # 前饋網絡 + 殘差連接 + DropPath
        ff_output = self.feed_forward(self.norm2(x))
        x = x + self.drop_path(self.dropout2(ff_output))
        
        return x


# ============================================================================
# Transformer Encoder（多層堆疊）
# ============================================================================
class TransformerEncoder(nn.Module):
    """
    多層Transformer Encoder堆疊
    
    每層的輸出作為下一層的輸入，逐層提取更高級的特徵。
    淺層捕捉局部模式（如短期趨勢），深層捕捉全局模式（如長期周期）。
    """
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, 
                 n_layers: int, dropout: float = 0.1, drop_path_rate: float = 0.0):
        super().__init__()
        # 線性增加的drop_path_rate (從0到drop_path_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers)] if drop_path_rate > 0 else [0.0] * n_layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, dpr[i])
            for i in range(n_layers)
        ])
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """逐層前向傳播"""
        for layer in self.layers:
            x = layer(x, mask)
        return x


# ============================================================================
# 完整的 Transformer 時序預測模型
# ============================================================================
class StockTransformer(nn.Module):
    """
    基於Transformer的股票收益預測模型
    
    完整架構：
    1. 輸入投影層: 將多特徵輸入投影到d_model維空間
    2. 位置編碼: 添加時間位置信息
    3. Encoder: 提取歷史序列的特徵表示
    4. 全局池化: 聚合序列維度的信息
    5. 預測頭: 多層感知機輸出預測結果
    
    輸出兩個分支：
    - 回歸分支: 預測未來N日收益率（連續值）
    - 分類分支: 預測漲跌方向（二分類）
    
    損失函數由兩部分組成：
    - MSE Loss (回歸): 衡量預測收益率與實際收益率的偏差
    - BCE Loss (分類): 衡量方向預測的準確性
    """
    
    def __init__(self, config: ModelConfig):
        """
        參數:
            config: 模型配置對象
        """
        super().__init__()
        
        self.config = config
        self.d_model = config.d_model
        n_features = len(config.features)
        
        # 1. 輸入投影層
        # 將原始特徵維度投影到d_model空間
        self.input_proj = nn.Linear(n_features, config.d_model)
        
        # 1b. 特徵組權重：情感特徵使用較小權重，降低對主信號的幹擾
        sentiment_names = getattr(config, 'sentiment_features', [])
        feature_weights = torch.ones(n_features)
        for i, fname in enumerate(config.features):
            if fname in sentiment_names:
                feature_weights[i] = getattr(config, 'news_feature_weight', 0.05)
        self.register_buffer('feature_weights', feature_weights)
        
        # 2. 位置編碼
        self.pos_encoding = PositionalEncoding(
            config.d_model, config.max_seq_len, config.dropout
        )
        
        # 3. Encoder (帶Stochastic Depth)
        drop_path_rate = getattr(config, 'drop_path_rate', 0.1)
        self.encoder = TransformerEncoder(
            config.d_model, config.n_heads, config.d_ff,
            config.n_encoder_layers, config.dropout, drop_path_rate
        )
        
        # 最終 LayerNorm（Pre-LN 架構需要 Encoder 輸出後的歸一化）
        self.final_norm = nn.LayerNorm(config.d_model)


        # 4. 注意力池化（比平均池化更好地聚焦關鍵時間點）
        self.attn_pool_query = nn.Parameter(torch.randn(1, 1, config.d_model) * 0.02)
        self.attn_pool = nn.MultiheadAttention(
            config.d_model, num_heads=1, dropout=config.dropout, batch_first=True
        )
        
        # 5. 預測頭
        # 回歸分支：預測連續收益率
        self.regression_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model // 2, config.prediction_horizon),
        )
        
        # 分類分支：預測漲跌方向
        self.classification_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model // 2, config.prediction_horizon),
        )
        
        # 初始化權重
        self._init_weights()
    
    def _init_weights(self) -> None:
        """
        Kaiming初始化 (針對GELU激活函數優化)
        
        Kaiming初始化比Xavier更適合GELU/ReLU激活函數，
        能更好地保持前向傳播的方差穩定性。
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
        前向傳播 (Pre-LN架構)
        
        參數:
            x: 輸入特徵張量
               形狀 (batch_size, seq_len, n_features)
               seq_len 通常為 lookback_window (如60天)
            mask: 注意力掩碼（可選）
        
        返回:
            regression_output: (batch_size, prediction_horizon)
                              預測的未來收益率
            classification_output: (batch_size, prediction_horizon)
                                  預測的漲跌logits
        """
        # 1. 特徵投影（情感特徵使用小權重）
        x = x * self.feature_weights.view(1, 1, -1)  # 應用分組權重
        x = self.input_proj(x)  # (batch, seq_len, d_model)
        
        # 2. 位置編碼
        x = self.pos_encoding(x)
        
        # 3. Encoder編碼 (Pre-LN在每層內部)
        x = self.encoder(x, mask)  # (batch, seq_len, d_model)
        # 最終LayerNorm（Pre-LN架構需要最後的歸一化）
        x = self.final_norm(x)
        
        # 4. 注意力池化
        # 用可學習query對所有時間步做注意力加權聚合
        batch_size = x.size(0)
        query = self.attn_pool_query.expand(batch_size, -1, -1)
        x, _ = self.attn_pool(query, x, x)  # (batch, 1, d_model)
        x = x.squeeze(1)  # (batch, d_model)
        
        # 5. 預測
        regression_output = self.regression_head(x)  # (batch, horizon)
        classification_output = self.classification_head(x)  # (batch, horizon)
        
        return regression_output, classification_output
    
    def predict(
        self, x: torch.Tensor, return_probs: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        推理接口
        
        在評估模式下運行，不計算梯度，速度更快。
        
        參數:
            x: 輸入特徵 (batch, seq_len, n_features)
            return_probs: 是否返回方向概率（否則返回logits）
        
        返回:
            包含 'returns'(預測收益率) 和 'direction'(漲跌概率) 的字典
        """
        self.eval()
        
        with torch.no_grad():
            reg_out, cls_out = self.forward(x)
            
            result = {
                'returns': reg_out.cpu().numpy(),  # 預測收益率
            }
            
            if return_probs:
                # Sigmoid轉換為概率
                result['direction'] = torch.sigmoid(cls_out).cpu().numpy()
            else:
                result['direction_logits'] = cls_out.cpu().numpy()
            
            return result


# ============================================================================
# 增強版：Encoder-Decoder Transformer（用於序列到序列預測）
# ============================================================================
class TimeSeriesTransformer(nn.Module):
    """
    Encoder-Decoder 架構的Transformer時序預測模型
    
    相比純Encoder模型(StockTransformer)的優勢：
    - 能直接輸出預測序列而非單點值
    - Decoder的交叉注意力能更好對齊歷史與未來
    - 適合多步預測（multi-step forecasting）
    
    使用場景：
    - 需要預測未來5-20天的完整走勢
    - 需要預測多個目標（收益率+波動率）
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        
        self.config = config
        n_features = len(config.features)
        
        # 編碼器輸入投影
        self.encoder_proj = nn.Linear(n_features, config.d_model)
        
        # 解碼器輸入投影（預測目標的佔位符）
        self.decoder_proj = nn.Linear(1, config.d_model)
        
        # 位置編碼（編碼器和解碼器各一個）
        self.encoder_pos = PositionalEncoding(
            config.d_model, config.max_seq_len, config.dropout
        )
        self.decoder_pos = PositionalEncoding(
            config.d_model, config.max_seq_len, config.dropout
        )
        
        # Encoder (Pre-LN 架構)
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
        
        # 輸出投影
        self.output_proj = nn.Linear(config.d_model, 1)
        
        self._init_weights()
    
    def _init_weights(self) -> None:
        """權重初始化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def _generate_causal_mask(self, seq_len: int) -> torch.Tensor:
        """
        生成因果掩碼（上三角矩陣）
        
        確保位置i只能關注位置0到i的信息，
        防止未來信息洩露。
        
        參數:
            seq_len: 序列長度
        
        返回:
            上三角掩碼 (seq_len, seq_len)
        """
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        return ~mask  # True表示可以關注
    
    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向傳播
        
        參數:
            src: 編碼器輸入（歷史數據）
                 (batch, src_seq_len, n_features)
            tgt: 解碼器輸入（目標序列佔位符）
                 (batch, tgt_seq_len, 1)
            src_mask: 編碼器自注意力掩碼
        
        返回:
            預測序列 (batch, tgt_seq_len, 1)
        """
        # 編碼器
        src_embedded = self.encoder_proj(src)
        src_embedded = self.encoder_pos(src_embedded)
        encoder_output = self.encoder(src_embedded, src_mask)
        
        # 解碼器
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
        
        # 輸出投影
        output = self.output_proj(decoder_output)
        return output


class TransformerDecoderLayer(nn.Module):
    """
    單層Transformer Decoder
    
    包含三個子層：
    1. Masked Self-Attention（因果自注意力）
    2. Cross-Attention（與Encoder輸出的交叉注意力）
    3. Feed-Forward Network
    
    每層後都有Layer Norm和殘差連接。
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
