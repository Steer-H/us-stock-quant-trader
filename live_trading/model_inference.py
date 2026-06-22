"""
实时模型推理模块

加载训练好的Transformer模型（encoder-only架构），基于最新特征进行预测。
模型输出：5日方向分类 + 5日收益率回归。

使用方式:
    infer = ModelInference()
    signal = infer.predict('AAPL')  
    # → {'direction': 1, 'confidence': 0.72, 'predicted_return': 0.015}
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Dict, List
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import ModelConfig, PROCESSED_DATA_DIR, MODELS_DIR
from data_pipeline.storage import ParquetStorage
from ml_model.transformer import StockTransformer
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = MODELS_DIR / 'transformer_stock_latest.pt'  # 最新训练模型


# 直接使用训练时的StockTransformer架构（ml_model.transformer）
# 不再定义独立的InferenceTransformer，确保架构完全匹配

class ModelInference:
    """实时模型推理引擎"""
    
    def __init__(
        self,
        model_path: Optional[Path] = None,
        config: Optional[ModelConfig] = None,
        device: Optional[str] = None
    ):
        self.config = config or ModelConfig()
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        
        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = 'cuda'
        elif torch.backends.mps.is_available():
            self.device = 'mps'  # MPS may be unstable, use CPU for reliability
        else:
            self.device = 'cpu'
        # Force CPU for server stability (MPS has memory alignment bugs)
        if self.device == 'mps':
            try:
                import os
                if os.environ.get('FORCE_MPS') != '1':
                    self.device = 'cpu'
            except Exception:
                self.device = 'cpu'
        
        self.model: Optional[nn.Module] = None  # StockTransformer from ml_model.transformer
        self.scaler: Optional[StandardScaler] = None
        self._loaded = False
        self._ticker_features: Dict[str, np.ndarray] = {}
        self._feature_count: int = 24  # 默认，加载后更新
        
        logger.info(f"ModelInference: device={self.device}")
    
    def load(self) -> bool:
        """加载模型和scaler"""
        if self._loaded:
            return True
        
        if not self.model_path.exists():
            fallback = MODELS_DIR / 'transformer_26stocks.pt'
            if fallback.exists():
                logger.info(f'20股票模型不存在，使用26股票模型: {fallback}')
                self.model_path = fallback
            else:
                logger.warning(f"模型文件不存在: {self.model_path}")
                return False
        
        try:
            checkpoint = torch.load(self.model_path, map_location='cpu', weights_only=True)
            
            # 兼容不同保存格式
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            # state_dict已在上面获取，metadata用于元信息
            metadata = checkpoint.get('metadata', {})
            
            # 推断特征数
            input_proj_weight = state_dict.get('input_proj.weight')
            if input_proj_weight is not None:
                self._feature_count = input_proj_weight.shape[1]
                logger.info(f"特征数: {self._feature_count}")
            
            # 创建模型 (使用StockTransformer，与训练时完全一致)
            # 注意: StockTransformer使用config.features确定输入维度
            # 如果checkpoint特征数与config不一致，自动适配
            n_features_in_config = len(self.config.features)
            if self._feature_count != n_features_in_config:
                logger.warning(
                    f'特征数不匹配: checkpoint={self._feature_count}, config={n_features_in_config}. '
                    f'使用checkpoint的特征数创建模型。'
                )
                # 临时调整config特征数以匹配checkpoint
                # 注意: features列表可能更短，只取前_feature_count个
                self.config.features = (
                    self.config.features[:self._feature_count]
                    if n_features_in_config >= self._feature_count
                    else self.config.features + [f'extra_{j}' for j in range(self._feature_count - n_features_in_config)]
                )
            
            self.model = StockTransformer(self.config)
            
            matched, total = self.model.load_state_dict(state_dict, strict=False)
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"模型加载: {len(matched)} 参数匹配, "
                       f"缺失 {len(state_dict)-len(matched)} 键")
            
            # 恢复或创建scaler
            self._load_scaler()
            
            self._loaded = True
            return True
            
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _load_scaler(self):
        """加载或创建特征标准化器"""
        self.scaler = StandardScaler()
        storage = ParquetStorage(PROCESSED_DATA_DIR)
        
        # 自动发现所有已处理股票的特征数据
        all_tickers = set()
        for f in storage.list_keys():
            if '_features' in f:
                t = f.replace('_features.parquet', '').replace('_features', '')
                all_tickers.add(t)
        tickers = sorted(all_tickers)
        logger.info(f'Scaler拟合: 发现 {len(tickers)} 只股票的特征数据')
        
        for ticker in tickers:
            try:
                df = storage.load(f"{ticker}_features")
                if df is not None:
                    available_f = [f for f in self.config.features if f in df.columns]
                    features = df[available_f].dropna().values
                    features = features[:, :self._feature_count]
                    if len(features) > 100:
                        self.scaler.partial_fit(features)
            except Exception:
                continue
        
        if hasattr(self.scaler, 'n_features_in_'):
            logger.info(f"Scaler就绪: {self.scaler.n_features_in_} 特征")
        else:
            logger.warning("Scaler未拟合，将使用在线标准化")
    
    def load_ticker_features(self, ticker: str) -> bool:
        """加载单只股票特征"""
        if ticker in self._ticker_features:
            return True
        
        try:
            storage = ParquetStorage(PROCESSED_DATA_DIR)
            df = storage.load(f"{ticker}_features")
            if df is None or len(df) < 60:
                return False
            
            available_f = [f for f in self.config.features if f in df.columns]
            features = df[available_f].dropna().values
            # 如果实际特征数少于模型期望，用零填充缺失列
            if features.shape[1] < self._feature_count:
                pad = np.zeros((features.shape[0], self._feature_count - features.shape[1]))
                features = np.hstack([features[:, :self._feature_count], pad])
            else:
                features = features[:, :self._feature_count]
            self._ticker_features[ticker] = features
            return True
        except Exception as e:
            logger.debug(f"加载 {ticker} 特征失败: {e}")
            return False
    
    def get_model_info(self) -> dict:
        """返回模型信息供前端展示"""
        info = {
            'loaded': self._loaded,
            'feature_count': self._feature_count,
            'features': self.config.features[:self._feature_count] if self._loaded else [],
            'sentiment_features': [f for f in (self.config.features[:self._feature_count] if self._loaded else [])
                                   if f in getattr(self.config, 'sentiment_features', [])],
            'd_model': self.config.d_model,
            'n_heads': self.config.n_heads,
            'n_layers': self.config.n_encoder_layers,
            'device': str(self.device),
            'model_path': str(self.model_path),
            'model_size_mb': round(self.model_path.stat().st_size / 1024 / 1024, 1) if self.model_path.exists() else 0,
            'has_sentiment': self._loaded and any(
                'sentiment' in f or 'earnings' in f
                for f in (self.config.features[:self._feature_count] if self._loaded else [])
            ),
            # Training parameters
            'epochs': getattr(self.config, 'epochs', 0),
            'batch_size': getattr(self.config, 'batch_size', 0),
            'learning_rate': getattr(self.config, 'learning_rate', 0),
            'lookback_window': getattr(self.config, 'lookback_window', 0),
        }
        return info

    def predict(self, ticker: str) -> Optional[Dict]:
        """
        对单只股票生成预测
        
        返回:
            {'direction': 1, 'confidence': 0.72, 'predicted_return': 0.015}
            或 None
        """
        if not self._loaded and not self.load():
            return None
        if not self.load_ticker_features(ticker):
            return None
        
        try:
            features = self._ticker_features[ticker]
            window = self.config.lookback_window
            
            if len(features) < window:
                return None
            
            recent = features[-window:, :self._feature_count]
            
            if self.scaler and hasattr(self.scaler, 'mean_'):
                if recent.shape[1] == len(self.scaler.mean_):
                    recent = self.scaler.transform(recent)
                else:
                    recent = (recent - recent.mean(axis=0)) / (recent.std(axis=0) + 1e-8)
            else:
                recent = (recent - recent.mean(axis=0)) / (recent.std(axis=0) + 1e-8)
            
            src = torch.FloatTensor(recent).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                reg_out, cls_out = self.model(src)
            
            # cls_out: (batch, horizon) logits → sigmoid → direction probability
            cls_probs = torch.sigmoid(cls_out).squeeze().cpu().numpy()
            reg_vals = reg_out.squeeze().cpu().numpy()
            
            # 方向预测: 取5日平均概率
            avg_prob = float(np.mean(cls_probs))
            direction = 1 if avg_prob > 0.5 else 0
            predicted_return = float(np.mean(reg_vals))
            
            # 置信度: 距离0.5的偏差映射到0.50-0.95
            confidence = 0.50 + min(abs(avg_prob - 0.5) * 0.9, 0.45)
            
            return {
                'direction': direction,
                'confidence': round(confidence, 4),
                'predicted_return': round(predicted_return, 6),
                'avg_prob': round(avg_prob, 4),
            }
            
        except Exception as e:
            logger.debug(f"预测 {ticker} 失败: {e}")
            return None
    
    def batch_predict(self, tickers: List[str]) -> Dict[str, Optional[Dict]]:
        return {t: self.predict(t) for t in tickers}
    
    def is_ready(self) -> bool:
        return self._loaded


_inference: Optional[ModelInference] = None

def get_inference() -> ModelInference:
    global _inference
    if _inference is None:
        _inference = ModelInference()
        _inference.load()
    return _inference
