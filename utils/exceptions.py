"""Custom exception classes."""

from typing import Optional, Any
from datetime import datetime, timezone


# ============================================================================
# 基礎異常類
# ============================================================================
class QuantError(Exception):
    """
    量化交易系統所有自定義異常的基類
    
    屬性:
        message: 人類可讀的錯誤描述
        timestamp: 異常發生時間（UTC）
        context: 額外的上下文信息字典
    """
    
    def __init__(self, message: str, context: Optional[dict] = None):
        self.message = message
        self.timestamp = datetime.now(timezone.utc).replace(tzinfo=None)
        self.context = context or {}
        super().__init__(self._format())
    
    def _format(self) -> str:
        """格式化異常信息，包含時間戳和上下文"""
        parts = [f"[{self.timestamp.isoformat()}] {self.__class__.__name__}: {self.message}"]
        if self.context:
            ctx_str = ', '.join(f'{k}={v}' for k, v in self.context.items())
            parts.append(f'  上下文: {ctx_str}')
        return '\n'.join(parts)


# ============================================================================
# 配置異常
# ============================================================================
class ConfigurationError(QuantError):
    """
    配置錯誤：缺少必要配置項、配置格式錯誤、配置值不合法等
    
    示例:
        raise ConfigurationError("IBKR API埠必須在1024-65535之間", {'port': 80})
    """
    pass


# ============================================================================
# 數據異常
# ============================================================================
class DataError(QuantError):
    """數據層異常的基類"""
    pass


class DataSourceError(DataError):
    """
    數據源錯誤：API連接失敗、響應超時、返回數據格式異常
    
    示例:
        raise DataSourceError("Polygon API返回403", {'status_code': 403, 'ticker': 'AAPL'})
    """
    pass


class DataQualityError(DataError):
    """
    數據質量錯誤：缺失值過多、異常值、數據不一致
    
    示例:
        raise DataQualityError("AAPL缺少2023-06-15的OHLCV數據", {'ticker': 'AAPL', 'date': '2023-06-15'})
    """
    pass


class DataAlignmentError(DataError):
    """
    數據對齊錯誤：多數據源時間戳不一致、復權方式衝突
    
    示例:
        raise DataAlignmentError("Yahoo和Polygon的AAPL復權因子不一致")
    """
    pass


# ============================================================================
# 模型異常
# ============================================================================
class ModelError(QuantError):
    """ML模型層異常的基類"""
    pass


class ModelTrainingError(ModelError):
    """
    模型訓練錯誤：梯度爆炸/消失、loss為NaN、OOM等
    
    示例:
        raise ModelTrainingError("訓練loss在第50輪變為NaN", {'epoch': 50})
    """
    pass


class ModelInferenceError(ModelError):
    """
    模型推理錯誤：輸入維度不匹配、模型未加載等
    
    示例:
        raise ModelInferenceError("模型期望256維輸入，但收到128維", {'expected': 256, 'got': 128})
    """
    pass


class LowAccuracyError(ModelError):
    """
    模型精度不足：方向預測準確率或RMSE不達標，觸發調參
    
    示例:
        raise LowAccuracyError("方向準確率0.48低於閾值0.55", {'accuracy': 0.48, 'threshold': 0.55})
    """
    pass


# ============================================================================
# 交易異常
# ============================================================================
class TradingError(QuantError):
    """交易層異常的基類"""
    pass


class OrderRejectedError(TradingError):
    """
    訂單被拒絕：資金不足、超出風控限制、交易時段限制、交易所拒絕
    
    示例:
        raise OrderRejectedError("超出單日最大成交次數", {'daily_count': 51, 'limit': 50})
    """
    pass


class OrderExecutionError(TradingError):
    """
    訂單執行錯誤：部分成交、超時未成交、連接斷開
    
    示例:
        raise OrderExecutionError("AAPL限價單超時未成交", {'order_id': 12345, 'ticker': 'AAPL'})
    """
    pass


class BrokerConnectionError(TradingError):
    """
    券商連接錯誤：TWS/Gateway斷開、認證失敗、API限流
    
    示例:
        raise BrokerConnectionError("IBKR TWS連接斷開", {'host': '127.0.0.1', 'port': 7497})
    """
    pass


# ============================================================================
# 風控異常
# ============================================================================
class RiskError(QuantError):
    """風控層異常的基類"""
    pass


class RiskLimitExceededError(RiskError):
    """
    風控限制被觸發：持倉超限、回撤超限、槓桿超限等
    
    示例:
        raise RiskLimitExceededError("回撤-28%超過最大25%限制", {'current_drawdown': -0.28, 'limit': -0.25})
    """
    pass


class CircuitBreakerError(RiskError):
    """
    熔斷觸發：個股LULD熔斷或全市場熔斷
    
    示例:
        raise CircuitBreakerError("AAPL觸發LULD Tier 1下跌熔斷", {'ticker': 'AAPL', 'band': 'down'})
    """
    pass


# ============================================================================
# 合規異常
# ============================================================================
class ComplianceError(QuantError):
    """合規層異常的基類"""
    pass


class WashSaleViolationError(ComplianceError):
    """
    潛在洗售違規：在30天窗口內買賣同一股票
    
    注意：這不是真正的"違規"而是需要標記並調整成本基礎
    """
    pass


class ShortSellLocateError(ComplianceError):
    """
    做空借券失敗：無法定位可借股票
    
    示例:
        raise ShortSellLocateError("無法為GME定位可借股票", {'ticker': 'GME'})
    """
    pass


class PDTRestrictionError(ComplianceError):
    """
    PDT限制被觸發：帳戶被標記為PDT且資金不足$25,000
    
    示例:
        raise PDTRestrictionError("帳戶被標記為PDT，資金$18,000不足$25,000", {'equity': 18000})
    """
    pass
