"""
美股量化交易系统 - 自定义异常层次结构

设计原则：
- 所有自定义异常继承自 QuantError 基类
- 按模块分层，便于精确定位错误来源
- 每个异常携带足够的上下文信息，便于排查
"""

from typing import Optional, Any
from datetime import datetime, timezone


# ============================================================================
# 基础异常类
# ============================================================================
class QuantError(Exception):
    """
    量化交易系统所有自定义异常的基类
    
    属性:
        message: 人类可读的错误描述
        timestamp: 异常发生时间（UTC）
        context: 额外的上下文信息字典
    """
    
    def __init__(self, message: str, context: Optional[dict] = None):
        self.message = message
        self.timestamp = datetime.now(timezone.utc).replace(tzinfo=None)
        self.context = context or {}
        super().__init__(self._format())
    
    def _format(self) -> str:
        """格式化异常信息，包含时间戳和上下文"""
        parts = [f"[{self.timestamp.isoformat()}] {self.__class__.__name__}: {self.message}"]
        if self.context:
            ctx_str = ', '.join(f'{k}={v}' for k, v in self.context.items())
            parts.append(f'  上下文: {ctx_str}')
        return '\n'.join(parts)


# ============================================================================
# 配置异常
# ============================================================================
class ConfigurationError(QuantError):
    """
    配置错误：缺少必要配置项、配置格式错误、配置值不合法等
    
    示例:
        raise ConfigurationError("IBKR API端口必须在1024-65535之间", {'port': 80})
    """
    pass


# ============================================================================
# 数据异常
# ============================================================================
class DataError(QuantError):
    """数据层异常的基类"""
    pass


class DataSourceError(DataError):
    """
    数据源错误：API连接失败、响应超时、返回数据格式异常
    
    示例:
        raise DataSourceError("Polygon API返回403", {'status_code': 403, 'ticker': 'AAPL'})
    """
    pass


class DataQualityError(DataError):
    """
    数据质量错误：缺失值过多、异常值、数据不一致
    
    示例:
        raise DataQualityError("AAPL缺少2023-06-15的OHLCV数据", {'ticker': 'AAPL', 'date': '2023-06-15'})
    """
    pass


class DataAlignmentError(DataError):
    """
    数据对齐错误：多数据源时间戳不一致、复权方式冲突
    
    示例:
        raise DataAlignmentError("Yahoo和Polygon的AAPL复权因子不一致")
    """
    pass


# ============================================================================
# 模型异常
# ============================================================================
class ModelError(QuantError):
    """ML模型层异常的基类"""
    pass


class ModelTrainingError(ModelError):
    """
    模型训练错误：梯度爆炸/消失、loss为NaN、OOM等
    
    示例:
        raise ModelTrainingError("训练loss在第50轮变为NaN", {'epoch': 50})
    """
    pass


class ModelInferenceError(ModelError):
    """
    模型推理错误：输入维度不匹配、模型未加载等
    
    示例:
        raise ModelInferenceError("模型期望256维输入，但收到128维", {'expected': 256, 'got': 128})
    """
    pass


class LowAccuracyError(ModelError):
    """
    模型精度不足：方向预测准确率或RMSE不达标，触发调参
    
    示例:
        raise LowAccuracyError("方向准确率0.48低于阈值0.55", {'accuracy': 0.48, 'threshold': 0.55})
    """
    pass


# ============================================================================
# 交易异常
# ============================================================================
class TradingError(QuantError):
    """交易层异常的基类"""
    pass


class OrderRejectedError(TradingError):
    """
    订单被拒绝：资金不足、超出风控限制、交易时段限制、交易所拒绝
    
    示例:
        raise OrderRejectedError("超出单日最大成交次数", {'daily_count': 51, 'limit': 50})
    """
    pass


class OrderExecutionError(TradingError):
    """
    订单执行错误：部分成交、超时未成交、连接断开
    
    示例:
        raise OrderExecutionError("AAPL限价单超时未成交", {'order_id': 12345, 'ticker': 'AAPL'})
    """
    pass


class BrokerConnectionError(TradingError):
    """
    券商连接错误：TWS/Gateway断开、认证失败、API限流
    
    示例:
        raise BrokerConnectionError("IBKR TWS连接断开", {'host': '127.0.0.1', 'port': 7497})
    """
    pass


# ============================================================================
# 风控异常
# ============================================================================
class RiskError(QuantError):
    """风控层异常的基类"""
    pass


class RiskLimitExceededError(RiskError):
    """
    风控限制被触发：持仓超限、回撤超限、杠杆超限等
    
    示例:
        raise RiskLimitExceededError("回撤-28%超过最大25%限制", {'current_drawdown': -0.28, 'limit': -0.25})
    """
    pass


class CircuitBreakerError(RiskError):
    """
    熔断触发：个股LULD熔断或全市场熔断
    
    示例:
        raise CircuitBreakerError("AAPL触发LULD Tier 1下跌熔断", {'ticker': 'AAPL', 'band': 'down'})
    """
    pass


# ============================================================================
# 合规异常
# ============================================================================
class ComplianceError(QuantError):
    """合规层异常的基类"""
    pass


class WashSaleViolationError(ComplianceError):
    """
    潜在洗售违规：在30天窗口内买卖同一股票
    
    注意：这不是真正的"违规"而是需要标记并调整成本基础
    """
    pass


class ShortSellLocateError(ComplianceError):
    """
    做空借券失败：无法定位可借股票
    
    示例:
        raise ShortSellLocateError("无法为GME定位可借股票", {'ticker': 'GME'})
    """
    pass


class PDTRestrictionError(ComplianceError):
    """
    PDT限制被触发：账户被标记为PDT且资金不足$25,000
    
    示例:
        raise PDTRestrictionError("账户被标记为PDT，资金$18,000不足$25,000", {'equity': 18000})
    """
    pass
