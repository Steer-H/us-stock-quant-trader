# backtesting/__init__.py
# 美股量化交易系统 - 回测引擎模块入口
from backtesting.engine import BacktestEngine, BacktestResult
from backtesting.broker_sim import BrokerSimulator, OrderFillSimulator
from backtesting.performance import PerformanceAnalyzer

__all__ = [
    'BacktestEngine', 'BacktestResult',
    'BrokerSimulator', 'OrderFillSimulator',
    'PerformanceAnalyzer',
]
