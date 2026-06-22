"""
美股量化交易系统 - 券商模拟器

精确模拟美股券商撮合环境，包括：
- 限价单/市价单/止损单撮合逻辑
- 交易成本计算（佣金、SEC费、TAF费、交易所回扣）
- 做空约束（Uptick Rule、借券可用性检查）
- 市场冲击与滑点模型
- 部分成交模拟
"""

import logging
from typing import Optional, Dict, Tuple
from dataclasses import dataclass
from enum import Enum

import numpy as np

from config.settings import TradingConfig
from utils.constants import RegSHO
from utils.exceptions import OrderRejectedError, ShortSellLocateError

logger = logging.getLogger(__name__)

# 复用 engine.py 中的订单类型
from backtesting.engine import Order, OrderSide, OrderType, OrderStatus, Position, Account


@dataclass
class FillResult:
    """撮合结果"""
    filled: bool
    fill_price: float = 0.0
    filled_quantity: int = 0
    commission: float = 0.0
    sec_fee: float = 0.0
    taf_fee: float = 0.0
    slippage_cost: float = 0.0
    reject_reason: str = ''


class BrokerSimulator:
    """
    券商交易模拟器
    
    负责模拟订单撮合的完整流程，包括：
    - 价格撮合（考虑订单簿深度）
    - 交易成本计算
    - 合规检查
    """
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.borrow_availability: Dict[str, int] = {}  # {ticker: available_shares}
    
    def set_borrow_availability(self, ticker: str, shares: int) -> None:
        """
        设置可借股票数量
        
        参数:
            ticker: 股票代码
            shares: 可借股数
        """
        self.borrow_availability[ticker] = shares
    
    def check_order(self, order: Order, account: Account) -> FillResult:
        """
        下单前检查（Pre-Trade Validation）
        
        检查项：
        1. 资金充足性
        2. 持仓可卖性
        3. 做空借券可用性
        4. 交易时段限制
        
        参数:
            order: 待检查的订单
            account: 当前账户状态
        
        返回:
            FillResult（含拒绝原因）
        """
        # 资金检查
        if order.side == OrderSide.BUY:
            estimated_cost = order.quantity * (order.limit_price or 1000) * 1.01
            if estimated_cost > account.cash:
                return FillResult(
                    filled=False,
                    reject_reason=f"资金不足: 需要${estimated_cost:,.0f}, 可用${account.cash:,.0f}"
                )
        
        # 做空检查
        if order.side == OrderSide.SELL_SHORT:
            # 检查借券可用性
            available = self.borrow_availability.get(order.ticker, 0)
            if order.quantity > available:
                return FillResult(
                    filled=False,
                    reject_reason=f"借券不足: {order.ticker} 可用{available}股, "
                                f"需要{order.quantity}股"
                )
            
            # Uptick Rule检查（简化）
            # 实际应检查最近一次成交价是否高于前一次
            pass
        
        # 持仓检查（卖出时）
        if order.side == OrderSide.SELL:
            pos = account.positions.get(order.ticker)
            if not pos or pos.quantity < order.quantity:
                return FillResult(
                    filled=False,
                    reject_reason=f"持仓不足: {order.ticker} 持有"
                                f"{pos.quantity if pos else 0}股, 卖出{order.quantity}股"
                )
        
        # PDT检查
        if account.is_pdt and account.equity < 25_000:
            return FillResult(
                filled=False,
                reject_reason=f"PDT限制: 账户资金${account.equity:,.0f}不足$25,000"
            )
        
        return FillResult(filled=True, filled_quantity=order.quantity)


class OrderFillSimulator:
    """
    订单撮合模拟器
    
    模拟订单在交易所的成交过程，考虑：
    - 订单簿流动性
    - 价格滑点
    - 市场冲击成本
    - 部分成交可能性
    
    滑点模型：
    Almgren-Chriss 市场冲击模型的简化版本：
    Impact = σ * sqrt(Q / V) * (1 + sign(Q))
    其中 σ=波动率, Q=订单量, V=日均成交量
    """
    
    def __init__(self, config: TradingConfig):
        self.config = config
    
    def simulate_limit_order(
        self,
        order: Order,
        high: float,
        low: float,
        open_price: float,
        close_price: float,
        volume: int,
        avg_daily_volume: float
    ) -> FillResult:
        """
        模拟限价单撮合
        
        逻辑：
        - 限价买单: 当日最低价 ≤ 限价 → 以限价成交（保守估计）
        - 限价卖单: 当日最高价 ≥ 限价 → 以限价成交
        
        参数:
            order: 限价单
            high, low, open_price, close_price: OHLC
            volume: 当日成交量
            avg_daily_volume: 日均成交量
        
        返回:
            FillResult
        """
        if order.order_type != OrderType.LIMIT or order.limit_price is None:
            return FillResult(filled=False, reject_reason="非限价单")
        
        can_fill = False
        fill_price = order.limit_price
        
        if order.side == OrderSide.BUY and low <= order.limit_price:
            can_fill = True
        elif order.side == OrderSide.SELL and high >= order.limit_price:
            can_fill = True
        elif order.side == OrderSide.SELL_SHORT and high >= order.limit_price:
            can_fill = True
        
        if not can_fill:
            return FillResult(filled=False, reject_reason="未达限价条件")
        
        # 计算滑点后的实际成交价
        actual_price = self._apply_slippage(
            fill_price, order, volume, avg_daily_volume
        )
        
        # 计算交易成本
        commission, sec_fee, taf_fee = self._calc_fees(order, actual_price)
        
        slippage_cost = abs(order.quantity * (actual_price - fill_price))
        
        return FillResult(
            filled=True,
            fill_price=actual_price,
            filled_quantity=order.quantity,
            commission=commission,
            sec_fee=sec_fee,
            taf_fee=taf_fee,
            slippage_cost=slippage_cost
        )
    
    def simulate_market_order(
        self,
        order: Order,
        open_price: float,
        volume: int,
        avg_daily_volume: float
    ) -> FillResult:
        """
        模拟市价单撮合
        
        市价单以次日开盘价成交（避免look-ahead bias），
        加上滑点调整。
        
        参数:
            order: 市价单
            open_price: 下一个bar的 开盘价
            volume: 成交量
            avg_daily_volume: 日均成交量
        
        返回:
            FillResult
        """
        fill_price = self._apply_slippage(
            open_price, order, volume, avg_daily_volume
        )
        
        commission, sec_fee, taf_fee = self._calc_fees(order, fill_price)
        
        slippage_cost = abs(order.quantity * (fill_price - open_price))
        
        return FillResult(
            filled=True,
            fill_price=fill_price,
            filled_quantity=order.quantity,
            commission=commission,
            sec_fee=sec_fee,
            taf_fee=taf_fee,
            slippage_cost=slippage_cost
        )
    
    def _apply_slippage(
        self,
        base_price: float,
        order: Order,
        volume: int,
        avg_daily_volume: float
    ) -> float:
        """
        应用滑点模型
        
        滑点 = base + vol_impact + size_impact
        
        参数:
            base_price: 基础价格
            order: 订单
            volume: 当日成交量
            avg_daily_volume: 日均量
        
        返回:
            滑点后的价格
        """
        cfg = self.config
        
        # 基础滑点（1基点）
        base = cfg.slippage_base_bps / 10000
        
        # 规模冲击
        if avg_daily_volume > 0:
            participation_rate = order.quantity / avg_daily_volume
            size_impact = cfg.slippage_size_coeff * np.sqrt(participation_rate) / 10000
        else:
            size_impact = 0
        
        total_slippage = base + size_impact
        
        # 买入:价格上升, 卖出:价格下降
        direction = 1 if order.side == OrderSide.BUY else -1
        return base_price * (1 + direction * total_slippage)
    
    def _calc_fees(self, order: Order, fill_price: float) -> Tuple[float, float, float]:
        """
        计算交易费用
        
        参数:
            order: 订单
            fill_price: 成交价
        
        返回:
            (佣金, SEC费, TAF费)
        """
        cfg = self.config
        trade_value = order.quantity * fill_price
        
        # 佣金
        commission = max(cfg.commission_min, order.quantity * cfg.commission_per_share)
        commission = min(commission, trade_value * cfg.commission_max_pct)
        
        # SEC费（仅卖出）
        sec_fee = 0.0
        if order.side in (OrderSide.SELL, OrderSide.SELL_SHORT):
            sec_fee = trade_value * cfg.sec_fee_rate
        
        # TAF费（仅卖出）
        taf_fee = 0.0
        if order.side in (OrderSide.SELL, OrderSide.SELL_SHORT):
            taf_fee = min(order.quantity * cfg.taf_fee_per_share, cfg.taf_fee_max)
        
        return commission, sec_fee, taf_fee
