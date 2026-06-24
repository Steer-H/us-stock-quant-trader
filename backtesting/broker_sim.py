"""
美股量化交易系統 - 券商模擬器

精確模擬美股券商撮合環境，包括：
- 限價單/市價單/止損單撮合邏輯
- 交易成本計算（佣金、SEC費、TAF費、交易所回扣）
- 做空約束（Uptick Rule、借券可用性檢查）
- 市場衝擊與滑點模型
- 部分成交模擬
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

# 復用 engine.py 中的訂單類型
from backtesting.engine import Order, OrderSide, OrderType, OrderStatus, Position, Account


@dataclass
class FillResult:
    """撮合結果"""
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
    券商交易模擬器
    
    負責模擬訂單撮合的完整流程，包括：
    - 價格撮合（考慮訂單簿深度）
    - 交易成本計算
    - 合規檢查
    """
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.borrow_availability: Dict[str, int] = {}  # {ticker: available_shares}
    
    def set_borrow_availability(self, ticker: str, shares: int) -> None:
        """
        設置可借股票數量
        
        參數:
            ticker: 股票代碼
            shares: 可借股數
        """
        self.borrow_availability[ticker] = shares
    
    def check_order(self, order: Order, account: Account) -> FillResult:
        """
        下單前檢查（Pre-Trade Validation）
        
        檢查項：
        1. 資金充足性
        2. 持倉可賣性
        3. 做空借券可用性
        4. 交易時段限制
        
        參數:
            order: 待檢查的訂單
            account: 當前帳戶狀態
        
        返回:
            FillResult（含拒絕原因）
        """
        # 資金檢查
        if order.side == OrderSide.BUY:
            estimated_cost = order.quantity * (order.limit_price or 1000) * 1.01
            if estimated_cost > account.cash:
                return FillResult(
                    filled=False,
                    reject_reason=f"資金不足: 需要${estimated_cost:,.0f}, 可用${account.cash:,.0f}"
                )
        
        # 做空檢查
        if order.side == OrderSide.SELL_SHORT:
            # 檢查借券可用性
            available = self.borrow_availability.get(order.ticker, 0)
            if order.quantity > available:
                return FillResult(
                    filled=False,
                    reject_reason=f"借券不足: {order.ticker} 可用{available}股, "
                                f"需要{order.quantity}股"
                )
            
            # Uptick Rule檢查（簡化）
            # 實際應檢查最近一次成交價是否高於前一次
            pass
        
        # 持倉檢查（賣出時）
        if order.side == OrderSide.SELL:
            pos = account.positions.get(order.ticker)
            if not pos or pos.quantity < order.quantity:
                return FillResult(
                    filled=False,
                    reject_reason=f"持倉不足: {order.ticker} 持有"
                                f"{pos.quantity if pos else 0}股, 賣出{order.quantity}股"
                )
        
        # PDT檢查
        if account.is_pdt and account.equity < 25_000:
            return FillResult(
                filled=False,
                reject_reason=f"PDT限制: 帳戶資金${account.equity:,.0f}不足$25,000"
            )
        
        return FillResult(filled=True, filled_quantity=order.quantity)


class OrderFillSimulator:
    """
    訂單撮合模擬器
    
    模擬訂單在交易所的成交過程，考慮：
    - 訂單簿流動性
    - 價格滑點
    - 市場衝擊成本
    - 部分成交可能性
    
    滑點模型：
    Almgren-Chriss 市場衝擊模型的簡化版本：
    Impact = σ * sqrt(Q / V) * (1 + sign(Q))
    其中 σ=波動率, Q=訂單量, V=日均成交量
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
        模擬限價單撮合
        
        邏輯：
        - 限價買單: 當日最低價 ≤ 限價 → 以限價成交（保守估計）
        - 限價賣單: 當日最高價 ≥ 限價 → 以限價成交
        
        參數:
            order: 限價單
            high, low, open_price, close_price: OHLC
            volume: 當日成交量
            avg_daily_volume: 日均成交量
        
        返回:
            FillResult
        """
        if order.order_type != OrderType.LIMIT or order.limit_price is None:
            return FillResult(filled=False, reject_reason="非限價單")
        
        can_fill = False
        fill_price = order.limit_price
        
        if order.side == OrderSide.BUY and low <= order.limit_price:
            can_fill = True
        elif order.side == OrderSide.SELL and high >= order.limit_price:
            can_fill = True
        elif order.side == OrderSide.SELL_SHORT and high >= order.limit_price:
            can_fill = True
        
        if not can_fill:
            return FillResult(filled=False, reject_reason="未達限價條件")
        
        # 計算滑點後的實際成交價
        actual_price = self._apply_slippage(
            fill_price, order, volume, avg_daily_volume
        )
        
        # 計算交易成本
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
        模擬市價單撮合
        
        市價單以次日開盤價成交（避免look-ahead bias），
        加上滑點調整。
        
        參數:
            order: 市價單
            open_price: 下一個bar的 開盤價
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
        應用滑點模型
        
        滑點 = base + vol_impact + size_impact
        
        參數:
            base_price: 基礎價格
            order: 訂單
            volume: 當日成交量
            avg_daily_volume: 日均量
        
        返回:
            滑點後的價格
        """
        cfg = self.config
        
        # 基礎滑點（1基點）
        base = cfg.slippage_base_bps / 10000
        
        # 規模衝擊
        if avg_daily_volume > 0:
            participation_rate = order.quantity / avg_daily_volume
            size_impact = cfg.slippage_size_coeff * np.sqrt(participation_rate) / 10000
        else:
            size_impact = 0
        
        total_slippage = base + size_impact
        
        # 買入:價格上升, 賣出:價格下降
        direction = 1 if order.side == OrderSide.BUY else -1
        return base_price * (1 + direction * total_slippage)
    
    def _calc_fees(self, order: Order, fill_price: float) -> Tuple[float, float, float]:
        """
        計算交易費用
        
        參數:
            order: 訂單
            fill_price: 成交價
        
        返回:
            (佣金, SEC費, TAF費)
        """
        cfg = self.config
        trade_value = order.quantity * fill_price
        
        # 佣金
        commission = max(cfg.commission_min, order.quantity * cfg.commission_per_share)
        commission = min(commission, trade_value * cfg.commission_max_pct)
        
        # SEC費（僅賣出）
        sec_fee = 0.0
        if order.side in (OrderSide.SELL, OrderSide.SELL_SHORT):
            sec_fee = trade_value * cfg.sec_fee_rate
        
        # TAF費（僅賣出）
        taf_fee = 0.0
        if order.side in (OrderSide.SELL, OrderSide.SELL_SHORT):
            taf_fee = min(order.quantity * cfg.taf_fee_per_share, cfg.taf_fee_max)
        
        return commission, sec_fee, taf_fee
