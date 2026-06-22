"""
在线模拟交易系统 - 持仓与盈亏管理模块

实时跟踪：
- 每只持仓股票的当前价格、成本价格、持仓数量
- 每只持仓的浮动盈亏（金额 + 百分比）
- 总盈亏（已实现 + 未实现）
- 总净资产、现金余额、总市值
- 交易历史记录

数据刷新频率：每分钟（根据行情数据更新）
"""

import logging
from typing import Optional, List, Dict, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, date
from collections import defaultdict

import pandas as pd
import numpy as np

from utils.helpers import safe_divide, format_currency, format_pct

logger = logging.getLogger(__name__)


# ============================================================================
# 数据结构
# ============================================================================
@dataclass
class HoldingPosition:
    """
    单只持仓详情
    
    属性:
        ticker: 股票代码
        quantity: 持有股数（正=做多，负=做空）
        avg_cost: 平均成本价格
        current_price: 当前市场价格
        market_value: 当前市值
        cost_basis: 总成本
        unrealized_pnl: 未实现盈亏（金额）
        unrealized_pnl_pct: 未实现盈亏百分比
        day_change: 当日涨跌（金额）
        day_change_pct: 当日涨跌幅
        weight: 占总资产比例
    """
    ticker: str
    quantity: int = 0
    avg_cost: float = 0.0
    current_price: float = 0.0
    market_value: float = 0.0
    cost_basis: float = 0.0
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0
    day_change: float = 0.0
    day_change_pct: float = 0.0
    weight: float = 0.0
    last_update: str = ''


@dataclass
class PortfolioSnapshot:
    """
    账户完整快照
    
    记录某一时刻的完整账户状态，用于展示和历史对比。
    """
    timestamp: str = ''
    
    # 资金
    initial_capital: float = 100_000.0
    cash: float = 100_000.0
    total_market_value: float = 0.0
    total_equity: float = 100_000.0  # 净资产 = 现金 + 持仓市值
    
    # 盈亏
    realized_pnl: float = 0.0       # 已实现盈亏
    unrealized_pnl: float = 0.0     # 未实现盈亏
    total_pnl: float = 0.0          # 总盈亏
    total_pnl_pct: float = 0.0      # 总盈亏百分比
    
    # 当日
    day_pnl: float = 0.0            # 当日盈亏
    day_pnl_pct: float = 0.0        # 当日盈亏百分比
    
    # 持仓
    positions: Dict[str, HoldingPosition] = field(default_factory=dict)
    position_count: int = 0
    
    # 风控
    leverage: float = 0.0
    max_drawdown_pct: float = 0.0
    
    # 市场状态
    market_status: str = ''


@dataclass
class TradeRecord:
    """单笔交易记录"""
    trade_id: str
    ticker: str
    side: str             # BUY/SELL
    quantity: int
    price: float
    total_value: float    # 成交金额
    commission: float
    timestamp: str
    pnl: float = 0.0      # 该交易的盈亏（仅平仓时有值）
    reason: str = ''


# ============================================================================
# 持仓管理器
# ============================================================================
class PortfolioManager:
    """
    持仓与账户管理器
    
    职责：
    - 记录每笔交易
    - 维护持仓状态
    - 按最新行情更新持仓市值
    - 生成完整的账户快照
    - 计算各类盈亏指标
    
    初始资金：$100,000
    
    使用示例:
        pm = PortfolioManager(initial_capital=100000.0)
        pm.execute_buy('AAPL', 100, 150.0, commission=1.0)
        pm.update_prices({'AAPL': 152.0, 'GOOGL': 140.0})
        snapshot = pm.get_snapshot()
    """
    
    def __init__(self, initial_capital: float = 100_000.0):
        """
        参数:
            initial_capital: 初始资金（默认10万美元）
        """
        self.initial_capital = initial_capital
        self.cash = initial_capital
        
        # 持仓 {ticker: HoldingPosition}
        self.positions: Dict[str, HoldingPosition] = {}
        
        # 已实现盈亏
        self.realized_pnl: float = 0.0
        
        # 交易历史
        self.trade_history: List[TradeRecord] = []
        self._trade_id_counter: int = 0
        
        # 每日快照历史（用于回撤计算）
        self._equity_history: List[Tuple[str, float]] = []  # [(date_str, equity)]
        
        # 当日初始资产（用于计算当日盈亏）
        self._day_start_equity: float = initial_capital
        self._day_start_date: Optional[date] = None
        
        # 峰值资产（用于回撤计算）
        self._peak_equity: float = initial_capital
        
        # 总佣金
        self.total_commission: float = 0.0
        
        # Leverage/Margin Tracking
        self.borrowed: float = 0.0
        self.max_leverage: float = 2.0
        self.margin_used: float = 0.0
        self.total_interest: float = 0.0
        self._interest_rate_annual: float = 0.05
        self._last_interest_calc: str = ''
        
        logger.info(f"Account init: ${initial_capital:,.2f} (max {self.max_leverage}x leverage)")
    
    def get_available_cash(self) -> float:
        equity = self.get_total_equity()
        max_borrow = equity * (self.max_leverage - 1)
        return max(self.cash + max(max_borrow - self.borrowed, 0), 0)
    
    def get_leverage_ratio(self) -> float:
        equity = self.get_total_equity()
        return self.get_total_market_value() / equity if equity > 0 else 0.0
    
    def get_margin_ratio(self) -> float:
        equity = self.get_total_equity()
        max_borrow = equity * (self.max_leverage - 1)
        return self.borrowed / max_borrow if max_borrow > 0 else 0.0
    
    def borrow(self, amount: float) -> bool:
        equity = self.get_total_equity()
        max_borrow = equity * (self.max_leverage - 1)
        if self.borrowed + amount > max_borrow:
            return False
        self.cash += amount
        self.borrowed += amount
        return True
    
    def repay(self, amount: float) -> float:
        repay_amount = min(amount, self.borrowed)
        if repay_amount <= 0:
            return 0.0
        self.cash -= repay_amount
        self.borrowed -= repay_amount
        return repay_amount
    
    def accrue_interest(self) -> float:
        today = date.today().isoformat()
        if today == self._last_interest_calc or self.borrowed <= 0:
            return 0.0
        daily_rate = self._interest_rate_annual / 365
        interest = self.borrowed * daily_rate
        self.total_interest += interest
        self.cash -= interest
        self._last_interest_calc = today
        return interest
    
    def is_margin_call_risk(self) -> bool:
        return self.get_margin_ratio() > 0.90
    
    def execute_buy(self, ticker: str, quantity: int, price: float,
                    commission: float = 0.0, reason: str = '') -> Optional[TradeRecord]:
        """
        执行买入操作
        
        参数:
            ticker: 股票代码
            quantity: 买入数量
            price: 成交价格
            commission: 佣金
            reason: 交易原因
        
        返回:
            TradeRecord或None（资金不足时）
        """
        ticker = ticker.upper()
        total_cost = quantity * price + commission
        
        available = self.get_available_cash()
        if total_cost > available:
            logger.warning(f"Insufficient: need ${total_cost:,.2f}, avail ${available:,.2f}")
            return None
        
        if total_cost > self.cash:
            shortage = total_cost - self.cash
            if self.borrow(shortage):
                logger.info(f"Leverage borrow: ${shortage:,.2f} (total ${self.borrowed:,.0f})")
            else:
                return None
        
        # 更新现金
        self.cash -= total_cost
        self.total_commission += commission
        
        # 更新持仓
        if ticker not in self.positions:
            self.positions[ticker] = HoldingPosition(
                ticker=ticker,
                quantity=quantity,
                avg_cost=(quantity*price+commission)/quantity,
                current_price=price,
                market_value=quantity * price,
                cost_basis=total_cost,
                last_update=datetime.now().strftime('%H:%M:%S')
            )
        else:
            pos = self.positions[ticker]
            # 计算新的平均成本
            total_qty = pos.quantity + quantity
            new_total_cost = (pos.quantity * pos.avg_cost) + (quantity * price) + commission
            pos.avg_cost = new_total_cost / total_qty
            pos.quantity = total_qty
            pos.cost_basis += total_cost
        
        # 记录交易
        self._trade_id_counter += 1
        trade = TradeRecord(
            trade_id=f"T{self._trade_id_counter:06d}",
            ticker=ticker,
            side='BUY',
            quantity=quantity,
            price=price,
            total_value=total_cost,
            commission=commission,
            timestamp=datetime.now().isoformat(),
            reason=reason
        )
        self.trade_history.append(trade)
        
        # 更新持仓市值
        self._update_position(ticker, price)
        
        logger.info(
            f"买入: {ticker} {quantity}股 @ ${price:.2f}, 佣金${commission:.2f}, "
            f"余额${self.cash:,.2f}"
        )
        
        return trade
    
    def execute_sell(self, ticker: str, quantity: int, price: float,
                     commission: float = 0.0, reason: str = '') -> Optional[TradeRecord]:
        """
        执行卖出操作
        
        参数:
            ticker: 股票代码
            quantity: 卖出数量
            price: 成交价格
            commission: 佣金
            reason: 交易原因
        
        返回:
            TradeRecord或None（持仓不足时）
        """
        ticker = ticker.upper()
        
        # 持仓检查
        pos = self.positions.get(ticker)
        if not pos or pos.quantity < quantity:
            logger.warning(f"持仓不足: {ticker}, 持有{pos.quantity if pos else 0}股")
            return None
        
        # 计算盈亏
        # 使用FIFO方法（简化）：按平均成本计算
        cost_per_share = pos.avg_cost
        trade_pnl = (price - cost_per_share) * quantity
        
        # 更新现金和盈亏
        proceeds = quantity * price - commission
        self.cash += proceeds
        self.realized_pnl += trade_pnl - commission
        self.total_commission += commission
        
        if self.borrowed > 0 and self.cash > 0:
            repay_amount = min(self.cash * 0.3, self.borrowed)
            self.repay(repay_amount)
        
        # 更新持仓
        pos.quantity -= quantity
        if pos.quantity == 0:
            pos.avg_cost = 0.0
            pos.cost_basis = 0.0
        else:
            # 按比例减少成本基础
            pos.cost_basis = pos.cost_basis * (pos.quantity / (pos.quantity + quantity))
        
        # 记录交易
        self._trade_id_counter += 1
        trade = TradeRecord(
            trade_id=f"T{self._trade_id_counter:06d}",
            ticker=ticker,
            side='SELL',
            quantity=quantity,
            price=price,
            total_value=proceeds,
            commission=commission,
            timestamp=datetime.now().isoformat(),
            pnl=trade_pnl - commission,
            reason=reason
        )
        self.trade_history.append(trade)
        
        logger.info(
            f"卖出: {ticker} {quantity}股 @ ${price:.2f}, "
            f"盈亏${trade_pnl:+.2f}, 余额${self.cash:,.2f}"
        )
        
        return trade
    
    def update_prices(self, prices: Dict[str, float]) -> None:
        """
        按最新行情更新所有持仓市值
        
        每分钟调用一次，确保持仓数据显示最新价格。
        
        参数:
            prices: {ticker: current_price}
        """
        for ticker, price in prices.items():
            self._update_position(ticker.upper(), price)
        
        # 清理零持仓
        self.positions = {
            t: p for t, p in self.positions.items() if p.quantity != 0
        }
        
        # 每日快照记录
        today = date.today().isoformat()
        current_equity = self.get_total_equity()
        
        if self._day_start_date != date.today():
            self._day_start_date = date.today()
            self._day_start_equity = current_equity
        
        self._equity_history.append((today, current_equity))
        if len(self._equity_history) > 7200:
            self._equity_history = self._equity_history[-3600:]
        
        # 更新峰值
        if current_equity > self._peak_equity:
            self._peak_equity = current_equity
    
    def _update_position(self, ticker: str, price: float) -> None:
        """更新单只持仓的市值"""
        if ticker not in self.positions:
            return
        
        pos = self.positions[ticker]
        prev_price = pos.current_price
        pos.current_price = price
        
        if pos.quantity != 0:
            pos.market_value = pos.quantity * price
            pos.unrealized_pnl = pos.quantity * (price - pos.avg_cost)
            pos.unrealized_pnl_pct = safe_divide(
                pos.unrealized_pnl, abs(pos.cost_basis), 0.0
            )
            pos.day_change = pos.quantity * (price - prev_price) if prev_price > 0 else 0
            pos.day_change_pct = safe_divide(price - prev_price, prev_price, 0.0)
        
        pos.last_update = datetime.now().strftime('%H:%M:%S')
    
    def get_total_market_value(self) -> float:
        """获取持仓总市值"""
        return sum(p.market_value for p in self.positions.values())
    
    def get_total_equity(self) -> float:
        """Net equity = cash + market value - borrowed"""
        return self.cash + self.get_total_market_value() - self.borrowed
    
    def get_total_pnl(self) -> float:
        """获取总盈亏（已实现 + 未实现）"""
        unrealized = sum(p.unrealized_pnl for p in self.positions.values())
        return self.realized_pnl + unrealized
    
    def get_total_pnl_pct(self) -> float:
        """获取总盈亏百分比"""
        total_pnl = self.get_total_pnl()
        # 基于历史投入成本计算
        total_invested = self.initial_capital
        if self.trade_history:
            total_cost = sum(
                t.total_value for t in self.trade_history if t.side == 'BUY'
            )
            if total_cost > 0:
                total_invested = max(total_invested, total_cost)
        
        return safe_divide(total_pnl, total_invested, 0.0)
    
    def get_day_pnl(self) -> float:
        """获取当日盈亏"""
        return self.get_total_equity() - self._day_start_equity
    
    def get_day_pnl_pct(self) -> float:
        """获取当日盈亏百分比"""
        if self._day_start_equity <= 0:
            return 0.0
        return self.get_day_pnl() / self._day_start_equity
    
    def get_max_drawdown_pct(self) -> float:
        """获取最大回撤百分比"""
        current = self.get_total_equity()
        if self._peak_equity <= 0:
            return 0.0
        return (current - self._peak_equity) / self._peak_equity
    
    def get_leverage(self) -> float:
        """获取当前杠杆倍数"""
        equity = self.get_total_equity()
        if equity <= 0:
            return 0.0
        return self.get_total_market_value() / equity
    
    def get_snapshot(self, market_status: str = '') -> PortfolioSnapshot:
        """
        获取完整的账户快照
        
        包含所有持仓详情、盈亏数据、风控指标。
        
        参数:
            market_status: 市场状态字符串
        
        返回:
            PortfolioSnapshot对象
        """
        total_market_value = self.get_total_market_value()
        total_equity = self.get_total_equity()
        unrealized = sum(p.unrealized_pnl for p in self.positions.values())
        total_pnl = self.realized_pnl + unrealized
        
        snapshot = PortfolioSnapshot(
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            initial_capital=self.initial_capital,
            cash=self.cash,
            total_market_value=total_market_value,
            total_equity=total_equity,
            realized_pnl=self.realized_pnl,
            unrealized_pnl=unrealized,
            total_pnl=total_pnl,
            total_pnl_pct=self.get_total_pnl_pct(),
            day_pnl=self.get_day_pnl(),
            day_pnl_pct=self.get_day_pnl_pct(),
            positions=dict(self.positions),
            position_count=len(self.positions),
            leverage=self.get_leverage(),
            max_drawdown_pct=self.get_max_drawdown_pct(),
            market_status=market_status,
        )
        
        # 计算权重
        for pos in snapshot.positions.values():
            if total_equity > 0:
                pos.weight = pos.market_value / total_equity
        
        return snapshot
    
    def get_positions_summary(self) -> pd.DataFrame:
        """
        获取持仓摘要DataFrame
        
        返回列：
        - Ticker: 股票代码
        - Quantity: 持仓数量
        - Avg_Cost: 平均成本价
        - Current_Price: 当前价格
        - Market_Value: 当前市值
        - Cost_Basis: 总成本
        - Unrealized_PnL: 未实现盈亏
        - PnL_Pct: 盈亏百分比
        - Weight: 权重
        - Day_Change: 当日涨跌
        """
        if not self.positions:
            return pd.DataFrame(
                columns=['Ticker', 'Quantity', 'Avg_Cost', 'Current_Price',
                        'Market_Value', 'Cost_Basis', 'Unrealized_PnL',
                        'PnL_Pct', 'Weight', 'Day_Change_Pct']
            )
        
        records = []
        for ticker, pos in sorted(self.positions.items()):
            records.append({
                'Ticker': ticker,
                'Quantity': pos.quantity,
                'Avg_Cost': pos.avg_cost,
                'Current_Price': pos.current_price,
                'Market_Value': pos.market_value,
                'Cost_Basis': pos.cost_basis,
                'Unrealized_PnL': pos.unrealized_pnl,
                'PnL_Pct': pos.unrealized_pnl_pct,
                'Weight': pos.weight,
                'Day_Change_Pct': pos.day_change_pct,
            })
        
        return pd.DataFrame(records)
    
    def get_trade_summary(self, last_n: int = 20) -> pd.DataFrame:
        """获取最近N笔交易摘要"""
        trades = self.trade_history[-last_n:]
        
        if not trades:
            return pd.DataFrame()
        
        records = []
        for t in trades:
            records.append({
                'ID': t.trade_id,
                'Ticker': t.ticker,
                'Side': t.side,
                'Qty': t.quantity,
                'Price': t.price,
                'Value': t.total_value,
                'PnL': t.pnl if t.side == 'SELL' else 0,
                'Reason': t.reason,
                'Time': t.timestamp[:19],
            })
        
        return pd.DataFrame(records)[::-1]  # 最新在前
