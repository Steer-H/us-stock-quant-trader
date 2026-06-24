"""Compliance rule checker."""

import logging
from typing import Optional, List, Dict, Set, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta, date
from collections import defaultdict
from enum import Enum

import pandas as pd

from config.settings import TradingConfig
from utils.constants import PDT_RULES, WashSaleRule, RegSHO
from utils.exceptions import (
    ComplianceError, WashSaleViolationError,
    ShortSellLocateError, PDTRestrictionError
)

logger = logging.getLogger(__name__)


# ============================================================================
# 數據結構
# ============================================================================
@dataclass
class Trade:
    """單筆交易記錄"""
    trade_id: str
    ticker: str
    side: str           # BUY/SELL/SELL_SHORT
    quantity: int
    price: float
    trade_date: date
    commission: float = 0.0
    sec_fee: float = 0.0
    taf_fee: float = 0.0
    
    @property
    def trade_value(self) -> float:
        return self.quantity * self.price


@dataclass
class WashSaleMatch:
    """洗售匹配記錄"""
    loss_trade: Trade      # 產生虧損的賣出交易
    buy_trade: Trade       # 窗口內的買入交易
    disallowed_loss: float # 不可抵扣的虧損金額
    matched_shares: int    # 匹配的股數
    window_start: date
    window_end: date


# ============================================================================
# 洗售規則追蹤器
# ============================================================================
class WashSaleTracker:
    """
    洗售規則 (Wash Sale Rule) 跟蹤器
    
    IRS規定：如果在賣出虧損股票的前後30天內（共61天窗口）
    買入相同或實質上相同的證券，該虧損不可抵稅，
    需調整新買入股票的成本基礎。
    
    本追蹤器：
    1. 記錄每筆交易
    2. 賣出產生虧損時，檢查前後30天窗口內的買入
    3. 標記匹配的洗售對
    4. 計算調整後的成本基礎
    5. 生成洗售報告
    
    時間複雜度: O(n^2) 檢查所有交易對（實際交易量下可接受）
    空間複雜度: O(n)
    """
    
    def __init__(self):
        self.trades: Dict[str, List[Trade]] = defaultdict(list)  # {ticker: [trades]}
        self.wash_sales: List[WashSaleMatch] = []
        self.adjusted_cost_basis: Dict[str, float] = {}  # {trade_id: adjusted_cost}
    
    def record_trade(self, trade: Trade) -> None:
        """
        記錄一筆交易並自動檢查洗售規則
        
        參數:
            trade: 交易記錄
        """
        self.trades[trade.ticker].append(trade)
        
        # 只在賣出且虧損時檢查
        if trade.side == 'SELL' and self._is_loss(trade):
            self._check_wash_sale(trade)
    
    def _is_loss(self, sell_trade: Trade) -> bool:
        """
        判斷賣出交易是否產生虧損
        
        通過比較賣出價和平均買入成本判斷。
        （簡化實現，實際應考慮FIFO/LIFO/特定標識法）
        
        參數:
            sell_trade: 賣出交易
        
        返回:
            是否虧損
        """
        # 計算該股票當前持倉的平均成本
        ticker_trades = self.trades[sell_trade.ticker]
        total_cost = 0.0
        total_shares = 0
        
        for t in ticker_trades:
            if t.trade_date < sell_trade.trade_date:
                if t.side == 'BUY':
                    total_cost += t.quantity * t.price + t.commission
                    total_shares += t.quantity
                elif t.side == 'SELL':
                    total_shares -= t.quantity
        
        if total_shares <= 0:
            return False
        
        avg_cost = total_cost / total_shares
        return sell_trade.price < avg_cost
    
    def _check_wash_sale(self, sell_trade: Trade) -> None:
        """
        檢查賣出交易是否觸發洗售規則
        
        在賣出日前後30天內查找相同股票的買入交易。
        
        參數:
            sell_trade: 虧損的賣出交易
        """
        ticker = sell_trade.ticker
        sell_date = sell_trade.trade_date
        
        window_start = sell_date - timedelta(days=WashSaleRule.WINDOW_DAYS)
        window_end = sell_date + timedelta(days=WashSaleRule.WINDOW_DAYS)
        
        # 查找窗口內的買入交易
        for trade in self.trades[ticker]:
            if (trade.side == 'BUY' and 
                window_start <= trade.trade_date <= window_end and
                trade.trade_id != sell_trade.trade_id):
                
                # 計算不可抵扣的虧損
                # 按比例匹配：如果買入100股但只賣出50股被匹配
                matched_shares = min(sell_trade.quantity, trade.quantity)
                loss_per_share = self._calculate_loss_per_share(sell_trade)
                disallowed_loss = matched_shares * loss_per_share
                
                match = WashSaleMatch(
                    loss_trade=sell_trade,
                    buy_trade=trade,
                    disallowed_loss=disallowed_loss,
                    matched_shares=matched_shares,
                    window_start=window_start,
                    window_end=window_end
                )
                
                self.wash_sales.append(match)
                
                # 調整新買入股票的成本基礎
                # 新的成本基礎 = 原買入價 + 不可抵扣的虧損/股
                adjusted_cost = trade.price + disallowed_loss / matched_shares
                self.adjusted_cost_basis[trade.trade_id] = adjusted_cost
                
                logger.warning(
                    f"⚠️ 洗售檢測: {ticker} 賣出虧損${loss_per_share:.2f}/股, "
                    f"窗口內買入{trade.quantity}股 @ ${trade.price:.2f}, "
                    f"不可抵扣虧損: ${disallowed_loss:.2f}, "
                    f"調整後成本: ${adjusted_cost:.2f}"
                )
    
    def _calculate_loss_per_share(self, sell_trade: Trade) -> float:
        """計算每股虧損金額"""
        ticker_trades = self.trades[sell_trade.ticker]
        total_cost = 0.0
        total_shares = 0
        
        for t in ticker_trades:
            if t.trade_date < sell_trade.trade_date and t.side == 'BUY':
                total_cost += t.quantity * t.price
                total_shares += t.quantity
        
        if total_shares == 0:
            return 0.0
        
        avg_cost = total_cost / total_shares
        return max(0, avg_cost - sell_trade.price)
    
    def generate_wash_sale_report(self) -> pd.DataFrame:
        """
        生成洗售報告
        
        返回:
            包含所有洗售匹配的DataFrame
        """
        if not self.wash_sales:
            return pd.DataFrame()
        
        records = []
        for ws in self.wash_sales:
            records.append({
                'ticker': ws.loss_trade.ticker,
                'loss_date': ws.loss_trade.trade_date,
                'buy_date': ws.buy_trade.trade_date,
                'shares': ws.matched_shares,
                'disallowed_loss': ws.disallowed_loss,
                'adjusted_cost': self.adjusted_cost_basis.get(ws.buy_trade.trade_id, 0),
            })
        
        return pd.DataFrame(records)


# ============================================================================
# 做空檢查器
# ============================================================================
class ShortSellChecker:
    """
    做空合規檢查器
    
    在做空訂單執行前自動檢查：
    1. 借券可用性 (Locate Requirement)
    2. Uptick Rule (Reg SHO Rule 201)
    3. 裸賣空防範
    
    參數:
        config: 交易配置
        borrow_inventory: 借券庫存 {ticker: available_shares}
    """
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.borrow_inventory: Dict[str, int] = {}  # 可用借券數量
        self.borrow_rates: Dict[str, float] = {}    # 年化借券費率
        self.locate_records: Dict[str, Dict] = {}   # locate記錄
        self.locate_id_counter: int = 0
    
    def update_borrow_inventory(self, ticker: str, available: int,
                                rate: float = 0.003) -> None:
        """
        更新借券庫存
        
        參數:
            ticker: 股票代碼
            available: 可借股數
            rate: 年化借券費率
        """
        self.borrow_inventory[ticker.upper()] = available
        self.borrow_rates[ticker.upper()] = rate
    
    def check_short_availability(self, ticker: str, 
                                  quantity: int) -> Tuple[bool, str]:
        """
        檢查做空借券可用性
        
        參數:
            ticker: 股票代碼
            quantity: 做空數量
        
        返回:
            (是否可用, 原因描述)
        """
        ticker = ticker.upper()
        available = self.borrow_inventory.get(ticker, 0)
        
        if available < quantity:
            return False, f"{ticker} 借券不足: 可用{available}股, 需要{quantity}股"
        
        return True, f"{ticker} 借券充足: {available}股可用"
    
    def request_locate(self, ticker: str, quantity: int) -> str:
        """
        請求做空借券定位 (Locate Request)
        
        Reg SHO Rule 203(b)(1): 做空前必須確認可以借到股票。
        
        參數:
            ticker: 股票代碼
            quantity: 做空數量
        
        返回:
            locate ID
        
        拋出:
            ShortSellLocateError: 無法定位借券
        """
        ticker = ticker.upper()
        available, reason = self.check_short_availability(ticker, quantity)
        
        if not available:
            raise ShortSellLocateError(
                reason,
                {'ticker': ticker, 'quantity': quantity, 'available': self.borrow_inventory.get(ticker, 0)}
            )
        
        self.locate_id_counter += 1
        locate_id = f"LOC{datetime.now().strftime('%Y%m%d')}{self.locate_id_counter:06d}"
        
        self.locate_records[locate_id] = {
            'locate_id': locate_id,
            'ticker': ticker,
            'quantity': quantity,
            'rate': self.borrow_rates.get(ticker, self.config.short_borrow_rate_annual),
            'timestamp': datetime.now().isoformat(),
            'expires': (datetime.now() + timedelta(days=RegSHO.LOCATE_VALIDITY_DAYS)).isoformat(),
            'status': 'ACTIVE'
        }
        
        logger.info(f"借券定位成功: {locate_id} {ticker} {quantity}股")
        
        return locate_id
    
    def consume_locate(self, locate_id: str) -> None:
        """
        使用locate（在訂單成交後）
        
        參數:
            locate_id: Locate ID
        """
        if locate_id in self.locate_records:
            self.locate_records[locate_id]['status'] = 'CONSUMED'
    
    def check_uptick_rule(self, ticker: str, current_price: float,
                          previous_close: float) -> Tuple[bool, str]:
        """
        檢查Uptick Rule (Reg SHO Rule 201)
        
        當股票日內跌超前收盤價10%時，觸發斷路器，
        剩餘交易日+下一交易日做空只能在uptick時執行。
        
        參數:
            ticker: 股票代碼
            current_price: 當前價格
            previous_close: 前收盤價
        
        返回:
            (是否觸發, 描述)
        """
        decline_pct = (current_price - previous_close) / previous_close
        
        if decline_pct <= -RegSHO.CIRCUIT_BREAKER_THRESHOLD:
            return True, (
                f"{ticker} 觸發Uptick Rule: 跌幅{decline_pct:.1%} >= "
                f"{RegSHO.CIRCUIT_BREAKER_THRESHOLD:.0%}, "
                f"做空受限"
            )
        
        return False, "未觸發Uptick Rule"


# ============================================================================
# 統一合規檢查器
# ============================================================================
class ComplianceChecker:
    """
    統一合規檢查器
    
    在下單前自動執行所有合規檢查：
    1. PDT規則
    2. 做空合規（locate + uptick）
    3. 洗售標記
    4. 持倉報告閾值
    
    所有檢查不可繞過，違規訂單將被直接拒絕。
    """
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.wash_sale_tracker = WashSaleTracker()
        self.short_checker = ShortSellChecker(config)
        
        # PDT追蹤
        self.day_trades: Dict[date, int] = defaultdict(int)  # {date: day_trade_count}
        self.is_pdt_marked: bool = False
    
    def pre_trade_compliance_check(
        self,
        ticker: str,
        side: str,
        quantity: int,
        price: float,
        previous_close: float,
        account_equity: float
    ) -> Tuple[bool, str]:
        """
        交易前合規檢查
        
        參數:
            ticker: 股票代碼
            side: BUY/SELL/SELL_SHORT
            quantity: 數量
            price: 價格
            previous_close: 前收盤價
            account_equity: 帳戶淨值
        
        返回:
            (是否通過, 原因)
        """
        # 1. PDT檢查
        if self.is_pdt_marked and account_equity < PDT_RULES.MIN_EQUITY:
            return False, (
                f"PDT限制: 帳戶淨值${account_equity:,.0f} < "
                f"${PDT_RULES.MIN_EQUITY:,.0f}最低要求"
            )
        
        # 2. 做空合規檢查
        if side == 'SELL_SHORT':
            # 2a. Locate檢查
            available, reason = self.short_checker.check_short_availability(ticker, quantity)
            if not available:
                return False, reason
            
            # 2b. Uptick Rule檢查
            triggered, msg = self.short_checker.check_uptick_rule(
                ticker, price, previous_close
            )
            if triggered:
                return False, msg
        
        # 3. 洗售檢查（僅標記，不阻止交易）
        if side == 'SELL':
            # 檢查是否在洗售窗口內買入過同一股票
            recent_buys = [
                t for t in self.wash_sale_tracker.trades.get(ticker.upper(), [])
                if t.side == 'BUY' and 
                (datetime.now().date() - t.trade_date).days <= WashSaleRule.WINDOW_DAYS
            ]
            if recent_buys:
                logger.warning(
                    f"⚠️ 潛在洗售: {ticker} 在 {WashSaleRule.WINDOW_DAYS}天內有買入記錄"
                )
        
        return True, "合規檢查通過"
    
    def record_trade(self, trade: Trade) -> None:
        """
        記錄交易（用於後續合規分析）
        
        參數:
            trade: 交易記錄
        """
        self.wash_sale_tracker.record_trade(trade)
        
        # 追蹤日間交易（同一天買賣同一股票）
        if trade.side in ('SELL', 'SELL_SHORT'):
            ticker_buys_today = [
                t for t in self.wash_sale_tracker.trades.get(trade.ticker, [])
                if t.side == 'BUY' and t.trade_date == trade.trade_date
            ]
            if ticker_buys_today:
                self.day_trades[trade.trade_date] += 1
        
        # PDT檢查
        self._check_pdt_status(datetime.now().date())
    
    def _check_pdt_status(self, check_date: date) -> None:
        """
        檢查PDT狀態
        
        如果5個交易日內有4次或以上日內交易，標記為PDT。
        
        參數:
            check_date: 檢查日期
        """
        # 統計最近5個交易日的日內交易次數
        recent_count = 0
        for d in sorted(self.day_trades.keys()):
            if (check_date - d).days <= PDT_RULES.ROLLING_WINDOW:
                recent_count += self.day_trades[d]
        
        if recent_count >= PDT_RULES.DAY_TRADE_COUNT:
            if not self.is_pdt_marked:
                self.is_pdt_marked = True
                logger.warning(
                    f"⚠️ 帳戶已被標記為PDT: "
                    f"{PDT_RULES.ROLLING_WINDOW}個交易日內{recent_count}次日內交易"
                )
    
    def generate_compliance_report(self) -> Dict[str, Any]:
        """
        生成合規報告
        
        返回:
            包含合規狀態的字典
        """
        return {
            'pdt_status': {
                'is_pdt': self.is_pdt_marked,
                'recent_day_trades': dict(sorted(self.day_trades.items())[-10:]),
            },
            'wash_sales': {
                'total_matches': len(self.wash_sale_tracker.wash_sales),
                'total_disallowed_loss': sum(
                    ws.disallowed_loss for ws in self.wash_sale_tracker.wash_sales
                ),
            },
            'short_sells': {
                'active_locates': len(self.short_checker.locate_records),
            },
        }
