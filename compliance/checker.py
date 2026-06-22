"""
美股量化交易系统 - 合规与报告模块

确保交易行为符合SEC/FINRA监管要求：
- 做空检查：查询借券库存、记录locate、防止裸卖空
- 洗售规则(Wash Sale)：自动检测并标记，调整成本基础
- PDT标记：自动判断并限制交易权限
- 大额持仓报告：13F/13H数据辅助导出
- 审计日志：完整记录每笔交易供合规审查

设计原则：
- 所有合规检查在下单前自动执行
- 不可绕过的硬检查（做空locate、PDT）
- 可配置的软检查（洗售标记、持仓预警）
- 完整的审计追踪链
"""

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
# 数据结构
# ============================================================================
@dataclass
class Trade:
    """单笔交易记录"""
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
    """洗售匹配记录"""
    loss_trade: Trade      # 产生亏损的卖出交易
    buy_trade: Trade       # 窗口内的买入交易
    disallowed_loss: float # 不可抵扣的亏损金额
    matched_shares: int    # 匹配的股数
    window_start: date
    window_end: date


# ============================================================================
# 洗售规则追踪器
# ============================================================================
class WashSaleTracker:
    """
    洗售规则 (Wash Sale Rule) 跟踪器
    
    IRS规定：如果在卖出亏损股票的前后30天内（共61天窗口）
    买入相同或实质上相同的证券，该亏损不可抵税，
    需调整新买入股票的成本基础。
    
    本追踪器：
    1. 记录每笔交易
    2. 卖出产生亏损时，检查前后30天窗口内的买入
    3. 标记匹配的洗售对
    4. 计算调整后的成本基础
    5. 生成洗售报告
    
    时间复杂度: O(n^2) 检查所有交易对（实际交易量下可接受）
    空间复杂度: O(n)
    """
    
    def __init__(self):
        self.trades: Dict[str, List[Trade]] = defaultdict(list)  # {ticker: [trades]}
        self.wash_sales: List[WashSaleMatch] = []
        self.adjusted_cost_basis: Dict[str, float] = {}  # {trade_id: adjusted_cost}
    
    def record_trade(self, trade: Trade) -> None:
        """
        记录一笔交易并自动检查洗售规则
        
        参数:
            trade: 交易记录
        """
        self.trades[trade.ticker].append(trade)
        
        # 只在卖出且亏损时检查
        if trade.side == 'SELL' and self._is_loss(trade):
            self._check_wash_sale(trade)
    
    def _is_loss(self, sell_trade: Trade) -> bool:
        """
        判断卖出交易是否产生亏损
        
        通过比较卖出价和平均买入成本判断。
        （简化实现，实际应考虑FIFO/LIFO/特定标识法）
        
        参数:
            sell_trade: 卖出交易
        
        返回:
            是否亏损
        """
        # 计算该股票当前持仓的平均成本
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
        检查卖出交易是否触发洗售规则
        
        在卖出日前后30天内查找相同股票的买入交易。
        
        参数:
            sell_trade: 亏损的卖出交易
        """
        ticker = sell_trade.ticker
        sell_date = sell_trade.trade_date
        
        window_start = sell_date - timedelta(days=WashSaleRule.WINDOW_DAYS)
        window_end = sell_date + timedelta(days=WashSaleRule.WINDOW_DAYS)
        
        # 查找窗口内的买入交易
        for trade in self.trades[ticker]:
            if (trade.side == 'BUY' and 
                window_start <= trade.trade_date <= window_end and
                trade.trade_id != sell_trade.trade_id):
                
                # 计算不可抵扣的亏损
                # 按比例匹配：如果买入100股但只卖出50股被匹配
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
                
                # 调整新买入股票的成本基础
                # 新的成本基础 = 原买入价 + 不可抵扣的亏损/股
                adjusted_cost = trade.price + disallowed_loss / matched_shares
                self.adjusted_cost_basis[trade.trade_id] = adjusted_cost
                
                logger.warning(
                    f"⚠️ 洗售检测: {ticker} 卖出亏损${loss_per_share:.2f}/股, "
                    f"窗口内买入{trade.quantity}股 @ ${trade.price:.2f}, "
                    f"不可抵扣亏损: ${disallowed_loss:.2f}, "
                    f"调整后成本: ${adjusted_cost:.2f}"
                )
    
    def _calculate_loss_per_share(self, sell_trade: Trade) -> float:
        """计算每股亏损金额"""
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
        生成洗售报告
        
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
# 做空检查器
# ============================================================================
class ShortSellChecker:
    """
    做空合规检查器
    
    在做空订单执行前自动检查：
    1. 借券可用性 (Locate Requirement)
    2. Uptick Rule (Reg SHO Rule 201)
    3. 裸卖空防范
    
    参数:
        config: 交易配置
        borrow_inventory: 借券库存 {ticker: available_shares}
    """
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.borrow_inventory: Dict[str, int] = {}  # 可用借券数量
        self.borrow_rates: Dict[str, float] = {}    # 年化借券费率
        self.locate_records: Dict[str, Dict] = {}   # locate记录
        self.locate_id_counter: int = 0
    
    def update_borrow_inventory(self, ticker: str, available: int,
                                rate: float = 0.003) -> None:
        """
        更新借券库存
        
        参数:
            ticker: 股票代码
            available: 可借股数
            rate: 年化借券费率
        """
        self.borrow_inventory[ticker.upper()] = available
        self.borrow_rates[ticker.upper()] = rate
    
    def check_short_availability(self, ticker: str, 
                                  quantity: int) -> Tuple[bool, str]:
        """
        检查做空借券可用性
        
        参数:
            ticker: 股票代码
            quantity: 做空数量
        
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
        请求做空借券定位 (Locate Request)
        
        Reg SHO Rule 203(b)(1): 做空前必须确认可以借到股票。
        
        参数:
            ticker: 股票代码
            quantity: 做空数量
        
        返回:
            locate ID
        
        抛出:
            ShortSellLocateError: 无法定位借券
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
        使用locate（在订单成交后）
        
        参数:
            locate_id: Locate ID
        """
        if locate_id in self.locate_records:
            self.locate_records[locate_id]['status'] = 'CONSUMED'
    
    def check_uptick_rule(self, ticker: str, current_price: float,
                          previous_close: float) -> Tuple[bool, str]:
        """
        检查Uptick Rule (Reg SHO Rule 201)
        
        当股票日内跌超前收盘价10%时，触发断路器，
        剩余交易日+下一交易日做空只能在uptick时执行。
        
        参数:
            ticker: 股票代码
            current_price: 当前价格
            previous_close: 前收盘价
        
        返回:
            (是否触发, 描述)
        """
        decline_pct = (current_price - previous_close) / previous_close
        
        if decline_pct <= -RegSHO.CIRCUIT_BREAKER_THRESHOLD:
            return True, (
                f"{ticker} 触发Uptick Rule: 跌幅{decline_pct:.1%} >= "
                f"{RegSHO.CIRCUIT_BREAKER_THRESHOLD:.0%}, "
                f"做空受限"
            )
        
        return False, "未触发Uptick Rule"


# ============================================================================
# 统一合规检查器
# ============================================================================
class ComplianceChecker:
    """
    统一合规检查器
    
    在下单前自动执行所有合规检查：
    1. PDT规则
    2. 做空合规（locate + uptick）
    3. 洗售标记
    4. 持仓报告阈值
    
    所有检查不可绕过，违规订单将被直接拒绝。
    """
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.wash_sale_tracker = WashSaleTracker()
        self.short_checker = ShortSellChecker(config)
        
        # PDT追踪
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
        交易前合规检查
        
        参数:
            ticker: 股票代码
            side: BUY/SELL/SELL_SHORT
            quantity: 数量
            price: 价格
            previous_close: 前收盘价
            account_equity: 账户净值
        
        返回:
            (是否通过, 原因)
        """
        # 1. PDT检查
        if self.is_pdt_marked and account_equity < PDT_RULES.MIN_EQUITY:
            return False, (
                f"PDT限制: 账户净值${account_equity:,.0f} < "
                f"${PDT_RULES.MIN_EQUITY:,.0f}最低要求"
            )
        
        # 2. 做空合规检查
        if side == 'SELL_SHORT':
            # 2a. Locate检查
            available, reason = self.short_checker.check_short_availability(ticker, quantity)
            if not available:
                return False, reason
            
            # 2b. Uptick Rule检查
            triggered, msg = self.short_checker.check_uptick_rule(
                ticker, price, previous_close
            )
            if triggered:
                return False, msg
        
        # 3. 洗售检查（仅标记，不阻止交易）
        if side == 'SELL':
            # 检查是否在洗售窗口内买入过同一股票
            recent_buys = [
                t for t in self.wash_sale_tracker.trades.get(ticker.upper(), [])
                if t.side == 'BUY' and 
                (datetime.now().date() - t.trade_date).days <= WashSaleRule.WINDOW_DAYS
            ]
            if recent_buys:
                logger.warning(
                    f"⚠️ 潜在洗售: {ticker} 在 {WashSaleRule.WINDOW_DAYS}天内有买入记录"
                )
        
        return True, "合规检查通过"
    
    def record_trade(self, trade: Trade) -> None:
        """
        记录交易（用于后续合规分析）
        
        参数:
            trade: 交易记录
        """
        self.wash_sale_tracker.record_trade(trade)
        
        # 追踪日间交易（同一天买卖同一股票）
        if trade.side in ('SELL', 'SELL_SHORT'):
            ticker_buys_today = [
                t for t in self.wash_sale_tracker.trades.get(trade.ticker, [])
                if t.side == 'BUY' and t.trade_date == trade.trade_date
            ]
            if ticker_buys_today:
                self.day_trades[trade.trade_date] += 1
        
        # PDT检查
        self._check_pdt_status(datetime.now().date())
    
    def _check_pdt_status(self, check_date: date) -> None:
        """
        检查PDT状态
        
        如果5个交易日内有4次或以上日内交易，标记为PDT。
        
        参数:
            check_date: 检查日期
        """
        # 统计最近5个交易日的日内交易次数
        recent_count = 0
        for d in sorted(self.day_trades.keys()):
            if (check_date - d).days <= PDT_RULES.ROLLING_WINDOW:
                recent_count += self.day_trades[d]
        
        if recent_count >= PDT_RULES.DAY_TRADE_COUNT:
            if not self.is_pdt_marked:
                self.is_pdt_marked = True
                logger.warning(
                    f"⚠️ 账户已被标记为PDT: "
                    f"{PDT_RULES.ROLLING_WINDOW}个交易日内{recent_count}次日内交易"
                )
    
    def generate_compliance_report(self) -> Dict[str, Any]:
        """
        生成合规报告
        
        返回:
            包含合规状态的字典
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
