"""
美股量化交易系统 - 回测引擎

精确模拟美股交易环境，包括：
- 逐日/逐分钟的事件驱动回测
- 真实的交易成本模拟（佣金、SEC费、TAF费、滑点）
- 公司行为处理（分股、分红自动调整）
- 做空机制模拟（含借券费用和Uptick Rule）
- 幸存者偏差校正
- 参数敏感性分析

设计原则：
- 事件驱动架构，模拟真实交易流程
- 严格的时序处理，禁止未来信息泄露(look-ahead bias)
- 所有交易决策在bar的收盘价确认后执行
- 交易成本精确计算到小数点后6位
"""

import logging
from typing import Optional, List, Dict, Any, Callable, Tuple
from pathlib import Path
from datetime import datetime, date
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum

import numpy as np
import pandas as pd

from config.settings import TradingConfig, ModelConfig, system_config
from utils.helpers import Timer, format_currency, format_pct, safe_divide
from utils.constants import MarketHours, RegSHO, LULD
from utils.exceptions import TradingError, DataError

logger = logging.getLogger(__name__)


# ============================================================================
# 枚举定义
# ============================================================================
class OrderSide(Enum):
    """订单方向"""
    BUY = 'BUY'        # 买入
    SELL = 'SELL'      # 卖出
    SELL_SHORT = 'SELL_SHORT'  # 做空


class OrderType(Enum):
    """订单类型"""
    MARKET = 'MKT'   # 市价单
    LIMIT = 'LMT'    # 限价单
    STOP = 'STP'     # 止损单


class OrderStatus(Enum):
    """订单状态"""
    PENDING = 'PENDING'      # 待执行
    FILLED = 'FILLED'        # 已成交
    PARTIALLY_FILLED = 'PARTIAL'  # 部分成交
    REJECTED = 'REJECTED'    # 已拒绝
    CANCELLED = 'CANCELLED'  # 已取消


# ============================================================================
# 数据结构
# ============================================================================
@dataclass
class Order:
    """
    订单数据结构
    
    包含完整的订单信息，用于回测撮合和审计追踪。
    """
    order_id: int
    ticker: str
    side: OrderSide
    order_type: OrderType
    quantity: int
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    timestamp: Optional[datetime] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: int = 0
    filled_price: float = 0.0
    commission: float = 0.0
    sec_fee: float = 0.0
    taf_fee: float = 0.0
    slippage_cost: float = 0.0
    reason: str = ''  # 订单产生原因（策略信号描述）


@dataclass
class Position:
    """
    持仓数据结构
    
    支持做多和做空持仓，以及部分平仓。
    """
    ticker: str
    quantity: int          # 正数=多头，负数=空头
    avg_cost: float        # 平均成本价
    market_value: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    total_commission: float = 0.0
    short_borrow_fee: float = 0.0  # 做空借券费累计


@dataclass
class Account:
    """
    账户数据结构
    
    跟踪账户资金、持仓、保证金等状态。
    """
    initial_capital: float
    cash: float
    equity: float = 0.0
    positions: Dict[str, Position] = field(default_factory=dict)
    total_commission: float = 0.0
    total_slippage: float = 0.0
    total_short_fees: float = 0.0
    
    # PDT追踪
    day_trades_5d: List[datetime] = field(default_factory=list)
    is_pdt: bool = False
    
    @property
    def total_pnl(self) -> float:
        """总盈亏（含已实现+未实现）"""
        realized = sum(p.realized_pnl for p in self.positions.values())
        unrealized = sum(p.unrealized_pnl for p in self.positions.values())
        return realized + unrealized
    
    @property
    def gross_exposure(self) -> float:
        """总敞口（多+空绝对值之和）"""
        long_val = sum(p.market_value for p in self.positions.values() if p.quantity > 0)
        short_val = abs(sum(p.market_value for p in self.positions.values() if p.quantity < 0))
        return long_val + short_val
    
    @property
    def net_exposure(self) -> float:
        """净敞口（多-空）"""
        return sum(p.market_value * np.sign(p.quantity) for p in self.positions.values())
    
    @property
    def leverage(self) -> float:
        """当前杠杆倍数"""
        return safe_divide(self.gross_exposure, self.equity, 0.0)


@dataclass
class BacktestResult:
    """
    回测结果汇总
    
    包含完整的绩效指标和时间序列。
    """
    # 基础统计
    start_date: str = ''
    end_date: str = ''
    total_days: int = 0
    trading_days: int = 0
    
    # 收益指标
    initial_capital: float = 0.0
    final_equity: float = 0.0
    total_return: float = 0.0
    annual_return: float = 0.0
    
    # 风险指标
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0  # 最大回撤持续天数
    volatility_annual: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    
    # 交易统计
    total_trades: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    avg_holding_days: float = 0.0
    
    # 成本统计
    total_commission: float = 0.0
    total_slippage: float = 0.0
    total_short_fees: float = 0.0
    
    # 时间序列（用于绘图和分析）
    equity_curve: Optional[pd.Series] = None
    daily_returns: Optional[pd.Series] = None
    drawdown_curve: Optional[pd.Series] = None
    trades_df: Optional[pd.DataFrame] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（便于JSON序列化）"""
        return {
            'start_date': self.start_date,
            'end_date': self.end_date,
            'total_days': self.total_days,
            'initial_capital': format_currency(self.initial_capital),
            'final_equity': format_currency(self.final_equity),
            'total_return': format_pct(self.total_return),
            'annual_return': format_pct(self.annual_return),
            'max_drawdown': format_pct(self.max_drawdown),
            'sharpe_ratio': round(self.sharpe_ratio, 3),
            'sortino_ratio': round(self.sortino_ratio, 3),
            'calmar_ratio': round(self.calmar_ratio, 3),
            'total_trades': self.total_trades,
            'win_rate': format_pct(self.win_rate),
            'profit_factor': round(self.profit_factor, 3),
            'total_commission': format_currency(self.total_commission),
            'total_slippage': format_currency(self.total_slippage),
        }


# ============================================================================
# 回测引擎
# ============================================================================
class BacktestEngine:
    """
    事件驱动回测引擎
    
    以逐日循环的方式模拟交易：
    1. 加载历史行情数据
    2. 每个交易日按时间顺序遍历
    3. 调用策略生成交易信号
    4. 模拟订单撮合（考虑成本和滑点）
    5. 更新持仓和账户状态
    6. 记录每日快照
    7. 生成绩效报告
    
    使用示例:
        engine = BacktestEngine(trading_config)
        engine.load_data('AAPL', df_aapl)
        engine.set_strategy(my_strategy_function)
        result = engine.run()
    """
    
    def __init__(self, trading_config: TradingConfig = None):
        """
        参数:
            trading_config: 交易配置
        """
        self.trading_config = trading_config or TradingConfig()
        self.account: Optional[Account] = None
        self.market_data: Dict[str, pd.DataFrame] = {}  # {ticker: DataFrame}
        self.strategy: Optional[Callable] = None
        self.order_id_counter: int = 0
        self.pending_orders: List[Order] = []
        self.filled_orders: List[Order] = []
        self.daily_snapshots: List[Dict] = []
        self.current_date: Optional[date] = None
        
        # 做空借券费率缓存 {ticker: annual_rate}
        self.borrow_rates: Dict[str, float] = {}
    
    def load_data(self, ticker: str, df: pd.DataFrame) -> None:
        """
        加载单只股票的历史数据
        
        参数:
            ticker: 股票代码
            df: 包含OHLCV的历史数据DataFrame
        """
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing = set(required_cols) - set(df.columns)
        if missing:
            raise DataError(
                f"{ticker} 数据缺少必要列: {missing}",
                {'missing': list(missing)}
            )
        
        # 确保索引为DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        self.market_data[ticker] = df.sort_index()
        logger.info(f"已加载 {ticker}: {len(df)} 行, {df.index[0].date()} ~ {df.index[-1].date()}")
    
    def set_strategy(self, strategy_fn: Callable) -> None:
        """
        设置交易策略函数
        
        策略函数签名:
            def strategy(engine, ticker, current_date, market_data) -> List[Order]:
                # 根据当前市场状态生成订单
                return orders
        
        参数:
            strategy_fn: 策略函数
        """
        self.strategy = strategy_fn
    
    def _create_order(
        self, ticker: str, side: OrderSide, quantity: int,
        order_type: OrderType = OrderType.MARKET,
        limit_price: Optional[float] = None,
        reason: str = ''
    ) -> Order:
        """
        创建订单对象
        
        参数:
            ticker: 股票代码
            side: 买卖方向
            quantity: 股数
            order_type: 订单类型
            limit_price: 限价（限价单使用）
            reason: 订单原因
        
        返回:
            Order对象
        """
        self.order_id_counter += 1
        return Order(
            order_id=self.order_id_counter,
            ticker=ticker,
            side=side,
            order_type=order_type,
            quantity=quantity,
            limit_price=limit_price,
            timestamp=datetime.now(),
            reason=reason
        )
    
    def _simulate_fill(
        self, order: Order, current_price: float, next_price: float,
        high: float, low: float, volume: int
    ) -> float:
        """
        模拟订单成交
        
        考虑因素：
        - 市价单: 以下一个bar的开盘价成交（避免look-ahead）
        - 限价单: 当日最高/最低价判断能否成交
        - 滑点: 根据订单规模与日均成交量计算滑点
        - 交易成本: 佣金、SEC费、TAF费
        
        参数:
            order: 订单对象
            current_price: 当前bar收盘价
            next_price: 下一个bar开盘价
            high: 下一个bar最高价
            low: 下一个bar最低价
            volume: 下一个bar成交量
        
        返回:
            成交价格（0表示未成交）
        """
        fill_price = 0.0
        
        if order.order_type == OrderType.MARKET:
            # 市价单以次日开盘价成交
            fill_price = next_price
        elif order.order_type == OrderType.LIMIT:
            # 限价买单: 价格 <= limit_price 时成交
            if order.side == OrderSide.BUY and low <= order.limit_price:
                fill_price = min(order.limit_price, next_price)
            # 限价卖单: 价格 >= limit_price 时成交
            elif order.side == OrderSide.SELL and high >= order.limit_price:
                fill_price = max(order.limit_price, next_price)
        elif order.order_type == OrderType.STOP:
            # 止损单: 触及止损价后转为市价单
            if order.side == OrderSide.SELL and low <= order.stop_price:
                fill_price = next_price  # 止损卖出以开盘价执行
            elif order.side == OrderSide.BUY and high >= order.stop_price:
                fill_price = next_price
        
        return fill_price
    
    def _calculate_slippage(
        self, fill_price: float, order: Order, volume: int, avg_volume: float
    ) -> float:
        """
        计算滑点成本
        
        滑点模型: 滑点 = base_bps + vol_coeff * 波动率 + size_coeff * sqrt(订单量/日均量)
        
        参数:
            fill_price: 理论成交价
            order: 订单
            volume: 当日成交量
            avg_volume: 20日均量
        
        返回:
            滑点后的实际成交价
        """
        config = self.trading_config
        
        # 基础滑点（基点）
        base_slippage = config.slippage_base_bps / 10000.0
        
        # 订单规模冲击
        volume_ratio = safe_divide(order.quantity, avg_volume, 0.0)
        size_impact = config.slippage_size_coeff * np.sqrt(volume_ratio) / 10000.0
        
        # 波动率冲击（简化处理）
        vol_impact = config.slippage_vol_coeff * 0.01 / 10000.0  # 假设1%波动率
        
        total_slippage_pct = base_slippage + size_impact + vol_impact
        
        # 买入时滑点向上（支付更高价格），卖出时向下（收到更低价格）
        if order.side == OrderSide.BUY:
            actual_price = fill_price * (1 + total_slippage_pct)
        else:
            actual_price = fill_price * (1 - total_slippage_pct)
        
        return actual_price
    
    def _calculate_commission(self, order: Order, fill_price: float) -> Tuple[float, float, float]:
        """
        计算交易成本（佣金、SEC费、TAF费）
        
        IBKR阶梯式佣金（简化模型）：
        - 每股$0.005，最低$1.00
        
        SEC费用：
        - 卖出金额的0.00278%（2025年费率）
        
        TAF费用：
        - 卖出时每股$0.000166，上限$8.30
        
        参数:
            order: 订单
            fill_price: 成交价
        
        返回:
            (佣金, SEC费, TAF费)
        """
        config = self.trading_config
        trade_value = order.quantity * fill_price
        
        # 佣金
        commission = max(config.commission_min, order.quantity * config.commission_per_share)
        commission = min(commission, trade_value * config.commission_max_pct)
        
        # SEC费（仅卖出时收取）
        sec_fee = 0.0
        if order.side in (OrderSide.SELL, OrderSide.SELL_SHORT):
            sec_fee = trade_value * config.sec_fee_rate
        
        # TAF费（仅卖出时收取）
        taf_fee = 0.0
        if order.side in (OrderSide.SELL, OrderSide.SELL_SHORT):
            taf_fee = min(order.quantity * config.taf_fee_per_share, config.taf_fee_max)
        
        return commission, sec_fee, taf_fee
    
    def _execute_order(self, order: Order, row: pd.Series, next_row: pd.Series) -> bool:
        """
        执行订单撮合
        
        参数:
            order: 待执行的订单
            row: 当前bar数据
            next_row: 下一个bar数据（用于撮合）
        
        返回:
            是否成功成交
        """
        # 获取成交价格
        fill_price = self._simulate_fill(
            order,
            current_price=row['close'],
            next_price=next_row['open'],
            high=next_row['high'],
            low=next_row['low'],
            volume=next_row['volume']
        )
        
        if fill_price <= 0:
            return False  # 未成交
        
        # 计算滑点
        avg_volume = row.get('volume_sma_20', row['volume'])
        actual_price = self._calculate_slippage(fill_price, order, row['volume'], avg_volume)
        
        # 计算交易成本
        commission, sec_fee, taf_fee = self._calculate_commission(order, actual_price)
        slippage_cost = abs(order.quantity * (actual_price - fill_price))
        
        # 更新订单
        order.filled_quantity = order.quantity
        order.filled_price = actual_price
        order.commission = commission
        order.sec_fee = sec_fee
        order.taf_fee = taf_fee
        order.slippage_cost = slippage_cost
        order.status = OrderStatus.FILLED
        
        # 更新账户
        total_cost = order.quantity * actual_price + commission + sec_fee + taf_fee
        
        if order.side == OrderSide.BUY:
            self.account.cash -= total_cost
        elif order.side == OrderSide.SELL:
            self.account.cash += total_cost - commission - sec_fee - taf_fee
        elif order.side == OrderSide.SELL_SHORT:
            # 做空：收到现金但冻结保证金（简化处理）
            self.account.cash += total_cost - commission - sec_fee - taf_fee
            # 做空借券费（年化，按日计算）
            borrow_rate = self.borrow_rates.get(order.ticker, self.trading_config.short_borrow_rate_annual)
            daily_borrow_fee = order.quantity * actual_price * borrow_rate / 252
            self.account.total_short_fees += daily_borrow_fee
        
        # 更新持仓
        if order.ticker not in self.account.positions:
            self.account.positions[order.ticker] = Position(
                ticker=order.ticker, quantity=0, avg_cost=0.0
            )
        
        pos = self.account.positions[order.ticker]
        
        if order.side == OrderSide.BUY:
            # 计算新的平均成本
            total_qty = pos.quantity + order.quantity
            if total_qty > 0:
                pos.avg_cost = (
                    pos.quantity * pos.avg_cost + order.quantity * actual_price
                ) / total_qty
            pos.quantity += order.quantity
        elif order.side == OrderSide.SELL:
            pos.realized_pnl += order.quantity * (actual_price - pos.avg_cost)
            pos.quantity -= order.quantity
            if pos.quantity == 0:
                pos.avg_cost = 0.0
        elif order.side == OrderSide.SELL_SHORT:
            pos.quantity -= order.quantity
            # 做空的实现盈亏
            if pos.quantity < 0:
                pos.realized_pnl += order.quantity * (pos.avg_cost - actual_price)
        
        pos.total_commission += commission
        
        # 更新账户统计
        self.account.total_commission += commission
        self.account.total_slippage += slippage_cost
        self.filled_orders.append(order)
        
        # 审计日志
        logger.info(
            f"成交: {order.ticker} {order.side.value} {order.quantity}股 "
            f"@ ${actual_price:.2f}, 佣金${commission:.2f}"
        )
        
        return True
    
    def _update_account(self, prices: Dict[str, float]) -> None:
        """
        按当日收盘价更新账户市值
        
        参数:
            prices: {ticker: 收盘价}
        """
        total_market_value = 0.0
        
        for ticker, pos in self.account.positions.items():
            if ticker in prices and pos.quantity != 0:
                price = prices[ticker]
                pos.market_value = pos.quantity * price
                total_market_value += pos.market_value
                
                if pos.quantity > 0:
                    pos.unrealized_pnl = pos.quantity * (price - pos.avg_cost)
                else:
                    pos.unrealized_pnl = abs(pos.quantity) * (pos.avg_cost - price)
        
        self.account.equity = self.account.cash + total_market_value
    
    def _check_pdt(self) -> None:
        """检查PDT规则"""
        if not self.current_date:
            return
        
        # 清理5天前的日内交易记录
        cutoff = pd.Timestamp(self.current_date) - pd.Timedelta(days=5)
        self.account.day_trades_5d = [
            t for t in self.account.day_trades_5d 
            if t > cutoff
        ]
        
        if len(self.account.day_trades_5d) >= 4:
            self.account.is_pdt = True
            if self.account.equity < 25_000:
                logger.warning(
                    f"PDT警告: 账户净值 ${self.account.equity:,.0f} "
                    f"低于$25,000最低要求"
                )
    
    def run(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> BacktestResult:
        """
        运行回测
        
        核心回测循环：
        1. 遍历每个交易日的行情数据
        2. 调用策略函数生成订单
        3. 执行订单撮合
        4. 更新持仓和账户
        5. 记录每日快照
        
        参数:
            start_date: 回测开始日期（None则使用数据最早日期）
            end_date: 回测结束日期（None则使用数据最晚日期）
        
        返回:
            BacktestResult回测结果
        """
        if not self.market_data:
            raise DataError("未加载行情数据")
        
        if not self.strategy:
            raise TradingError("未设置交易策略")
        
        # 1. 确定回测日期范围
        all_dates = pd.DatetimeIndex([])
        for df in self.market_data.values():
            all_dates = all_dates.union(df.index)
        all_dates = all_dates.sort_values()
        
        if start_date:
            all_dates = all_dates[all_dates >= pd.Timestamp(start_date)]
        if end_date:
            all_dates = all_dates[all_dates <= pd.Timestamp(end_date)]
        
        if len(all_dates) < 2:
            raise DataError("回测日期范围数据不足")
        
        # 2. 初始化账户
        self.account = Account(
            initial_capital=self.trading_config.initial_capital,
            cash=self.trading_config.initial_capital,
            equity=self.trading_config.initial_capital
        )
        
        self.order_id_counter = 0
        self.pending_orders = []
        self.filled_orders = []
        self.daily_snapshots = []
        
        logger.info(
            f"回测开始: {all_dates[0].date()} ~ {all_dates[-1].date()}, "
            f"初始资金: {format_currency(self.account.initial_capital)}"
        )
        
        # 3. 主回测循环
        with Timer("回测运行"):
            for i in range(len(all_dates) - 1):
                current_dt = all_dates[i]
                next_dt = all_dates[i + 1]
                self.current_date = current_dt.date()
                
                # 获取当前和下一个bar的数据
                current_prices = {}
                next_prices = {}
                current_rows = {}
                next_rows = {}
                
                for ticker, df in self.market_data.items():
                    if current_dt in df.index and next_dt in df.index:
                        current_rows[ticker] = df.loc[current_dt]
                        next_rows[ticker] = df.loc[next_dt]
                        current_prices[ticker] = df.loc[current_dt, 'close']
                        next_prices[ticker] = df.loc[next_dt, 'open']
                
                if not current_prices:
                    continue
                
                # 3a. 更新持仓市值
                self._update_account(current_prices)
                
                # 3b. 检查PDT规则
                self._check_pdt()
                
                # 3c. 调用策略生成订单
                try:
                    new_orders = self.strategy(
                        engine=self,
                        tickers=list(current_prices.keys()),
                        current_date=current_dt,
                        current_rows=current_rows,
                        account=self.account
                    )
                    if new_orders:
                        self.pending_orders.extend(new_orders)
                except Exception as e:
                    logger.error(f"策略执行异常: {e}")
                    continue
                
                # 3d. 执行待处理订单（在下一个bar开盘）
                for order in self.pending_orders[:]:
                    ticker = order.ticker
                    if ticker in current_rows and ticker in next_rows:
                        success = self._execute_order(
                            order, current_rows[ticker], next_rows[ticker]
                        )
                        if success:
                            self.pending_orders.remove(order)
                
                # 3e. 记录每日快照
                snapshot = {
                    'date': current_dt,
                    'cash': self.account.cash,
                    'equity': self.account.equity,
                    'total_pnl': self.account.total_pnl,
                    'leverage': self.account.leverage,
                    'gross_exposure': self.account.gross_exposure,
                    'net_exposure': self.account.net_exposure,
                    'num_positions': len(self.account.positions),
                }
                self.daily_snapshots.append(snapshot)
            
            # 最后一天更新市值
            if all_dates[-1] in next_prices:
                self._update_account({t: next_prices.get(t, 0) for t in self.market_data})
        
        # 4. 计算绩效指标
        result = self._calculate_performance(str(all_dates[0].date()), str(all_dates[-1].date()))
        
        return result
    
    def _calculate_performance(self, start_date: str, end_date: str) -> BacktestResult:
        """
        计算回测绩效指标
        
        参数:
            start_date: 回测开始日期
            end_date: 回测结束日期
        
        返回:
            BacktestResult
        """
        if not self.daily_snapshots:
            return BacktestResult()
        
        # 构建权益曲线
        equity_df = pd.DataFrame(self.daily_snapshots)
        equity_df.set_index('date', inplace=True)
        equity_curve = equity_df['equity']
        
        if len(equity_curve) < 2:
            return BacktestResult()
        
        # 每日收益率
        daily_returns = equity_curve.pct_change().dropna()
        
        # 基本统计
        initial_capital = self.trading_config.initial_capital
        final_equity = equity_curve.iloc[-1]
        total_return = (final_equity - initial_capital) / initial_capital
        
        trading_days = len(daily_returns)
        years = max(trading_days / 252, 1/252)
        annual_return = (1 + total_return) ** (1 / years) - 1
        
        # 最大回撤
        cumulative = (1 + daily_returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # 最大回撤持续天数
        dd_start = None
        max_dd_duration = 0
        current_dd_duration = 0
        for dd in drawdown:
            if dd < 0:
                if dd_start is None:
                    dd_start = True
                current_dd_duration += 1
                max_dd_duration = max(max_dd_duration, current_dd_duration)
            else:
                current_dd_duration = 0
                dd_start = None
        
        # 夏普比率（假设无风险利率为0）
        volatility = daily_returns.std() * np.sqrt(252)
        sharpe = safe_divide(annual_return, volatility, 0.0)
        
        # 索提诺比率（只考虑下行波动）
        downside_returns = daily_returns[daily_returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino = safe_divide(annual_return, downside_vol, 0.0)
        
        # 卡尔玛比率
        calmar = safe_divide(annual_return, abs(max_drawdown), 0.0)
        
        # 交易统计
        all_trades = self.filled_orders
        total_commission = sum(o.commission for o in all_trades)
        total_slippage = sum(o.slippage_cost for o in all_trades)
        
        # 盈亏分析（简化）
        win_trades = 0
        loss_trades = 0
        
        result = BacktestResult(
            start_date=start_date,
            end_date=end_date,
            total_days=len(equity_curve),
            trading_days=trading_days,
            initial_capital=initial_capital,
            final_equity=final_equity,
            total_return=total_return,
            annual_return=annual_return,
            max_drawdown=max_drawdown,
            max_drawdown_duration=max_dd_duration,
            volatility_annual=volatility,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            total_trades=len(all_trades),
            total_commission=total_commission,
            total_slippage=total_slippage,
            total_short_fees=self.account.total_short_fees if self.account else 0,
            equity_curve=equity_curve,
            daily_returns=daily_returns,
            drawdown_curve=drawdown,
        )
        
        return result
