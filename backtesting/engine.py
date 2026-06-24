"""Backtesting engine for strategy evaluation."""

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
# 枚舉定義
# ============================================================================
class OrderSide(Enum):
    """訂單方向"""
    BUY = 'BUY'        # 買入
    SELL = 'SELL'      # 賣出
    SELL_SHORT = 'SELL_SHORT'  # 做空


class OrderType(Enum):
    """訂單類型"""
    MARKET = 'MKT'   # 市價單
    LIMIT = 'LMT'    # 限價單
    STOP = 'STP'     # 止損單


class OrderStatus(Enum):
    """訂單狀態"""
    PENDING = 'PENDING'      # 待執行
    FILLED = 'FILLED'        # 已成交
    PARTIALLY_FILLED = 'PARTIAL'  # 部分成交
    REJECTED = 'REJECTED'    # 已拒絕
    CANCELLED = 'CANCELLED'  # 已取消


# ============================================================================
# 數據結構
# ============================================================================
@dataclass
class Order:
    """
    訂單數據結構
    
    包含完整的訂單信息，用於回測撮合和審計追蹤。
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
    reason: str = ''  # 訂單產生原因（策略信號描述）


@dataclass
class Position:
    """
    持倉數據結構
    
    支持做多和做空持倉，以及部分平倉。
    """
    ticker: str
    quantity: int          # 正數=多頭，負數=空頭
    avg_cost: float        # 平均成本價
    market_value: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    total_commission: float = 0.0
    short_borrow_fee: float = 0.0  # 做空借券費累計


@dataclass
class Account:
    """
    帳戶數據結構
    
    跟蹤帳戶資金、持倉、保證金等狀態。
    """
    initial_capital: float
    cash: float
    equity: float = 0.0
    positions: Dict[str, Position] = field(default_factory=dict)
    total_commission: float = 0.0
    total_slippage: float = 0.0
    total_short_fees: float = 0.0
    
    # PDT追蹤
    day_trades_5d: List[datetime] = field(default_factory=list)
    is_pdt: bool = False
    
    @property
    def total_pnl(self) -> float:
        """總盈虧（含已實現+未實現）"""
        realized = sum(p.realized_pnl for p in self.positions.values())
        unrealized = sum(p.unrealized_pnl for p in self.positions.values())
        return realized + unrealized
    
    @property
    def gross_exposure(self) -> float:
        """總敞口（多+空絕對值之和）"""
        long_val = sum(p.market_value for p in self.positions.values() if p.quantity > 0)
        short_val = abs(sum(p.market_value for p in self.positions.values() if p.quantity < 0))
        return long_val + short_val
    
    @property
    def net_exposure(self) -> float:
        """淨敞口（多-空）"""
        return sum(p.market_value * np.sign(p.quantity) for p in self.positions.values())
    
    @property
    def leverage(self) -> float:
        """當前槓桿倍數"""
        return safe_divide(self.gross_exposure, self.equity, 0.0)


@dataclass
class BacktestResult:
    """
    回測結果匯總
    
    包含完整的績效指標和時間序列。
    """
    # 基礎統計
    start_date: str = ''
    end_date: str = ''
    total_days: int = 0
    trading_days: int = 0
    
    # 收益指標
    initial_capital: float = 0.0
    final_equity: float = 0.0
    total_return: float = 0.0
    annual_return: float = 0.0
    
    # 風險指標
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0  # 最大回撤持續天數
    volatility_annual: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    
    # 交易統計
    total_trades: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    avg_holding_days: float = 0.0
    
    # 成本統計
    total_commission: float = 0.0
    total_slippage: float = 0.0
    total_short_fees: float = 0.0
    
    # 時間序列（用於繪圖和分析）
    equity_curve: Optional[pd.Series] = None
    daily_returns: Optional[pd.Series] = None
    drawdown_curve: Optional[pd.Series] = None
    trades_df: Optional[pd.DataFrame] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典（便於JSON序列化）"""
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
# 回測引擎
# ============================================================================
class BacktestEngine:
    """
    事件驅動回測引擎
    
    以逐日循環的方式模擬交易：
    1. 加載歷史行情數據
    2. 每個交易日按時間順序遍歷
    3. 調用策略生成交易信號
    4. 模擬訂單撮合（考慮成本和滑點）
    5. 更新持倉和帳戶狀態
    6. 記錄每日快照
    7. 生成績效報告
    
    使用示例:
        engine = BacktestEngine(trading_config)
        engine.load_data('AAPL', df_aapl)
        engine.set_strategy(my_strategy_function)
        result = engine.run()
    """
    
    def __init__(self, trading_config: TradingConfig = None):
        """
        參數:
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
        
        # 做空借券費率緩存 {ticker: annual_rate}
        self.borrow_rates: Dict[str, float] = {}
    
    def load_data(self, ticker: str, df: pd.DataFrame) -> None:
        """
        加載單只股票的歷史數據
        
        參數:
            ticker: 股票代碼
            df: 包含OHLCV的歷史數據DataFrame
        """
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing = set(required_cols) - set(df.columns)
        if missing:
            raise DataError(
                f"{ticker} 數據缺少必要列: {missing}",
                {'missing': list(missing)}
            )
        
        # 確保索引為DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        self.market_data[ticker] = df.sort_index()
        logger.info(f"已加載 {ticker}: {len(df)} 行, {df.index[0].date()} ~ {df.index[-1].date()}")
    
    def set_strategy(self, strategy_fn: Callable) -> None:
        """
        設置交易策略函數
        
        策略函數籤名:
            def strategy(engine, ticker, current_date, market_data) -> List[Order]:
                # 根據當前市場狀態生成訂單
                return orders
        
        參數:
            strategy_fn: 策略函數
        """
        self.strategy = strategy_fn
    
    def _create_order(
        self, ticker: str, side: OrderSide, quantity: int,
        order_type: OrderType = OrderType.MARKET,
        limit_price: Optional[float] = None,
        reason: str = ''
    ) -> Order:
        """
        創建訂單對象
        
        參數:
            ticker: 股票代碼
            side: 買賣方向
            quantity: 股數
            order_type: 訂單類型
            limit_price: 限價（限價單使用）
            reason: 訂單原因
        
        返回:
            Order對象
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
        模擬訂單成交
        
        考慮因素：
        - 市價單: 以下一個bar的開盤價成交（避免look-ahead）
        - 限價單: 當日最高/最低價判斷能否成交
        - 滑點: 根據訂單規模與日均成交量計算滑點
        - 交易成本: 佣金、SEC費、TAF費
        
        參數:
            order: 訂單對象
            current_price: 當前bar收盤價
            next_price: 下一個bar開盤價
            high: 下一個bar最高價
            low: 下一個bar最低價
            volume: 下一個bar成交量
        
        返回:
            成交價格（0表示未成交）
        """
        fill_price = 0.0
        
        if order.order_type == OrderType.MARKET:
            # 市價單以次日開盤價成交
            fill_price = next_price
        elif order.order_type == OrderType.LIMIT:
            # 限價買單: 價格 <= limit_price 時成交
            if order.side == OrderSide.BUY and low <= order.limit_price:
                fill_price = min(order.limit_price, next_price)
            # 限價賣單: 價格 >= limit_price 時成交
            elif order.side == OrderSide.SELL and high >= order.limit_price:
                fill_price = max(order.limit_price, next_price)
        elif order.order_type == OrderType.STOP:
            # 止損單: 觸及止損價後轉為市價單
            if order.side == OrderSide.SELL and low <= order.stop_price:
                fill_price = next_price  # 止損賣出以開盤價執行
            elif order.side == OrderSide.BUY and high >= order.stop_price:
                fill_price = next_price
        
        return fill_price
    
    def _calculate_slippage(
        self, fill_price: float, order: Order, volume: int, avg_volume: float
    ) -> float:
        """
        計算滑點成本
        
        滑點模型: 滑點 = base_bps + vol_coeff * 波動率 + size_coeff * sqrt(訂單量/日均量)
        
        參數:
            fill_price: 理論成交價
            order: 訂單
            volume: 當日成交量
            avg_volume: 20日均量
        
        返回:
            滑點後的實際成交價
        """
        config = self.trading_config
        
        # 基礎滑點（基點）
        base_slippage = config.slippage_base_bps / 10000.0
        
        # 訂單規模衝擊
        volume_ratio = safe_divide(order.quantity, avg_volume, 0.0)
        size_impact = config.slippage_size_coeff * np.sqrt(volume_ratio) / 10000.0
        
        # 波動率衝擊（簡化處理）
        vol_impact = config.slippage_vol_coeff * 0.01 / 10000.0  # 假設1%波動率
        
        total_slippage_pct = base_slippage + size_impact + vol_impact
        
        # 買入時滑點向上（支付更高價格），賣出時向下（收到更低價格）
        if order.side == OrderSide.BUY:
            actual_price = fill_price * (1 + total_slippage_pct)
        else:
            actual_price = fill_price * (1 - total_slippage_pct)
        
        return actual_price
    
    def _calculate_commission(self, order: Order, fill_price: float) -> Tuple[float, float, float]:
        """
        計算交易成本（佣金、SEC費、TAF費）
        
        IBKR階梯式佣金（簡化模型）：
        - 每股$0.005，最低$1.00
        
        SEC費用：
        - 賣出金額的0.00278%（2025年費率）
        
        TAF費用：
        - 賣出時每股$0.000166，上限$8.30
        
        參數:
            order: 訂單
            fill_price: 成交價
        
        返回:
            (佣金, SEC費, TAF費)
        """
        config = self.trading_config
        trade_value = order.quantity * fill_price
        
        # 佣金
        commission = max(config.commission_min, order.quantity * config.commission_per_share)
        commission = min(commission, trade_value * config.commission_max_pct)
        
        # SEC費（僅賣出時收取）
        sec_fee = 0.0
        if order.side in (OrderSide.SELL, OrderSide.SELL_SHORT):
            sec_fee = trade_value * config.sec_fee_rate
        
        # TAF費（僅賣出時收取）
        taf_fee = 0.0
        if order.side in (OrderSide.SELL, OrderSide.SELL_SHORT):
            taf_fee = min(order.quantity * config.taf_fee_per_share, config.taf_fee_max)
        
        return commission, sec_fee, taf_fee
    
    def _execute_order(self, order: Order, row: pd.Series, next_row: pd.Series) -> bool:
        """
        執行訂單撮合
        
        參數:
            order: 待執行的訂單
            row: 當前bar數據
            next_row: 下一個bar數據（用於撮合）
        
        返回:
            是否成功成交
        """
        # 獲取成交價格
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
        
        # 計算滑點
        avg_volume = row.get('volume_sma_20', row['volume'])
        actual_price = self._calculate_slippage(fill_price, order, row['volume'], avg_volume)
        
        # 計算交易成本
        commission, sec_fee, taf_fee = self._calculate_commission(order, actual_price)
        slippage_cost = abs(order.quantity * (actual_price - fill_price))
        
        # 更新訂單
        order.filled_quantity = order.quantity
        order.filled_price = actual_price
        order.commission = commission
        order.sec_fee = sec_fee
        order.taf_fee = taf_fee
        order.slippage_cost = slippage_cost
        order.status = OrderStatus.FILLED
        
        # 更新帳戶
        total_cost = order.quantity * actual_price + commission + sec_fee + taf_fee
        
        if order.side == OrderSide.BUY:
            self.account.cash -= total_cost
        elif order.side == OrderSide.SELL:
            self.account.cash += total_cost - commission - sec_fee - taf_fee
        elif order.side == OrderSide.SELL_SHORT:
            # 做空：收到現金但凍結保證金（簡化處理）
            self.account.cash += total_cost - commission - sec_fee - taf_fee
            # 做空借券費（年化，按日計算）
            borrow_rate = self.borrow_rates.get(order.ticker, self.trading_config.short_borrow_rate_annual)
            daily_borrow_fee = order.quantity * actual_price * borrow_rate / 252
            self.account.total_short_fees += daily_borrow_fee
        
        # 更新持倉
        if order.ticker not in self.account.positions:
            self.account.positions[order.ticker] = Position(
                ticker=order.ticker, quantity=0, avg_cost=0.0
            )
        
        pos = self.account.positions[order.ticker]
        
        if order.side == OrderSide.BUY:
            # 計算新的平均成本
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
            # 做空的實現盈虧
            if pos.quantity < 0:
                pos.realized_pnl += order.quantity * (pos.avg_cost - actual_price)
        
        pos.total_commission += commission
        
        # 更新帳戶統計
        self.account.total_commission += commission
        self.account.total_slippage += slippage_cost
        self.filled_orders.append(order)
        
        # 審計日誌
        logger.info(
            f"成交: {order.ticker} {order.side.value} {order.quantity}股 "
            f"@ ${actual_price:.2f}, 佣金${commission:.2f}"
        )
        
        return True
    
    def _update_account(self, prices: Dict[str, float]) -> None:
        """
        按當日收盤價更新帳戶市值
        
        參數:
            prices: {ticker: 收盤價}
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
        """檢查PDT規則"""
        if not self.current_date:
            return
        
        # 清理5天前的日內交易記錄
        cutoff = pd.Timestamp(self.current_date) - pd.Timedelta(days=5)
        self.account.day_trades_5d = [
            t for t in self.account.day_trades_5d 
            if t > cutoff
        ]
        
        if len(self.account.day_trades_5d) >= 4:
            self.account.is_pdt = True
            if self.account.equity < 25_000:
                logger.warning(
                    f"PDT警告: 帳戶淨值 ${self.account.equity:,.0f} "
                    f"低於$25,000最低要求"
                )
    
    def run(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> BacktestResult:
        """
        運行回測
        
        核心回測循環：
        1. 遍歷每個交易日的行情數據
        2. 調用策略函數生成訂單
        3. 執行訂單撮合
        4. 更新持倉和帳戶
        5. 記錄每日快照
        
        參數:
            start_date: 回測開始日期（None則使用數據最早日期）
            end_date: 回測結束日期（None則使用數據最晚日期）
        
        返回:
            BacktestResult回測結果
        """
        if not self.market_data:
            raise DataError("未加載行情數據")
        
        if not self.strategy:
            raise TradingError("未設置交易策略")
        
        # 1. 確定回測日期範圍
        all_dates = pd.DatetimeIndex([])
        for df in self.market_data.values():
            all_dates = all_dates.union(df.index)
        all_dates = all_dates.sort_values()
        
        if start_date:
            all_dates = all_dates[all_dates >= pd.Timestamp(start_date)]
        if end_date:
            all_dates = all_dates[all_dates <= pd.Timestamp(end_date)]
        
        if len(all_dates) < 2:
            raise DataError("回測日期範圍數據不足")
        
        # 2. 初始化帳戶
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
            f"回測開始: {all_dates[0].date()} ~ {all_dates[-1].date()}, "
            f"初始資金: {format_currency(self.account.initial_capital)}"
        )
        
        # 3. 主回測循環
        with Timer("回測運行"):
            for i in range(len(all_dates) - 1):
                current_dt = all_dates[i]
                next_dt = all_dates[i + 1]
                self.current_date = current_dt.date()
                
                # 獲取當前和下一個bar的數據
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
                
                # 3a. 更新持倉市值
                self._update_account(current_prices)
                
                # 3b. 檢查PDT規則
                self._check_pdt()
                
                # 3c. 調用策略生成訂單
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
                    logger.error(f"策略執行異常: {e}")
                    continue
                
                # 3d. 執行待處理訂單（在下一個bar開盤）
                for order in self.pending_orders[:]:
                    ticker = order.ticker
                    if ticker in current_rows and ticker in next_rows:
                        success = self._execute_order(
                            order, current_rows[ticker], next_rows[ticker]
                        )
                        if success:
                            self.pending_orders.remove(order)
                
                # 3e. 記錄每日快照
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
            
            # 最後一天更新市值
            if all_dates[-1] in next_prices:
                self._update_account({t: next_prices.get(t, 0) for t in self.market_data})
        
        # 4. 計算績效指標
        result = self._calculate_performance(str(all_dates[0].date()), str(all_dates[-1].date()))
        
        return result
    
    def _calculate_performance(self, start_date: str, end_date: str) -> BacktestResult:
        """
        計算回測績效指標
        
        參數:
            start_date: 回測開始日期
            end_date: 回測結束日期
        
        返回:
            BacktestResult
        """
        if not self.daily_snapshots:
            return BacktestResult()
        
        # 構建權益曲線
        equity_df = pd.DataFrame(self.daily_snapshots)
        equity_df.set_index('date', inplace=True)
        equity_curve = equity_df['equity']
        
        if len(equity_curve) < 2:
            return BacktestResult()
        
        # 每日收益率
        daily_returns = equity_curve.pct_change().dropna()
        
        # 基本統計
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
        
        # 最大回撤持續天數
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
        
        # 夏普比率（假設無風險利率為0）
        volatility = daily_returns.std() * np.sqrt(252)
        sharpe = safe_divide(annual_return, volatility, 0.0)
        
        # 索提諾比率（只考慮下行波動）
        downside_returns = daily_returns[daily_returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino = safe_divide(annual_return, downside_vol, 0.0)
        
        # 卡爾瑪比率
        calmar = safe_divide(annual_return, abs(max_drawdown), 0.0)
        
        # 交易統計
        all_trades = self.filled_orders
        total_commission = sum(o.commission for o in all_trades)
        total_slippage = sum(o.slippage_cost for o in all_trades)
        
        # 盈虧分析（簡化）
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
