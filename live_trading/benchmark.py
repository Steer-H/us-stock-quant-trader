"""
在线模拟交易系统 - 纳指基准对比模块

实时追踪纳斯达克综合指数(^IXIC)的表现，
与策略收益进行多维度对比，评估策略是否跑赢大盘。

对比维度：
1. 累计收益率曲线对比
2. 年化收益率对比
3. 夏普比率对比
4. 最大回撤对比
5. Alpha / Beta 分析
6. 月度胜率对比

数据源：Yahoo Finance (^IXIC 实时数据)
刷新频率：每分钟
"""

import logging
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime, date

import numpy as np
import pandas as pd

from utils.helpers import safe_divide, format_pct

logger = logging.getLogger(__name__)


# ============================================================================
# 基准快照
# ============================================================================
@dataclass
class BenchmarkSnapshot:
    """
    基准对比快照
    
    包含纳指和策略的实时对比数据。
    """
    timestamp: str = ''
    
    # 纳指数据
    nasdaq_ticker: str = '^IXIC'
    nasdaq_price: float = 0.0
    nasdaq_day_change: float = 0.0
    nasdaq_day_change_pct: float = 0.0
    nasdaq_ytd_change_pct: float = 0.0
    
    # 纳指基准资金（假设同等资金投入纳指）
    nasdaq_equity: float = 100_000.0
    nasdaq_return_pct: float = 0.0
    nasdaq_annual_return: float = 0.0
    nasdaq_max_drawdown: float = 0.0
    nasdaq_sharpe: float = 0.0
    
    # 策略数据
    strategy_equity: float = 100_000.0
    strategy_return_pct: float = 0.0
    strategy_annual_return: float = 0.0
    strategy_max_drawdown: float = 0.0
    strategy_sharpe: float = 0.0
    
    # 对比指标
    excess_return: float = 0.0         # 超额收益 (策略 - 纳指)
    alpha: float = 0.0                 # Alpha
    beta: float = 0.0                  # Beta
    information_ratio: float = 0.0     # 信息比率
    tracking_error: float = 0.0        # 跟踪误差
    outperformance_pct: float = 0.0    # 跑赢百分比
    
    # 期间
    days_elapsed: int = 0


class BenchmarkTracker:
    """
    基准对比追踪器
    
    追踪纳斯达克指数表现，与策略收益进行全面对比。
    
    初始化时会拉取纳指历史数据，用于计算各项指标。
    随后每分钟更新最新价格。
    
    使用示例:
        tracker = BenchmarkTracker(initial_capital=100000.0)
        tracker.fetch_nasdaq_history()
        tracker.update(nasdaq_price=18500.0, strategy_equity=105000.0)
        snapshot = tracker.get_snapshot()
    """
    
    NASDAQ_TICKER = '^IXIC'  # 纳斯达克综合指数代码
    
    def __init__(self, initial_capital: float = 100_000.0):
        """
        参数:
            initial_capital: 初始资金
        """
        self.initial_capital = initial_capital
        
        # 纳指历史价格序列
        self.nasdaq_prices: pd.Series = pd.Series(dtype=float)
        self.nasdaq_returns: pd.Series = pd.Series(dtype=float)
        
        # 纳指基准权益曲线（假设初始资金全部买入纳指）
        self.nasdaq_equity_curve: pd.Series = pd.Series(dtype=float)
        self._nasdaq_equity_dict: dict = {}
        self.nasdaq_start_price: float = 0.0
        self.nasdaq_shares: float = 0.0
        
        # 策略权益曲线
        self.strategy_equity_curve: pd.Series = pd.Series(dtype=float)
        self._strategy_equity_dict: dict = {}
        
        # 起始日期
        self.start_date: Optional[date] = None
        
        # 最新数据
        self.current_nasdaq_price: float = 0.0
        self.nasdaq_prev_close: float = 0.0
        
        # 峰值（用于回撤计算）
        self.nasdaq_peak: float = 0.0
        self.strategy_peak: float = initial_capital
        self._nasdaq_worst_drawdown: float = 0.0
        self._strategy_worst_drawdown: float = 0.0
    
    def initialize_from_history(self, nasdaq_history: pd.DataFrame,
                                 start_date: Optional[date] = None) -> None:
        """
        从历史数据初始化纳指基准
        
        参数:
            nasdaq_history: 包含'close'列的纳指历史数据
            start_date: 策略起始日期
        """
        if nasdaq_history is None or nasdaq_history.empty:
            logger.warning("纳指历史数据为空，基准对比将不可用")
            return
        
        # 提取收盘价序列
        if 'close' in nasdaq_history.columns:
            self.nasdaq_prices = nasdaq_history['close'].copy()
        elif 'Close' in nasdaq_history.columns:
            self.nasdaq_prices = nasdaq_history['Close'].copy()
        
        # 移除 Yahoo Finance 返回的时区信息（避免 tz-aware vs tz-naive 比较报错）
        if hasattr(self.nasdaq_prices.index, 'tz') and self.nasdaq_prices.index.tz is not None:
            self.nasdaq_prices.index = self.nasdaq_prices.index.tz_localize(None)
        
        if start_date:
            self.start_date = start_date
            self.nasdaq_prices = self.nasdaq_prices[self.nasdaq_prices.index >= pd.Timestamp(start_date)]
        
        if self.nasdaq_prices.empty:
            return
        
        # 起始价格
        self.nasdaq_start_price = self.nasdaq_prices.iloc[0]
        
        # 计算买入份额
        self.nasdaq_shares = self.initial_capital / self.nasdaq_start_price
        
        # 构建基准权益曲线
        self.nasdaq_equity_curve = self.nasdaq_prices * self.nasdaq_shares
        self._nasdaq_equity_dict = dict(self.nasdaq_equity_curve.items())
        
        # 计算日收益率
        self.nasdaq_returns = self.nasdaq_prices.pct_change().dropna()
        
        # 峰值
        self.nasdaq_peak = self.nasdaq_prices.max()
        self.current_nasdaq_price = self.nasdaq_prices.iloc[-1]
        
        if len(self.nasdaq_prices) >= 2:
            self.nasdaq_prev_close = self.nasdaq_prices.iloc[-2]
        
        logger.info(
            f"纳指基准初始化: 起始${self.nasdaq_start_price:,.0f}, "
            f"当前${self.current_nasdaq_price:,.0f}, "
            f"份额{self.nasdaq_shares:.4f}"
        )
    
    def update(self, nasdaq_price: float, strategy_equity: float,
               timestamp: Optional[str] = None) -> None:
        """
        每分钟更新最新数据
        
        参数:
            nasdaq_price: 纳指最新价格
            strategy_equity: 策略最新净资产
            timestamp: 时间戳
        """
        self.current_nasdaq_price = nasdaq_price
        
        # 更新策略权益曲线
        if timestamp:
            dt = pd.Timestamp(timestamp).floor('s')
        else:
            dt = pd.Timestamp.now().floor('s')
        
        # 纳指回撤峰值更新
        if nasdaq_price > self.nasdaq_peak:
            self.nasdaq_peak = nasdaq_price
        
        # 策略峰值更新
        if strategy_equity > self.strategy_peak:
            self.strategy_peak = strategy_equity
        
        # 追加权益曲线到 dict (O(1) 操作)
        new_nasdaq_equity = nasdaq_price * self.nasdaq_shares if self.nasdaq_shares > 0 else 0
        self._nasdaq_equity_dict[dt] = new_nasdaq_equity
        self._strategy_equity_dict[dt] = strategy_equity
        
        # 每100次同步到 Series（性能优化）
        if len(self._nasdaq_equity_dict) % 100 == 0:
            self._ensure_curves_synced()
    
    def append_strategy_snapshot(self, timestamp: str, equity: float) -> None:
        """
        追加策略权益快照
        
        参数:
            timestamp: 时间戳
            equity: 策略净资产
        """
        dt = pd.Timestamp(timestamp)
        self._strategy_equity_dict[dt] = equity
        self.strategy_equity_curve = pd.Series(self._strategy_equity_dict)
        if equity > self.strategy_peak:
            self.strategy_peak = equity
    
    def _ensure_curves_synced(self):
        if self._nasdaq_equity_dict:
            self.nasdaq_equity_curve = pd.Series(self._nasdaq_equity_dict)
        if self._strategy_equity_dict:
            self.strategy_equity_curve = pd.Series(self._strategy_equity_dict)
    
    def get_snapshot(self) -> BenchmarkSnapshot:
        """
        获取当前基准对比快照
        
        返回:
            BenchmarkSnapshot对象
        """
        snap = BenchmarkSnapshot(
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            nasdaq_price=self.current_nasdaq_price,
        )
        
        # 纳指基准净值
        if self.nasdaq_shares > 0:
            snap.nasdaq_equity = self.current_nasdaq_price * self.nasdaq_shares
            snap.nasdaq_return_pct = safe_divide(
                snap.nasdaq_equity - self.initial_capital, self.initial_capital, 0.0
            )
        else:
            snap.nasdaq_equity = self.initial_capital
        
        # 策略净值（取权益曲线最新值）
        if not self.strategy_equity_curve.empty:
            snap.strategy_equity = self.strategy_equity_curve.iloc[-1]
            snap.strategy_return_pct = safe_divide(
                snap.strategy_equity - self.initial_capital, self.initial_capital, 0.0
            )
        
        # 纳指当日涨跌
        if self.nasdaq_prev_close > 0:
            snap.nasdaq_day_change = self.current_nasdaq_price - self.nasdaq_prev_close
            snap.nasdaq_day_change_pct = safe_divide(
                snap.nasdaq_day_change, self.nasdaq_prev_close, 0.0
            )
        
        # 年化收益率（假设已过天数）
        if self.start_date:
            days_elapsed = max((date.today() - self.start_date).days, 1)
            snap.days_elapsed = days_elapsed
            years = days_elapsed / 365.0
            
            snap.nasdaq_annual_return = (1 + snap.nasdaq_return_pct) ** (1 / years) - 1
            snap.strategy_annual_return = (1 + snap.strategy_return_pct) ** (1 / years) - 1
        
        # 回撤
        if self.nasdaq_peak > 0 and self.nasdaq_shares > 0:
            current_nasdaq_dd = (self.current_nasdaq_price - self.nasdaq_peak) / self.nasdaq_peak
            if current_nasdaq_dd < self._nasdaq_worst_drawdown:
                self._nasdaq_worst_drawdown = current_nasdaq_dd
            snap.nasdaq_max_drawdown = self._nasdaq_worst_drawdown
        
        if self.strategy_peak > 0:
            current_dd = (snap.strategy_equity - self.strategy_peak) / self.strategy_peak
            if current_dd < self._strategy_worst_drawdown:
                self._strategy_worst_drawdown = current_dd
            snap.strategy_max_drawdown = self._strategy_worst_drawdown
        
        # 夏普比率（使用历史数据）
        if len(self.nasdaq_returns) > 0:
            nasdaq_vol = self.nasdaq_returns.std() * np.sqrt(252)
            snap.nasdaq_sharpe = safe_divide(
                snap.nasdaq_annual_return, nasdaq_vol, 0.0
            )
        
        # 策略夏普（如果有策略收益率）
        if len(self.strategy_equity_curve) >= 2:
            strategy_returns = self.strategy_equity_curve.pct_change().dropna()
            if len(strategy_returns) > 0:
                strategy_vol = strategy_returns.std() * np.sqrt(252)
                snap.strategy_sharpe = safe_divide(
                    snap.strategy_annual_return, strategy_vol, 0.0
                )
        
        # 超额收益
        snap.excess_return = snap.strategy_return_pct - snap.nasdaq_return_pct
        snap.outperformance_pct = snap.excess_return  # 别名，与 excess_return 同值
        
        # Alpha/Beta（使用历史日收益率做回归）
        if (len(self.nasdaq_returns) > 20 and 
            len(self.strategy_equity_curve) >= 20):
            try:
                strategy_rets = self.strategy_equity_curve.pct_change().dropna()
                
                # 对齐日期
                aligned_nasdaq = self.nasdaq_returns.reindex(strategy_rets.index).dropna()
                aligned_strategy = strategy_rets.reindex(aligned_nasdaq.index)
                
                if len(aligned_nasdaq) > 20:
                    # Beta = Cov(strategy, market) / Var(market)
                    cov = np.cov(aligned_strategy, aligned_nasdaq)[0, 1]
                    var = np.var(aligned_nasdaq)
                    snap.beta = safe_divide(cov, var, 1.0)
                    
                    # Alpha = strategy_return - beta * market_return
                    snap.alpha = (aligned_strategy.mean() - snap.beta * aligned_nasdaq.mean()) * 252
                    
                    # 跟踪误差和信息比率
                    tracking_diff = aligned_strategy - aligned_nasdaq
                    snap.tracking_error = tracking_diff.std() * np.sqrt(252)
                    snap.information_ratio = safe_divide(
                        snap.alpha, snap.tracking_error, 0.0
                    )
            except Exception as e:
                logger.debug(f"Non-critical error in benchmark.py: {e}", exc_info=True)
        
        return snap
    
    def get_comparison_summary(self) -> Dict[str, str]:
        """
        获取对比摘要（格式化字符串）
        
        返回:
            {指标名: 格式化值}
        """
        snap = self.get_snapshot()
        
        def colorize(val: float) -> str:
            """颜色标记"""
            if val > 0:
                return f"\033[92m{val:+.2f}%\033[0m"  # 绿色
            elif val < 0:
                return f"\033[91m{val:+.2f}%\033[0m"  # 红色
            return f"{val:+.2f}%"
        
        return {
            '纳指累计收益': f"{snap.nasdaq_return_pct:+.2%}",
            '策略累计收益': f"{snap.strategy_return_pct:+.2%}",
            '超额收益': colorize(snap.outperformance_pct * 100),
            '纳指年化收益': f"{snap.nasdaq_annual_return:+.2%}",
            '策略年化收益': f"{snap.strategy_annual_return:+.2%}",
            '纳指夏普比率': f"{snap.nasdaq_sharpe:.3f}",
            '策略夏普比率': f"{snap.strategy_sharpe:.3f}",
            '纳指最大回撤': f"{snap.nasdaq_max_drawdown:.2%}",
            '策略最大回撤': f"{snap.strategy_max_drawdown:.2%}",
            'Beta': f"{snap.beta:.3f}",
            '信息比率': f"{snap.information_ratio:.3f}",
            '跟踪误差': f"{snap.tracking_error:.2%}",
        }
    
    def fetch_nasdaq_history(self, period: str = '6mo') -> bool:
        """
        从Yahoo Finance拉取纳指近期历史数据
        
        参数:
            period: 数据周期 (1mo, 3mo, 6mo, 1y, 2y, 5y)
        
        返回:
            是否成功
        """
        try:
            import yfinance as yf
            
            nasdaq = yf.Ticker(self.NASDAQ_TICKER)
            df = nasdaq.history(period=period)
            
            if df.empty:
                logger.warning("无法获取纳指历史数据")
                return False
            
            self.initialize_from_history(df, start_date=df.index[0].date())
            
            logger.info(f"纳指历史数据: {len(df)} 行, "
                       f"{df.index[0].date()} ~ {df.index[-1].date()}")
            
            return True
            
        except Exception as e:
            logger.error(f"获取纳指历史数据失败: {e}")
            return False
