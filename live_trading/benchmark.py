"""Benchmark tracker comparing portfolio against Nasdaq-100."""

import logging
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime, date

import numpy as np
import pandas as pd

from utils.helpers import safe_divide, format_pct

logger = logging.getLogger(__name__)


# ============================================================================
# 基準快照
# ============================================================================
@dataclass
class BenchmarkSnapshot:
    """
    基準對比快照
    
    包含納指和策略的實時對比數據。
    """
    timestamp: str = ''
    
    # 納指數據
    nasdaq_ticker: str = '^IXIC'
    nasdaq_price: float = 0.0
    nasdaq_day_change: float = 0.0
    nasdaq_day_change_pct: float = 0.0
    nasdaq_ytd_change_pct: float = 0.0
    
    # 納指基準資金（假設同等資金投入納指）
    nasdaq_equity: float = 100_000.0
    nasdaq_return_pct: float = 0.0
    nasdaq_annual_return: float = 0.0
    nasdaq_max_drawdown: float = 0.0
    nasdaq_sharpe: float = 0.0
    
    # 策略數據
    strategy_equity: float = 100_000.0
    strategy_return_pct: float = 0.0
    strategy_annual_return: float = 0.0
    strategy_max_drawdown: float = 0.0
    strategy_sharpe: float = 0.0
    
    # 對比指標
    excess_return: float = 0.0         # 超額收益 (策略 - 納指)
    alpha: float = 0.0                 # Alpha
    beta: float = 0.0                  # Beta
    information_ratio: float = 0.0     # 信息比率
    tracking_error: float = 0.0        # 跟蹤誤差
    outperformance_pct: float = 0.0    # 跑贏百分比
    
    # 期間
    days_elapsed: int = 0


class BenchmarkTracker:
    """
    基準對比追蹤器
    
    追蹤納斯達克指數表現，與策略收益進行全面對比。
    
    初始化時會拉取納指歷史數據，用於計算各項指標。
    隨後每分鐘更新最新價格。
    
    使用示例:
        tracker = BenchmarkTracker(initial_capital=100000.0)
        tracker.fetch_nasdaq_history()
        tracker.update(nasdaq_price=18500.0, strategy_equity=105000.0)
        snapshot = tracker.get_snapshot()
    """
    
    NASDAQ_TICKER = '^IXIC'  # 納斯達克綜合指數代碼
    
    def __init__(self, initial_capital: float = 100_000.0):
        """
        參數:
            initial_capital: 初始資金
        """
        self.initial_capital = initial_capital
        
        # 納指歷史價格序列
        self.nasdaq_prices: pd.Series = pd.Series(dtype=float)
        self.nasdaq_returns: pd.Series = pd.Series(dtype=float)
        
        # 納指基準權益曲線（假設初始資金全部買入納指）
        self.nasdaq_equity_curve: pd.Series = pd.Series(dtype=float)
        self._nasdaq_equity_dict: dict = {}
        self.nasdaq_start_price: float = 0.0
        self.nasdaq_shares: float = 0.0
        
        # 策略權益曲線
        self.strategy_equity_curve: pd.Series = pd.Series(dtype=float)
        self._strategy_equity_dict: dict = {}
        
        # 起始日期
        self.start_date: Optional[date] = None
        
        # 最新數據
        self.current_nasdaq_price: float = 0.0
        self.nasdaq_prev_close: float = 0.0
        
        # 峰值（用於回撤計算）
        self.nasdaq_peak: float = 0.0
        self.strategy_peak: float = initial_capital
        self._nasdaq_worst_drawdown: float = 0.0
        self._strategy_worst_drawdown: float = 0.0
    
    def initialize_from_history(self, nasdaq_history: pd.DataFrame,
                                 start_date: Optional[date] = None) -> None:
        """
        從歷史數據初始化納指基準
        
        參數:
            nasdaq_history: 包含'close'列的納指歷史數據
            start_date: 策略起始日期
        """
        if nasdaq_history is None or nasdaq_history.empty:
            logger.warning("納指歷史數據為空，基準對比將不可用")
            return
        
        # 提取收盤價序列
        if 'close' in nasdaq_history.columns:
            self.nasdaq_prices = nasdaq_history['close'].copy()
        elif 'Close' in nasdaq_history.columns:
            self.nasdaq_prices = nasdaq_history['Close'].copy()
        
        # 移除 Yahoo Finance 返回的時區信息（避免 tz-aware vs tz-naive 比較報錯）
        if hasattr(self.nasdaq_prices.index, 'tz') and self.nasdaq_prices.index.tz is not None:
            self.nasdaq_prices.index = self.nasdaq_prices.index.tz_localize(None)
        
        if start_date:
            self.start_date = start_date
            self.nasdaq_prices = self.nasdaq_prices[self.nasdaq_prices.index >= pd.Timestamp(start_date)]
        
        if self.nasdaq_prices.empty:
            return
        
        # 起始價格
        self.nasdaq_start_price = self.nasdaq_prices.iloc[0]
        
        # 計算買入份額
        self.nasdaq_shares = self.initial_capital / self.nasdaq_start_price
        
        # 構建基準權益曲線
        self.nasdaq_equity_curve = self.nasdaq_prices * self.nasdaq_shares
        self._nasdaq_equity_dict = dict(self.nasdaq_equity_curve.items())
        
        # 計算日收益率
        self.nasdaq_returns = self.nasdaq_prices.pct_change().dropna()
        
        # 峰值
        self.nasdaq_peak = self.nasdaq_prices.max()
        self.current_nasdaq_price = self.nasdaq_prices.iloc[-1]
        
        if len(self.nasdaq_prices) >= 2:
            self.nasdaq_prev_close = self.nasdaq_prices.iloc[-2]
        
        logger.info(
            f"納指基準初始化: 起始${self.nasdaq_start_price:,.0f}, "
            f"當前${self.current_nasdaq_price:,.0f}, "
            f"份額{self.nasdaq_shares:.4f}"
        )
    
    def update(self, nasdaq_price: float, strategy_equity: float,
               timestamp: Optional[str] = None) -> None:
        """
        每分鐘更新最新數據
        
        參數:
            nasdaq_price: 納指最新價格
            strategy_equity: 策略最新淨資產
            timestamp: 時間戳
        """
        self.current_nasdaq_price = nasdaq_price
        
        # 更新策略權益曲線
        if timestamp:
            dt = pd.Timestamp(timestamp).floor('s')
        else:
            dt = pd.Timestamp.now().floor('s')
        
        # 納指回撤峰值更新
        if nasdaq_price > self.nasdaq_peak:
            self.nasdaq_peak = nasdaq_price
        
        # 策略峰值更新
        if strategy_equity > self.strategy_peak:
            self.strategy_peak = strategy_equity
        
        # 追加權益曲線到 dict (O(1) 操作)
        new_nasdaq_equity = nasdaq_price * self.nasdaq_shares if self.nasdaq_shares > 0 else 0
        self._nasdaq_equity_dict[dt] = new_nasdaq_equity
        self._strategy_equity_dict[dt] = strategy_equity
        
        # 每100次同步到 Series（性能優化）
        if len(self._nasdaq_equity_dict) % 100 == 0:
            self._ensure_curves_synced()
    
    def append_strategy_snapshot(self, timestamp: str, equity: float) -> None:
        """
        追加策略權益快照
        
        參數:
            timestamp: 時間戳
            equity: 策略淨資產
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
        獲取當前基準對比快照
        
        返回:
            BenchmarkSnapshot對象
        """
        snap = BenchmarkSnapshot(
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            nasdaq_price=self.current_nasdaq_price,
        )
        
        # 納指基準淨值
        if self.nasdaq_shares > 0:
            snap.nasdaq_equity = self.current_nasdaq_price * self.nasdaq_shares
            snap.nasdaq_return_pct = safe_divide(
                snap.nasdaq_equity - self.initial_capital, self.initial_capital, 0.0
            )
        else:
            snap.nasdaq_equity = self.initial_capital
        
        # 策略淨值（取權益曲線最新值）
        if not self.strategy_equity_curve.empty:
            snap.strategy_equity = self.strategy_equity_curve.iloc[-1]
            snap.strategy_return_pct = safe_divide(
                snap.strategy_equity - self.initial_capital, self.initial_capital, 0.0
            )
        
        # 納指當日漲跌
        if self.nasdaq_prev_close > 0:
            snap.nasdaq_day_change = self.current_nasdaq_price - self.nasdaq_prev_close
            snap.nasdaq_day_change_pct = safe_divide(
                snap.nasdaq_day_change, self.nasdaq_prev_close, 0.0
            )
        
        # 年化收益率（假設已過天數）
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
        
        # 夏普比率（使用歷史數據）
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
        
        # 超額收益
        snap.excess_return = snap.strategy_return_pct - snap.nasdaq_return_pct
        snap.outperformance_pct = snap.excess_return  # 別名，與 excess_return 同值
        
        # Alpha/Beta（使用歷史日收益率做回歸）
        if (len(self.nasdaq_returns) > 20 and 
            len(self.strategy_equity_curve) >= 20):
            try:
                strategy_rets = self.strategy_equity_curve.pct_change().dropna()
                
                # 對齊日期
                aligned_nasdaq = self.nasdaq_returns.reindex(strategy_rets.index).dropna()
                aligned_strategy = strategy_rets.reindex(aligned_nasdaq.index)
                
                if len(aligned_nasdaq) > 20:
                    # Beta = Cov(strategy, market) / Var(market)
                    cov = np.cov(aligned_strategy, aligned_nasdaq)[0, 1]
                    var = np.var(aligned_nasdaq)
                    snap.beta = safe_divide(cov, var, 1.0)
                    
                    # Alpha = strategy_return - beta * market_return
                    snap.alpha = (aligned_strategy.mean() - snap.beta * aligned_nasdaq.mean()) * 252
                    
                    # 跟蹤誤差和信息比率
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
        獲取對比摘要（格式化字符串）
        
        返回:
            {指標名: 格式化值}
        """
        snap = self.get_snapshot()
        
        def colorize(val: float) -> str:
            """顏色標記"""
            if val > 0:
                return f"\033[92m{val:+.2f}%\033[0m"  # 綠色
            elif val < 0:
                return f"\033[91m{val:+.2f}%\033[0m"  # 紅色
            return f"{val:+.2f}%"
        
        return {
            '納指累計收益': f"{snap.nasdaq_return_pct:+.2%}",
            '策略累計收益': f"{snap.strategy_return_pct:+.2%}",
            '超額收益': colorize(snap.outperformance_pct * 100),
            '納指年化收益': f"{snap.nasdaq_annual_return:+.2%}",
            '策略年化收益': f"{snap.strategy_annual_return:+.2%}",
            '納指夏普比率': f"{snap.nasdaq_sharpe:.3f}",
            '策略夏普比率': f"{snap.strategy_sharpe:.3f}",
            '納指最大回撤': f"{snap.nasdaq_max_drawdown:.2%}",
            '策略最大回撤': f"{snap.strategy_max_drawdown:.2%}",
            'Beta': f"{snap.beta:.3f}",
            '信息比率': f"{snap.information_ratio:.3f}",
            '跟蹤誤差': f"{snap.tracking_error:.2%}",
        }
    
    def fetch_nasdaq_history(self, period: str = '6mo') -> bool:
        """
        從Yahoo Finance拉取納指近期歷史數據
        
        參數:
            period: 數據周期 (1mo, 3mo, 6mo, 1y, 2y, 5y)
        
        返回:
            是否成功
        """
        try:
            import yfinance as yf
            
            nasdaq = yf.Ticker(self.NASDAQ_TICKER)
            df = nasdaq.history(period=period)
            
            if df.empty:
                logger.warning("無法獲取納指歷史數據")
                return False
            
            self.initialize_from_history(df, start_date=df.index[0].date())
            
            logger.info(f"納指歷史數據: {len(df)} 行, "
                       f"{df.index[0].date()} ~ {df.index[-1].date()}")
            
            return True
            
        except Exception as e:
            logger.error(f"獲取納指歷史數據失敗: {e}")
            return False
