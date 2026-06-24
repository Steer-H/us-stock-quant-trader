"""Live trading simulator with configurable parameters."""

import logging
import time
import threading
from typing import Optional, List, Dict, Callable
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np

from config.settings import TradingConfig, ModelConfig, system_config
from config.logging_config import setup_logging
from live_trading.market_clock import MarketClock, MarketStatus
from live_trading.portfolio import PortfolioManager, PortfolioSnapshot
from live_trading.benchmark import BenchmarkTracker, BenchmarkSnapshot
from live_trading.accuracy_tracker import AccuracyTracker, AccuracySnapshot

logger = logging.getLogger(__name__)

# 默認追蹤的股票
DEFAULT_TICKERS = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'NVDA', 'META', 'TSLA', 'NFLX']


class LiveSimulator:
    """
    在線模擬交易核心引擎
    
    以分鐘頻率運行，與真實市場同步：
    1. 每60秒檢查市場狀態
    2. 交易時段拉取實時行情
    3. 調用ML模型生成預測信號
    4. 執行模擬交易
    5. 更新持倉和帳戶
    6. 刷新儀錶盤展示
    
    非交易時段：
    - 顯示距離開市的倒計時
    - 不執行任何交易
    - 可查看歷史記錄
    """
    
    def __init__(
        self,
        tickers: Optional[List[str]] = None,
        initial_capital: float = 100_000.0,
        model: Optional[object] = None,   # 訓練好的模型
        scaler: Optional[object] = None,  # 特徵標準化器
        trading_config: Optional[TradingConfig] = None,
        model_config: Optional[ModelConfig] = None,
        refresh_interval: int = 60  # 刷新間隔（秒）
    ):
        """
        參數:
            tickers: 追蹤的股票代碼列表
            initial_capital: 初始資金（默認10萬美元）
            model: 已訓練的Transformer模型
            scaler: 特徵標準化器
            trading_config: 交易配置
            model_config: 模型配置
            refresh_interval: 數據刷新間隔（秒）
        """
        self.tickers = [t.upper() for t in (tickers or DEFAULT_TICKERS)]
        self.initial_capital = initial_capital
        self.refresh_interval = refresh_interval
        
        # 子模塊
        self.clock = MarketClock()
        self.portfolio = PortfolioManager(initial_capital)
        self.benchmark = BenchmarkTracker(initial_capital)
        self.accuracy_tracker = AccuracyTracker()
        
        # ML模型
        self.model = model
        self.scaler = scaler
        self.model_config = model_config or ModelConfig()
        self.trading_config = trading_config or TradingConfig()
        
        # 運行時狀態
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._current_prices: Dict[str, float] = {}
        self._previous_prices: Dict[str, float] = {}
        self._iteration_count: int = 0
        self._last_refresh_time: float = 0.0
        
        # 回調函數
        self.on_refresh_callbacks: List[Callable] = []
        
        # 每日交易計數器
        self._daily_trades: int = 0
        self._daily_trade_date: Optional[object] = None
        
        logger.info(
            f"LiveSimulator初始化: {len(self.tickers)}只股票, "
            f"初始資金${initial_capital:,.0f}, 刷新間隔{refresh_interval}s"
        )
    
    def add_refresh_callback(self, callback: Callable) -> None:
        """
        添加刷新回調（用於自定義儀錶盤更新）
        
        參數:
            callback: 回調函數 callback(simulator)
        """
        self.on_refresh_callbacks.append(callback)
    
    def start(self) -> None:
        """在後臺線程啟動模擬器"""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(
            target=self._run_loop,
            daemon=True,
            name='live_simulator'
        )
        self._thread.start()
        logger.info("在線模擬器已啟動")
    
    def stop(self) -> None:
        """停止模擬器"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=10)
        logger.info("在線模擬器已停止")
    
    def _run_loop(self) -> None:
        """主運行循環"""
        # 初始化：拉取納指基準數據
        self._initialize_benchmark()
        
        while self._running:
            try:
                self._tick()
            except Exception as e:
                logger.error(f"模擬器異常: {e}", exc_info=True)
            
            # 等待下一次刷新
            time.sleep(self.refresh_interval)
    
    def _tick(self) -> None:
        """單次tick（每分鐘執行一次）"""
        self._iteration_count += 1
        now = datetime.now()
        
        # 1. 獲取市場狀態
        status, desc = self.clock.get_status()
        
        # 2. 拉取實時行情
        prices = self._fetch_realtime_prices()
        
        if not prices:
            logger.debug("無法獲取行情數據，跳過本輪")
            return
        
        # 記錄前一價格
        self._previous_prices = dict(self._current_prices)
        self._current_prices = prices
        
        # 3. 更新持倉市值
        self.portfolio.update_prices(prices)
        
        # 4. 更新基準
        nasdaq_price = prices.get('^IXIC', 0)
        if nasdaq_price > 0:
            self.benchmark.update(
                nasdaq_price,
                self.portfolio.get_total_equity(),
                now.isoformat()
            )
        
        # 5. 交易時段：生成信號並執行
        if status == MarketStatus.REGULAR_HOURS:
            self._execute_trading_cycle(prices)
        
        # 6. 調用刷新回調（更新面板）
        for callback in self.on_refresh_callbacks:
            try:
                callback(self)
            except Exception as e:
                logger.error(f"刷新回調異常: {e}")
        
        self._last_refresh_time = time.time()
    
    def _initialize_benchmark(self) -> None:
        """初始化納指基準"""
        try:
            success = self.benchmark.fetch_nasdaq_history(period='6mo')
            if success:
                logger.info("納指基準數據已加載")
        except Exception as e:
            logger.warning(f"納指基準初始化失敗: {e}")
    
    def _fetch_realtime_prices(self) -> Dict[str, float]:
        """
        拉取實時行情
        
        從Yahoo Finance獲取股票的實時報價。
        使用yfinance的Ticker.info或history(period='1d')。
        
        返回:
            {ticker: current_price}
        """
        try:
            import yfinance as yf
            
            prices = {}
            
            # 批量獲取（所有ticker + 納指）
            all_tickers = self.tickers + ['^IXIC']
            
            for ticker in all_tickers:
                try:
                    stock = yf.Ticker(ticker)
                    
                    # 優先使用 fast_info 獲取最新價格
                    info = stock.fast_info
                    price = info.get('lastPrice') or info.get('regularMarketPrice')
                    
                    if price and price > 0:
                        prices[ticker] = float(price)
                    else:
                        # 降級到 history
                        hist = stock.history(period='1d')
                        if not hist.empty:
                            prices[ticker] = float(hist['Close'].iloc[-1])
                            
                except Exception as e:
                    logger.debug(f"獲取 {ticker} 價格失敗: {e}")
                    # 使用上次價格（如果有）
                    if ticker in self._current_prices:
                        prices[ticker] = self._current_prices[ticker]
            
            return prices
            
        except ImportError:
            logger.error("yfinance未安裝，使用模擬數據")
            return self._get_mock_prices()
        except Exception as e:
            logger.error(f"行情拉取失敗: {e}")
            return self._get_mock_prices()
    
    def _get_mock_prices(self) -> Dict[str, float]:
        """
        生成模擬價格（僅用於測試）
        
        當yfinance不可用時，返回上次已知價格不做修改。
        不偽造隨機價格波動，確保數據可信度。
        """
        mock = {}
        for ticker in self.tickers + ['^IXIC']:
            prev = self._current_prices.get(ticker, 150.0)
            mock[ticker] = round(prev, 2)
        
        return mock
    
    def _execute_trading_cycle(self, prices: Dict[str, float]) -> None:
        """
        執行一個交易周期
        
        步驟：
        1. 檢查每日交易限制
        2. 生成ML預測信號
        3. 風控檢查
        4. 執行訂單
        5. 記錄預測
        """
        # 每日交易重置
        today = datetime.now().date()
        if self._daily_trade_date != today:
            self._daily_trades = 0
            self._daily_trade_date = today
        
        if self._daily_trades >= self.trading_config.max_daily_trades:
            return  # 達到每日上限
        
        # 對每隻追蹤股票生成信號
        for ticker in self.tickers:
            if ticker not in prices:
                continue
            
            price = prices[ticker]
            
            # 生成ML預測
            prediction = self._generate_prediction(ticker)
            
            if prediction is None:
                continue
            
            predicted_return = prediction.get('return', 0)
            predicted_direction = 1 if predicted_return > 0 else 0
            confidence = prediction.get('confidence', 0.5)
            
            # 記錄預測
            pred_id = self.accuracy_tracker.record_prediction(
                ticker, predicted_return, predicted_direction, confidence
            )
            
            # 如果置信度太低，跳過交易
            if confidence < 0.6:
                continue
            
            # 生成交易信號
            if predicted_direction == 1 and ticker not in self.portfolio.positions:
                # 買入信號
                qty = self._calculate_position_size(ticker, price, predicted_return)
                if qty > 0:
                    self.portfolio.execute_buy(
                        ticker, qty, price,
                        commission=self.trading_config.commission_min,
                        reason=f'ML預測漲 {predicted_return:+.2%}'
                    )
                    self._daily_trades += 1
                    
            elif predicted_direction == 0 and ticker in self.portfolio.positions:
                pos = self.portfolio.positions[ticker]
                if pos.unrealized_pnl_pct < -0.05:
                    # 止損：持有股的預測方向轉跌且跌幅超5%
                    self.portfolio.execute_sell(
                        ticker, pos.quantity, price,
                        commission=self.trading_config.commission_min,
                        reason=f'ML預測跌+止損 {pos.unrealized_pnl_pct:.1%}'
                    )
                    self._daily_trades += 1
                elif pos.unrealized_pnl_pct > 0.10:
                    # 止盈：盈利超10%且預測轉跌
                    self.portfolio.execute_sell(
                        ticker, pos.quantity, price,
                        commission=self.trading_config.commission_min,
                        reason=f'止盈+ML預測跌 {pos.unrealized_pnl_pct:.1%}'
                    )
                    self._daily_trades += 1
    
    def _generate_prediction(self, ticker: str) -> Optional[Dict]:
        """
        使用ML模型生成預測
        
        如果有加載的模型，使用模型預測。
        否則使用簡單的隨機信號（用於演示/測試）。
        
        參數:
            ticker: 股票代碼
        
        返回:
            {'return': 預測收益率, 'direction': 漲跌方向, 'confidence': 置信度}
        """
        if self.model is not None and self.scaler is not None:
            # TODO: 使用真實模型預測
            # 需要構建輸入特徵序列，使用模型推理
            try:
                # 模型推理的佔位代碼
                # features = self._build_features(ticker)
                # prediction = self.model.predict(features)
                pass
            except Exception as e:
                logger.debug(f"模型預測失敗: {e}")
        
        # 降級方案：基於簡單規則的信號
        price = self._current_prices.get(ticker, 0)
        prev_price = self._previous_prices.get(ticker, price)
        
        if prev_price > 0:
            momentum = (price - prev_price) / prev_price
        else:
            momentum = 0
        
        # 簡單動量信號（不做隨機噪聲汙染）
        signal = momentum
        predicted_return = signal
        confidence = min(0.9, max(0.5, abs(signal) / 0.02 * 0.3 + 0.5))
        
        return {
            'return': predicted_return,
            'direction': 1 if predicted_return > 0 else 0,
            'confidence': confidence
        }
    
    def _calculate_position_size(self, ticker: str, price: float,
                                  predicted_return: float) -> int:
        """
        計算倉位大小
        
        基於Kelly準則的簡化版本：
        position_size = equity * max_position_pct * min(confidence, 0.5)
        
        參數:
            ticker: 股票代碼
            price: 當前價格
            predicted_return: 預測收益率
        
        返回:
            股數
        """
        equity = self.portfolio.get_total_equity()
        max_position = equity * self.trading_config.max_position_pct
        
        # 預測收益率越高，倉位越大（但有上限）
        confidence_factor = min(abs(predicted_return) / 0.02, 1.0)  # 2%收益率=滿倉
        target_value = max_position * confidence_factor
        
        qty = int(target_value / price)
        
        # 至少買1股，最多不超過現金的10%
        max_qty = int(self.portfolio.cash * 0.1 / price)
        qty = max(1, min(qty, max_qty))
        
        return qty
    
    def get_full_snapshot(self) -> Dict:
        """
        獲取完整系統快照（供儀錶盤使用）
        
        返回:
            {
                'market': 市場狀態信息,
                'portfolio': 持倉快照,
                'benchmark': 基準對比快照,
                'accuracy': 準確率快照,
                'runtime': 運行時信息
            }
        """
        market_info = self.clock.get_trading_session_info()
        portfolio_snapshot = self.portfolio.get_snapshot(
            market_info.get('description', '')
        )
        benchmark_snapshot = self.benchmark.get_snapshot()
        accuracy_snapshot = self.accuracy_tracker.get_snapshot()
        
        return {
            'market': market_info,
            'portfolio': portfolio_snapshot,
            'benchmark': benchmark_snapshot,
            'accuracy': accuracy_snapshot,
            'runtime': {
                'iteration': self._iteration_count,
                'last_refresh': datetime.fromtimestamp(self._last_refresh_time).strftime('%H:%M:%S') if self._last_refresh_time else 'N/A',
                'tickers_tracked': len(self.tickers),
                'refresh_interval': self.refresh_interval,
            }
        }
    
    def is_running(self) -> bool:
        """檢查引擎是否運行中"""
        return self._running
