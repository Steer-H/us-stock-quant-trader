"""
在线模拟交易系统 - 实时模拟交易核心引擎

核心功能：
- 同步真实时间，每分钟刷新行情数据
- 根据ML模型预测生成交易信号
- 自动执行模拟交易（市价单）
- 非交易时段显示倒计时
- 持续追踪持仓和盈亏
- 记录所有交易信号和执行的审计日志

架构：
┌──────────────────────────────────┐
│         LiveSimulator             │
│  ┌────────────┐  ┌─────────────┐ │
│  │ MarketClock│  │PortfolioMgr │ │
│  └────────────┘  └─────────────┘ │
│  ┌────────────┐  ┌─────────────┐ │
│  │BenchmarkTrk│  │AccuracyTrkr │ │
│  └────────────┘  └─────────────┘ │
│  ┌────────────┐  ┌─────────────┐ │
│  │  Dashboard │  │ AlertMgr    │ │
│  └────────────┘  └─────────────┘ │
└──────────────────────────────────┘

使用示例:
    sim = LiveSimulator(tickers=['AAPL','GOOGL','MSFT'])
    sim.run()
"""

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

# 默认追踪的股票
DEFAULT_TICKERS = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'NVDA', 'META', 'TSLA', 'NFLX']


class LiveSimulator:
    """
    在线模拟交易核心引擎
    
    以分钟频率运行，与真实市场同步：
    1. 每60秒检查市场状态
    2. 交易时段拉取实时行情
    3. 调用ML模型生成预测信号
    4. 执行模拟交易
    5. 更新持仓和账户
    6. 刷新仪表盘展示
    
    非交易时段：
    - 显示距离开市的倒计时
    - 不执行任何交易
    - 可查看历史记录
    """
    
    def __init__(
        self,
        tickers: Optional[List[str]] = None,
        initial_capital: float = 100_000.0,
        model: Optional[object] = None,   # 训练好的模型
        scaler: Optional[object] = None,  # 特征标准化器
        trading_config: Optional[TradingConfig] = None,
        model_config: Optional[ModelConfig] = None,
        refresh_interval: int = 60  # 刷新间隔（秒）
    ):
        """
        参数:
            tickers: 追踪的股票代码列表
            initial_capital: 初始资金（默认10万美元）
            model: 已训练的Transformer模型
            scaler: 特征标准化器
            trading_config: 交易配置
            model_config: 模型配置
            refresh_interval: 数据刷新间隔（秒）
        """
        self.tickers = [t.upper() for t in (tickers or DEFAULT_TICKERS)]
        self.initial_capital = initial_capital
        self.refresh_interval = refresh_interval
        
        # 子模块
        self.clock = MarketClock()
        self.portfolio = PortfolioManager(initial_capital)
        self.benchmark = BenchmarkTracker(initial_capital)
        self.accuracy_tracker = AccuracyTracker()
        
        # ML模型
        self.model = model
        self.scaler = scaler
        self.model_config = model_config or ModelConfig()
        self.trading_config = trading_config or TradingConfig()
        
        # 运行时状态
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._current_prices: Dict[str, float] = {}
        self._previous_prices: Dict[str, float] = {}
        self._iteration_count: int = 0
        self._last_refresh_time: float = 0.0
        
        # 回调函数
        self.on_refresh_callbacks: List[Callable] = []
        
        # 每日交易计数器
        self._daily_trades: int = 0
        self._daily_trade_date: Optional[object] = None
        
        logger.info(
            f"LiveSimulator初始化: {len(self.tickers)}只股票, "
            f"初始资金${initial_capital:,.0f}, 刷新间隔{refresh_interval}s"
        )
    
    def add_refresh_callback(self, callback: Callable) -> None:
        """
        添加刷新回调（用于自定义仪表盘更新）
        
        参数:
            callback: 回调函数 callback(simulator)
        """
        self.on_refresh_callbacks.append(callback)
    
    def start(self) -> None:
        """在后台线程启动模拟器"""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(
            target=self._run_loop,
            daemon=True,
            name='live_simulator'
        )
        self._thread.start()
        logger.info("在线模拟器已启动")
    
    def stop(self) -> None:
        """停止模拟器"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=10)
        logger.info("在线模拟器已停止")
    
    def _run_loop(self) -> None:
        """主运行循环"""
        # 初始化：拉取纳指基准数据
        self._initialize_benchmark()
        
        while self._running:
            try:
                self._tick()
            except Exception as e:
                logger.error(f"模拟器异常: {e}", exc_info=True)
            
            # 等待下一次刷新
            time.sleep(self.refresh_interval)
    
    def _tick(self) -> None:
        """单次tick（每分钟执行一次）"""
        self._iteration_count += 1
        now = datetime.now()
        
        # 1. 获取市场状态
        status, desc = self.clock.get_status()
        
        # 2. 拉取实时行情
        prices = self._fetch_realtime_prices()
        
        if not prices:
            logger.debug("无法获取行情数据，跳过本轮")
            return
        
        # 记录前一价格
        self._previous_prices = dict(self._current_prices)
        self._current_prices = prices
        
        # 3. 更新持仓市值
        self.portfolio.update_prices(prices)
        
        # 4. 更新基准
        nasdaq_price = prices.get('^IXIC', 0)
        if nasdaq_price > 0:
            self.benchmark.update(
                nasdaq_price,
                self.portfolio.get_total_equity(),
                now.isoformat()
            )
        
        # 5. 交易时段：生成信号并执行
        if status == MarketStatus.REGULAR_HOURS:
            self._execute_trading_cycle(prices)
        
        # 6. 调用刷新回调（更新面板）
        for callback in self.on_refresh_callbacks:
            try:
                callback(self)
            except Exception as e:
                logger.error(f"刷新回调异常: {e}")
        
        self._last_refresh_time = time.time()
    
    def _initialize_benchmark(self) -> None:
        """初始化纳指基准"""
        try:
            success = self.benchmark.fetch_nasdaq_history(period='6mo')
            if success:
                logger.info("纳指基准数据已加载")
        except Exception as e:
            logger.warning(f"纳指基准初始化失败: {e}")
    
    def _fetch_realtime_prices(self) -> Dict[str, float]:
        """
        拉取实时行情
        
        从Yahoo Finance获取股票的实时报价。
        使用yfinance的Ticker.info或history(period='1d')。
        
        返回:
            {ticker: current_price}
        """
        try:
            import yfinance as yf
            
            prices = {}
            
            # 批量获取（所有ticker + 纳指）
            all_tickers = self.tickers + ['^IXIC']
            
            for ticker in all_tickers:
                try:
                    stock = yf.Ticker(ticker)
                    
                    # 优先使用 fast_info 获取最新价格
                    info = stock.fast_info
                    price = info.get('lastPrice') or info.get('regularMarketPrice')
                    
                    if price and price > 0:
                        prices[ticker] = float(price)
                    else:
                        # 降级到 history
                        hist = stock.history(period='1d')
                        if not hist.empty:
                            prices[ticker] = float(hist['Close'].iloc[-1])
                            
                except Exception as e:
                    logger.debug(f"获取 {ticker} 价格失败: {e}")
                    # 使用上次价格（如果有）
                    if ticker in self._current_prices:
                        prices[ticker] = self._current_prices[ticker]
            
            return prices
            
        except ImportError:
            logger.error("yfinance未安装，使用模拟数据")
            return self._get_mock_prices()
        except Exception as e:
            logger.error(f"行情拉取失败: {e}")
            return self._get_mock_prices()
    
    def _get_mock_prices(self) -> Dict[str, float]:
        """
        生成模拟价格（仅用于测试）
        
        当yfinance不可用时，返回上次已知价格不做修改。
        不伪造随机价格波动，确保数据可信度。
        """
        mock = {}
        for ticker in self.tickers + ['^IXIC']:
            prev = self._current_prices.get(ticker, 150.0)
            mock[ticker] = round(prev, 2)
        
        return mock
    
    def _execute_trading_cycle(self, prices: Dict[str, float]) -> None:
        """
        执行一个交易周期
        
        步骤：
        1. 检查每日交易限制
        2. 生成ML预测信号
        3. 风控检查
        4. 执行订单
        5. 记录预测
        """
        # 每日交易重置
        today = datetime.now().date()
        if self._daily_trade_date != today:
            self._daily_trades = 0
            self._daily_trade_date = today
        
        if self._daily_trades >= self.trading_config.max_daily_trades:
            return  # 达到每日上限
        
        # 对每只追踪股票生成信号
        for ticker in self.tickers:
            if ticker not in prices:
                continue
            
            price = prices[ticker]
            
            # 生成ML预测
            prediction = self._generate_prediction(ticker)
            
            if prediction is None:
                continue
            
            predicted_return = prediction.get('return', 0)
            predicted_direction = 1 if predicted_return > 0 else 0
            confidence = prediction.get('confidence', 0.5)
            
            # 记录预测
            pred_id = self.accuracy_tracker.record_prediction(
                ticker, predicted_return, predicted_direction, confidence
            )
            
            # 如果置信度太低，跳过交易
            if confidence < 0.6:
                continue
            
            # 生成交易信号
            if predicted_direction == 1 and ticker not in self.portfolio.positions:
                # 买入信号
                qty = self._calculate_position_size(ticker, price, predicted_return)
                if qty > 0:
                    self.portfolio.execute_buy(
                        ticker, qty, price,
                        commission=self.trading_config.commission_min,
                        reason=f'ML预测涨 {predicted_return:+.2%}'
                    )
                    self._daily_trades += 1
                    
            elif predicted_direction == 0 and ticker in self.portfolio.positions:
                pos = self.portfolio.positions[ticker]
                if pos.unrealized_pnl_pct < -0.05:
                    # 止损：持有股的预测方向转跌且跌幅超5%
                    self.portfolio.execute_sell(
                        ticker, pos.quantity, price,
                        commission=self.trading_config.commission_min,
                        reason=f'ML预测跌+止损 {pos.unrealized_pnl_pct:.1%}'
                    )
                    self._daily_trades += 1
                elif pos.unrealized_pnl_pct > 0.10:
                    # 止盈：盈利超10%且预测转跌
                    self.portfolio.execute_sell(
                        ticker, pos.quantity, price,
                        commission=self.trading_config.commission_min,
                        reason=f'止盈+ML预测跌 {pos.unrealized_pnl_pct:.1%}'
                    )
                    self._daily_trades += 1
    
    def _generate_prediction(self, ticker: str) -> Optional[Dict]:
        """
        使用ML模型生成预测
        
        如果有加载的模型，使用模型预测。
        否则使用简单的随机信号（用于演示/测试）。
        
        参数:
            ticker: 股票代码
        
        返回:
            {'return': 预测收益率, 'direction': 涨跌方向, 'confidence': 置信度}
        """
        if self.model is not None and self.scaler is not None:
            # TODO: 使用真实模型预测
            # 需要构建输入特征序列，使用模型推理
            try:
                # 模型推理的占位代码
                # features = self._build_features(ticker)
                # prediction = self.model.predict(features)
                pass
            except Exception as e:
                logger.debug(f"模型预测失败: {e}")
        
        # 降级方案：基于简单规则的信号
        price = self._current_prices.get(ticker, 0)
        prev_price = self._previous_prices.get(ticker, price)
        
        if prev_price > 0:
            momentum = (price - prev_price) / prev_price
        else:
            momentum = 0
        
        # 简单动量信号（不做随机噪声污染）
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
        计算仓位大小
        
        基于Kelly准则的简化版本：
        position_size = equity * max_position_pct * min(confidence, 0.5)
        
        参数:
            ticker: 股票代码
            price: 当前价格
            predicted_return: 预测收益率
        
        返回:
            股数
        """
        equity = self.portfolio.get_total_equity()
        max_position = equity * self.trading_config.max_position_pct
        
        # 预测收益率越高，仓位越大（但有上限）
        confidence_factor = min(abs(predicted_return) / 0.02, 1.0)  # 2%收益率=满仓
        target_value = max_position * confidence_factor
        
        qty = int(target_value / price)
        
        # 至少买1股，最多不超过现金的10%
        max_qty = int(self.portfolio.cash * 0.1 / price)
        qty = max(1, min(qty, max_qty))
        
        return qty
    
    def get_full_snapshot(self) -> Dict:
        """
        获取完整系统快照（供仪表盘使用）
        
        返回:
            {
                'market': 市场状态信息,
                'portfolio': 持仓快照,
                'benchmark': 基准对比快照,
                'accuracy': 准确率快照,
                'runtime': 运行时信息
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
        """检查引擎是否运行中"""
        return self._running
