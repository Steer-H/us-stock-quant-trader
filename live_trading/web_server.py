"""
在线模拟交易系统 - Web服务器

Flask Web应用，提供：
- 实时交易仪表盘页面 (localhost:8080)
- RESTful API 数据接口（每秒刷新）
- 后台模拟交易引擎（每秒刷新）
- 开盘后才启动交易逻辑

启动方式:
    python live_trading/web_server.py
    或
    python main.py web
"""

import sys
import time
import threading
import hashlib
from pathlib import Path
from datetime import datetime
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# macOS Python 3.9 werkzeug selectors bug fix
# werkzeug 在 macOS + Python 3.9 上报 "changelist must be an iterable of select.kevent objects"
# 用 PollSelector 替换 DefaultSelector 以消除此错误
import selectors
if hasattr(selectors, 'PollSelector'):
    _Selector = selectors.PollSelector
    selectors.DefaultSelector = _Selector

from flask import Flask, render_template, jsonify, request

from live_trading.portfolio import PortfolioManager
from live_trading.benchmark import BenchmarkTracker
from live_trading.accuracy_tracker import AccuracyTracker
from live_trading.market_clock import MarketClock, MarketStatus
from live_trading.predictor import RealtimePredictor
from live_trading.model_inference import ModelInference
from risk.manager import RiskManager
from config.settings import trading_config
from live_trading.leverage_engine import LeverageEngine
import yfinance as yf
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeout
import logging
logger = logging.getLogger(__name__)
import signal
from live_trading.state_persistence import (
    save_state, load_state,
    deserialize_portfolio, deserialize_accuracy, deserialize_benchmark
)

app = Flask(__name__,
            template_folder='templates',
            static_folder='static')
app.config['TEMPLATES_AUTO_RELOAD'] = True
# socketio removed - using polling mode

# 全局引擎实例
_clock = MarketClock()
_portfolio = PortfolioManager(initial_capital=100_000.0)
_benchmark = BenchmarkTracker(initial_capital=100_000.0)
_accuracy = AccuracyTracker(rolling_window=50)
_predictor = RealtimePredictor(window_size=120)
# ML模型推理（加载Transformer）
_ml_inference: ModelInference = None  # 延迟加载，避免阻塞启动
_ml_ready: bool = False
_risk_mgr = RiskManager(trading_config)
_leverage_engine = LeverageEngine()  # 动态杠杆引擎
_risk_checks_passed: int = 0
_risk_checks_failed: int = 0

_current_prices: dict = {}
_previous_prices: dict = {}
_iteration_count: int = 0
_engine_running: bool = False
_engine_thread: threading.Thread = None

# 是否已开盘（开盘后才开始交易）
_market_opened: bool = False
# 是否已建仓
_positions_initialized: bool = False

# ═══════════════════════════════════════════════════════════════
# 数据延迟与精度监控
# ═══════════════════════════════════════════════════════════════
_last_price_update: str = ''       # 最后一次价格更新时间
_data_source: str = 'Yahoo Finance (启动中)'    # 数据源: simulated / yahoo / akshare
_data_latency_ms: float = 0.0      # 数据延迟(毫秒)
_price_update_count: int = 0       # 价格更新总次数
_price_is_stale: bool = True       # 价格数据是否过期
_price_data_age_s: float = 999.0  # 数据已存在秒数(上次更新距今)
# ═══════════════════════════════════════════════════════════
# K线缓存
# ═══════════════════════════════════════════════════════════
_kline_cache: dict = {}         # {ticker: [{time,open,high,low,close,volume}]}
_last_kline_fetch: float = 0.0  # 上次K线抓取时间
_kline_fetch_interval: int = 300  # 每5分钟更新一次K线

# ═══════════════════════════════════════════════════════════════
# 交易指令追踪
# ═══════════════════════════════════════════════════════════════
_recent_signals: list = []         # 最近交易指令 [{time, action, ticker, qty, price, reason}]
_prediction_iters: dict = {}       # {prediction_id: iteration} 用于超时确认

TRACKED_TICKERS = [
    # 科技七巨头
    'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'META', 'TSLA',
    # 软件/SaaS
    'NFLX', 'ADBE', 'CRM', 'NOW', 'ORCL',
    # 金融
    'JPM', 'V', 'MA', 'BAC',
    # 消费
    'WMT', 'HD', 'NKE', 'SBUX', 'UBER',
    # 芯片/半导体
    'AVGO', 'AMD', 'INTC', 'QCOM', 'TXN',
    # 光模块/光通信 (高波动)
    'AAOI', 'COHR', 'LITE', 'FN',
    # 存储 (高波动)
    'WDC', 'STX', 'NTAP',
    # 数据中心芯片 (高波动)
    'MRVL', 'MU',
    # 半导体设备 (高波动)
    'LRCX', 'AMAT', 'KLAC',
    # EDA软件 (高波动)
    'SNPS', 'CDNS',
]

# ═══════════════════════════════════════════════════════════════
# Yahoo Finance 实时数据抓取
# ═══════════════════════════════════════════════════════════════
_yahoo_fetch_interval: int = 30      # 基础抓取间隔（秒）

# 按市场时段分层抓取间隔（优化版）
YAHOO_INTERVALS = {
    'REGULAR': 12,       # 正常交易：12秒（快速刷新）
    'PRE_MARKET': 60,    # 盘前：1分钟
    'AFTER_HOURS': 60,   # 盘后：1分钟
    'CLOSED': 60,        # 闭市：1分钟
}
_last_yahoo_fetch: float = 0.0       # 上次抓取时间戳
_first_yahoo_fetch_done: bool = False  # 启动后首次抓取标志
_yahoo_fetch_success: bool = False   # 上次抓取是否成功
_yahoo_error_count: int = 0          # 连续失败次数
_yahoo_session = None               # 持久HTTP会话（v7 API用）
_yahoo_crumb: str = None            # Yahoo API crumb
_yahoo_crumb_ts: float = 0.0        # crumb获取时间戳
_v7_active: bool = False            # v7 API是否可用


def fetch_kline_data(ticker: str, period: str = "1d", interval: str = "5m"):
    """
    获取单只股票的K线数据
    
    参数:
        ticker: 股票代码
        period: 时间范围 (1d/5d/1mo/3mo)
        interval: K线周期 (1m/5m/15m/30m/1h/1d)
    
    返回:
        [{time, open, high, low, close, volume}, ...] 或空列表
    """

    
    cache_key = f"{ticker}_{period}_{interval}"
    
    # 检查缓存
    now = time.time()
    if cache_key in _kline_cache:
        cached = _kline_cache[cache_key]
        if cached and now - cached.get("_ts", 0) < _kline_fetch_interval:
            return cached.get("data", [])
    
    
    try:
        tk = yf.Ticker(ticker)
        hist = tk.history(period=period, interval=interval)
        
        if hist.empty:
            return []
        
        candles = []
        for idx, row in hist.iterrows():
            candles.append({
                'time': int(idx.timestamp()),
                'open': round(float(row['Open']), 2),
                'high': round(float(row['High']), 2),
                'low': round(float(row['Low']), 2),
                'close': round(float(row['Close']), 2),
                'volume': int(row['Volume']),
            })
        
        _kline_cache[cache_key] = {"data": candles, "_ts": now}
        return candles
    except Exception as e:
        logger.debug(f"K线数据获取失败 {ticker}: {e}")
        return []

def fetch_yahoo_prices():
    """从 Yahoo Finance 抓取实时价格（v7批量API，单次请求获取全部股票）"""
    global _last_yahoo_fetch, _yahoo_fetch_success, _yahoo_error_count
    global _yahoo_fetch_interval, _data_source, _first_yahoo_fetch_done
    global _yahoo_session, _yahoo_crumb, _yahoo_crumb_ts, _v7_active

    now = time.time()

    base_interval = YAHOO_INTERVALS.get(
        _clock.get_status()[0].value if _clock else 'CLOSED', 10)
    jitter = (_iteration_count % 5) - 2
    effective_interval = max(3, base_interval + jitter)
    if not _first_yahoo_fetch_done:
        effective_interval = 0

    if now - _last_yahoo_fetch < effective_interval:
        return None, 0

    _last_yahoo_fetch = now
    t0 = time.time()
    all_tickers = list(TRACKED_TICKERS) + ['^IXIC']
    prices = {}

    # ── 初始化/刷新 Yahoo Session ──
    if _yahoo_session is None or (now - _yahoo_crumb_ts) > 900:
        try:
            if _yahoo_session is not None:
                _yahoo_session.close()
            _yahoo_session = requests.Session()
            _yahoo_session.headers.update({
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
            })
            _yahoo_session.get('https://fc.yahoo.com/', timeout=5)
            r = _yahoo_session.get('https://query2.finance.yahoo.com/v1/test/getcrumb', timeout=5)
            _yahoo_crumb = r.text.strip()
            _yahoo_crumb_ts = now
        except Exception:
            _yahoo_crumb = None

    # ── 方案1: v7 批量报价API（单请求，<0.2s） ──
    if _yahoo_crumb:
        try:
            symbols = ','.join(all_tickers)
            url = f'https://query2.finance.yahoo.com/v7/finance/quote?symbols={symbols}&crumb={_yahoo_crumb}'
            resp = _yahoo_session.get(url, timeout=8)
            if resp.status_code == 200:
                data = resp.json()
                for r in data.get('quoteResponse', {}).get('result', []):
                    sym = r.get('symbol', '')
                    p = r.get('regularMarketPrice') or r.get('lastPrice') or r.get('previousClose')
                    if sym and p and float(p) > 0:
                        prices[sym] = float(p)

                if len(prices) >= 5:
                    latency = (time.time() - t0) * 1000
                    _yahoo_fetch_success = True
                    _yahoo_error_count = 0
                    _first_yahoo_fetch_done = True
                    _data_source = 'Yahoo v7 (批量)'
                    _v7_active = True
                    print(f"[Yahoo] OK {len(prices)}只 {latency:.0f}ms (v7 batch)", flush=True)
                    return prices, latency
        except Exception:
            _yahoo_crumb = None  # crumb失效，下次刷新

    # ── 方案2: fast_info 并行回退（6 workers, 8s超时） ──
    try:
        dl = [t for t in all_tickers if t not in prices]

        def _get_one(tkr):
            try:
                tk = yf.Ticker(tkr)
                p = tk.fast_info.get('lastPrice') or tk.fast_info.get('regularMarketPrice') or tk.fast_info.get('previousClose')
                if p and p > 0:
                    return (tkr, float(p))
            except Exception as e:
                logger.debug(f"yfinance fetch failed for {tkr}: {e}")
            return (tkr, None)

        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = {pool.submit(_get_one, t): t for t in dl}
            for f in as_completed(futures, timeout=8):
                try:
                    tkr, p = f.result(timeout=3)
                    if p:
                        prices[tkr] = p
                except Exception as e:
                    logger.debug(f"ThreadPool result error: {e}")
    except Exception as e:
        logger.debug(f"Yahoo fetch thread pool error: {e}")

    latency = (time.time() - t0) * 1000

    if len(prices) >= 5:
        _yahoo_fetch_success = True
        _yahoo_error_count = 0
        _first_yahoo_fetch_done = True
        _data_source = 'Yahoo Finance (fast_info)'
        print(f"[Yahoo] OK {len(prices)}只 {latency:.0f}ms (fast_info)", flush=True)
        return prices, latency
    else:
        _yahoo_fetch_success = False
        _yahoo_error_count += 1
        _first_yahoo_fetch_done = True
        print(f"[Yahoo] FAIL {len(prices)}只 {latency:.0f}ms", flush=True)
        return None, latency

def calculate_commission(trade_value: float, quantity: int, side: str = 'BUY') -> float:
    """
    蚂蚁银行澳门 美股佣金计算
    
    费用组成:
    - 佣金: 0.05% × 成交金额, 最低 USD 2.00
    - SEC费(仅卖出): 0.0008% × 成交金额
    - TAF费(仅卖出): USD 0.00013 × 股数
    
    参数:
        trade_value: 成交金额 (price × quantity)
        quantity: 股数
        side: 'BUY' 或 'SELL'
    
    返回:
        总佣金(USD)
    """
    # 基础佣金: 0.05%, 最低$2.00
    base_commission = max(trade_value * 0.0005, 2.00)
    
    total = base_commission
    
    if side == 'SELL':
        # SEC费用: 0.0008% × 成交金额
        sec_fee = max(trade_value * 0.000008, 0.01)
        # TAF费用: $0.00013 × 股数
        taf_fee = max(quantity * 0.00013, 0.01)
        total += sec_fee + taf_fee
    
    return round(total, 2)


# ═══════════════════════════════════════════════════════════════
# 激进短线交易参数
# ═══════════════════════════════════════════════════════════════
TRADE_CHECK_INTERVAL = 60      # 每60秒检查一次交易信号(原300秒)

# Yahoo Finance 抓取配置

PROFIT_TAKE_THRESHOLD = 0.03   # 止盈阈值3%(原10%)
STOP_LOSS_THRESHOLD = -0.04    # 止损阈值-4%(原-8%)
MAX_POSITION_HOLD_TIME = 30    # 最大持仓时间(分钟),超时强制平仓
REENTRY_COOLDOWN = 5           # 卖出后冷却时间(分钟)
POSITION_MAX_PCT = 0.08        # 单只股票最大仓位8%
PREDICTIVE_SELL_THRESHOLD = 0.55  # 模型预测准确率>55%时触发预测卖出

# Leverage Trading Parameters
MAX_LEVERAGE = 2.0
LEVERAGE_TIERS = {0.55: 1.0, 0.60: 1.25, 0.65: 1.5, 0.70: 1.75, 0.75: 2.0}
MAX_POSITION_PCT_LEVERAGED = 0.12
LEVERAGE_STOP_LOSS = -0.025
DRAWDOWN_DELEVERAGE = 0.10

# 持仓计时器: {ticker: buy_iteration}
_position_entry_time: dict = {}
# 卖出冷却: {ticker: sell_iteration}
_position_sell_cooldown: dict = {}

# 状态持久化: 启动时尝试恢复上次保存的状态
_saved_state = load_state()
if _saved_state:
    try:
        deserialize_portfolio(_portfolio, _saved_state['portfolio'])
        deserialize_accuracy(_accuracy, _saved_state['accuracy'])
        deserialize_benchmark(_benchmark, _saved_state['benchmark'])
        # 恢复统计预测器状态
        pp = _saved_state.get('predictor')
        if pp:
            _predictor = RealtimePredictor.from_dict(pp)
        g = _saved_state.get('globals', {})
        _current_prices.update(g.get('current_prices', {}))
        _previous_prices.update(g.get('previous_prices', {}))
        _iteration_count = g.get('iteration_count', 0)
        _positions_initialized = g.get('positions_initialized', False)
        _ml_ready = g.get('ml_ready', False)
        _market_opened = g.get('market_opened', False)
        # 恢复持仓计时器和冷却
        _position_entry_time.update(g.get('position_entry_time', {}))
        # 恢复预测迭代映射
        _prediction_iters.update(g.get('prediction_iters', {}))
        _position_sell_cooldown.update(g.get('position_sell_cooldown', {}))
        # 恢复杠杆引擎状态
        if g.get('leverage_engine'):
            _leverage_engine = LeverageEngine.from_dict(g['leverage_engine'])
        # 恢复交易信号
        sigs = g.get('recent_signals', [])
        if sigs:
            _recent_signals.extend(sigs)
        saved_at = _saved_state.get('saved_at', '?')
        print(f'  ✅ 已恢复交易状态 (保存于 {saved_at})')
        print(f'     现金: ${_portfolio.cash:,.2f} | 持仓: {len(_portfolio.positions)}只')
        print(f'     预测: {len(_accuracy.predictions)}条 | 交易: {len(_portfolio.trade_history)}笔')
        print(f'     迭代: {_iteration_count} | 建仓: {_positions_initialized}')
    except Exception as e:
        print(f'  ⚠️ 状态恢复失败: {e}，使用全新状态')
else:
    print('  🆕 使用全新交易状态')

# ═══════════════════════════════════════════════════════════════
# 基准初始化：拉取纳指历史数据用于曲线对比
# ═══════════════════════════════════════════════════════════════
if _benchmark.nasdaq_equity_curve.empty:
    print('  📊 拉取纳指历史数据用于基准曲线...')
    ok = _benchmark.fetch_nasdaq_history('6mo')
    if ok:
        print(f'  ✅ 纳指基准已初始化 ({len(_benchmark.nasdaq_equity_curve)} 数据点)')
    else:
        print('  ⚠️ 纳指历史数据拉取失败，基准曲线将在交易开始后逐步建立')

# 初始建仓配置
INITIAL_POSITIONS = {
    'AAPL':  {'qty': 17, 'cost': 210},
    'MSFT':  {'qty': 8, 'cost': 430},
    'NVDA':  {'qty': 30, 'cost': 125},
    'GOOGL':  {'qty': 20, 'cost': 180},
    'AMZN':  {'qty': 17, 'cost': 215},
    'META':  {'qty': 6, 'cost': 590},
    'TSLA':  {'qty': 12, 'cost': 310},
    'NFLX':  {'qty': 4, 'cost': 900},
    'JPM':  {'qty': 15, 'cost': 240},
    'V':  {'qty': 12, 'cost': 310},
    'MA':  {'qty': 7, 'cost': 500},
    'WMT':  {'qty': 39, 'cost': 95},
    'HD':  {'qty': 9, 'cost': 390},
    'NKE':  {'qty': 44, 'cost': 85},
    'SBUX':  {'qty': 37, 'cost': 100},
    'AVGO':  {'qty': 2, 'cost': 1800},
    'AMD':  {'qty': 23, 'cost': 160},
    'ADBE':  {'qty': 7, 'cost': 520},
    'CRM':  {'qty': 13, 'cost': 280},
    'NOW':  {'qty': 4, 'cost': 850},
    'BAC':  {'qty': 21, 'cost': 44},
    'INTC': {'qty': 33, 'cost': 28},
    'ORCL': {'qty': 10, 'cost': 180},
    'QCOM': {'qty': 8, 'cost': 220},
    'TXN':  {'qty': 8, 'cost': 195},
    'UBER': {'qty': 15, 'cost': 72},
    # 光模块/光通信
    'AAOI': {'qty': 30, 'cost': 15},
    'COHR': {'qty': 8, 'cost': 100},
    'LITE': {'qty': 15, 'cost': 85},
    'FN':   {'qty': 7, 'cost': 220},
    # 存储
    'WDC':  {'qty': 10, 'cost': 70},
    'STX':  {'qty': 11, 'cost': 105},
    'NTAP': {'qty': 9, 'cost': 120},
    # 数据中心芯片
    'MRVL': {'qty': 13, 'cost': 110},
    'MU':   {'qty': 10, 'cost': 140},
    # 半导体设备
    'LRCX': {'qty': 6, 'cost': 95},
    'AMAT': {'qty': 8, 'cost': 230},
    'KLAC': {'qty': 5, 'cost': 800},
    # EDA软件
    'SNPS': {'qty': 5, 'cost': 560},
    'CDNS': {'qty': 5, 'cost': 310},
}


def init_positions():
    """初始化建仓 - 使用Yahoo实时价格作为成本价"""
    global _positions_initialized, _portfolio, _current_prices, _data_source, _last_price_update
    
    # 已有持仓(从持久化恢复): 跳过建仓
    if len(_portfolio.positions) > 0:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ 恢复持仓: {len(_portfolio.positions)}只, 跳过建仓")
        for ticker in list(_portfolio.positions.keys()):
            if ticker not in _current_prices:
                _current_prices[ticker] = _portfolio.positions[ticker].avg_cost
        if '^IXIC' not in _current_prices:
            _current_prices['^IXIC'] = 18200.0
        _positions_initialized = True
        return
    
    # 先从Yahoo获取实时价格作为建仓成本（批量下载避免限流）
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 📡 获取Yahoo实时价格用于建仓...", flush=True)
    
    real_prices = {}
    
    # 方式1: 尝试批量下载（单次请求）
    try:
        batch_tickers = list(INITIAL_POSITIONS.keys())
        data = yf.download(batch_tickers, period='1d', interval='1m', progress=False, threads=False)
        if not data.empty:
            for ticker in batch_tickers:
                col = ('Close', ticker)
                if col in data.columns:
                    prices_col = data[col].dropna()
                    if len(prices_col) > 0:
                        real_prices[ticker] = float(prices_col.iloc[-1])
            if real_prices:
                print(f"  批量获取: {len(real_prices)}/{len(batch_tickers)} 只", flush=True)
    except Exception as e:
        print(f"  批量下载失败: {e}", flush=True)
    
    # 方式2: 对失败的逐个重试
    for ticker in INITIAL_POSITIONS:
        if ticker in real_prices:
            continue
        try:
            tk = yf.Ticker(ticker)
            # 多数据源fallback
            p = (tk.fast_info.get('lastPrice') or 
                 tk.fast_info.get('regularMarketPrice') or
                 tk.fast_info.get('previousClose'))
            if not p:
                hist = tk.history(period='1d')
                if not hist.empty:
                    p = float(hist['Close'].iloc[-1])
            if p and p > 0:
                real_prices[ticker] = float(p)
        except Exception:
            pass
    
    # 获取纳指价格
    try:
        ix = yf.Ticker('^IXIC')
        ix_p = ix.fast_info.get('lastPrice') or ix.fast_info.get('regularMarketPrice') or 18000
        if ix_p:
            _current_prices['^IXIC'] = float(ix_p)
    except Exception:
        _current_prices['^IXIC'] = 18000.0
    
    if len(real_prices) < 5:
        # Yahoo不可用: 回退到预设价格
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚠️ Yahoo价格不足, 使用预设价格")
        for ticker, info in INITIAL_POSITIONS.items():
            real_prices[ticker] = info['cost']
        _data_source = 'fallback (预设价格)'
    else:
        _data_source = 'Yahoo Finance (实时)'
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✓ 获取到 {len(real_prices)} 只实时价格")
    
    # 用真实价格建仓, 调整数量以匹配初始资金分配
    # 总资金80%用于建仓，20%留作短线交易现金
    trading_cash = 100000.0 * 0.20
    investable = 100000.0 - trading_cash
    cash_per_stock = investable / len(INITIAL_POSITIONS)  # 每只约$10,000
    total_commission = 0
    
    for ticker, info in INITIAL_POSITIONS.items():
        price = real_prices.get(ticker, info['cost'])
        qty = max(1, int(cash_per_stock / price))  # 按价格计算股数
        trade_val = qty * price
        comm = calculate_commission(trade_val, qty, 'BUY')
        total_commission += comm
        
        _portfolio.execute_buy(ticker, qty, price, commission=comm, reason='开盘建仓')
        _current_prices[ticker] = price
        _recent_signals.append({
            'time': datetime.now().strftime('%H:%M:%S'),
            'action': 'BUY', 'ticker': ticker,
            'qty': qty, 'price': round(price, 2),
            'reason': '开盘建仓(Yahoo价格)',
        })
        print(f"  {ticker}: {qty}股 @ ${price:.2f} (佣金${comm:.2f})", flush=True)
    
    _last_price_update = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    if '^IXIC' not in _current_prices:
        _current_prices['^IXIC'] = 18000.0
    
    # 设置纳指基准（如果 fetch_nasdaq_history 还未设置）
    if _benchmark.nasdaq_start_price == 0:
        nasdaq_price = _current_prices.get('^IXIC', 18000.0)
        _benchmark.nasdaq_start_price = nasdaq_price
        _benchmark.nasdaq_shares = 100000.0 / nasdaq_price if nasdaq_price > 0 else 5.5
    
    _positions_initialized = True
    total_invested = 100000.0 - _portfolio.cash - total_commission
    print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ 建仓完成: {len(_portfolio.positions)}只, " +
          f"投入${total_invested:,.0f}, 佣金${total_commission:.2f}", flush=True)


def tick_engine():
    """引擎一个tick（模拟价格波动+交易决策）"""
    global _current_prices, _previous_prices, _iteration_count
    global _market_opened, _positions_initialized, _recent_signals
    
    _iteration_count += 1
    
    # 检查市场是否开盘
    status, _ = _clock.get_status()
    is_open_now = (status == MarketStatus.REGULAR_HOURS)
    # 是否为正常交易时段（仅此时段允许买卖交易）
    is_trading_session = is_open_now
    
    # 开盘瞬间执行建仓
    if is_trading_session and not _positions_initialized:
        _market_opened = True
        init_positions()
    
    # ═══════════════════════════════════════════════════════════
    # ═══════════════════════════════════════════════════════════
    # 价格更新：始终尝试获取Yahoo价格（即使未建仓），确保数据新鲜度
    # ═══════════════════════════════════════════════════════════
    global _last_price_update, _price_update_count, _data_latency_ms
    global _price_is_stale, _data_source, _price_data_age_s
    _previous_prices = dict(_current_prices)
    
    # 尝试从 Yahoo Finance 抓取真实价格
    # 启动后首次无条件尝试（_last_yahoo_fetch=0 保证首次通过）
    yahoo_prices, yahoo_latency = fetch_yahoo_prices()
    
    if yahoo_prices:
        # 使用Yahoo真实价格
        for ticker, price in yahoo_prices.items():
            _current_prices[ticker] = price
        _data_latency_ms = round(yahoo_latency, 2)
        _data_source = 'Yahoo Finance (实时)'
        _last_price_update = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        _price_update_count += 1
        _price_is_stale = False; _data_is_stale = False
        _price_data_age_s = 0  # 数据新鲜
    else:
        # Yahoo未返回新数据：可能是间隔未到，或请求失败
        # 更新数据年龄（距上次更新秒数）
        if _last_price_update:
            try:
                last_dt = datetime.strptime(_last_price_update[:19], '%Y-%m-%d %H:%M:%S')
                _price_data_age_s = (datetime.now() - last_dt).total_seconds()
            except Exception:
                pass
        # 更新 _data_latency_ms 反映数据新鲜度（距上次成功更新的毫秒数）
        if _price_data_age_s > 0 and _price_data_age_s < 999:
            _data_latency_ms = _price_data_age_s * 1000
        
        # 仅当连续失败超过阈值时才标记数据为过期
        if _yahoo_error_count > 3:
            _price_is_stale = True; _data_is_stale = True
            _data_source = f'离线 (Yahoo不可用, 数据已过期){" [v7回退]" if _v7_active else ""}'
        elif _yahoo_error_count > 0:
            # 有失败但未超过阈值：保持当前状态，来源标记为等待重连
            _price_is_stale = False  # 上次数据仍有效，不标记过期
            _data_source = f'Yahoo Finance (部分失败, 上次数据仍有效){" [v7]" if _v7_active else ""}'
        else:
            # 完全正常：仅因间隔未到，保持现有数据
            _price_is_stale = False  # 上次数据仍然有效
            # 计算数据年龄用于前端展示
            age_s = int(_price_data_age_s) if _price_data_age_s < 999 else 0
            _data_source = f'Yahoo Finance (缓存{" [v7]" if _v7_active else ""}, {age_s}s前)'
    
    # 未开盘且未建仓：不执行交易操作（但价格已更新供前端展示）
    if not _positions_initialized:
        return
    
    # 更新持仓市值 & 喂入统计预测器
    prices_for_portfolio = {t: p for t, p in _current_prices.items() if t != '^IXIC'}
    _portfolio.update_prices(prices_for_portfolio)
    # 喂入价格到统计预测器
    for ticker, price in _current_prices.items():
        if ticker != '^IXIC':
            _predictor.update_price(ticker, price)
    
    # 更新基准
    _benchmark.update(_current_prices['^IXIC'], _portfolio.get_total_equity())
    
    # Daily interest accrual for borrowed funds
    _portfolio.accrue_interest()
    
    # ═══════════════════════════════════════════════════════════
    # 交易会话门控：仅在正常交易时段执行买卖
    # ═══════════════════════════════════════════════════════════
    if not is_trading_session:
        return  # 休市时不执行任何交易，仅更新价格供展示
    
    # 价格数据过期时跳过交易决策（避免基于伪造数据交易）
    if _price_is_stale:
        return
    
    # 激进短线交易逻辑 (每60秒检查)
    if _iteration_count % TRADE_CHECK_INTERVAL == 0 and _portfolio.cash > 1000:
        current_equity = _portfolio.get_total_equity()
        
        for ticker, pos in list(_portfolio.positions.items()):
            if ticker not in _current_prices:
                continue
            
            price = _current_prices[ticker]
            pnl_pct = pos.unrealized_pnl_pct
            hold_minutes = (_iteration_count - _position_entry_time.get(ticker, _iteration_count)) / 60
            
            # 使用已更新的持仓数据
            
            should_sell = False
            sell_reason = ''
            
            # 条件1: 止盈 (盈利≥3%时80%概率卖出,锁定利润)
            # 使用确定性哈希避免随机数滥用，同时保持概率分散效果
            _tp_seed = int(hashlib.md5(f'{_iteration_count}:{ticker}:tp'.encode()).hexdigest()[:8], 16)
            if pnl_pct >= PROFIT_TAKE_THRESHOLD and (_tp_seed % 100) < 80:
                should_sell = True
                sell_reason = f'短线止盈({pnl_pct:.1%})'
            
            # 条件2: 止损 (杠杆-2.5%, 普通-4%)
            effective_sl = LEVERAGE_STOP_LOSS if _portfolio.get_leverage_ratio() > 1.2 else STOP_LOSS_THRESHOLD
            if pnl_pct <= effective_sl:
                _sl_seed = int(hashlib.md5(f"{_iteration_count}:{ticker}:sl".encode()).hexdigest()[:8], 16)
                if (_sl_seed % 100) < 90:
                    should_sell = True
                    lev_tag = "[L]" if _portfolio.get_leverage_ratio() > 1.2 else ""
                    sell_reason = f"止损({pnl_pct:.1%}){lev_tag}"
            
            # 条件3: 预测性卖出 (模型预测下跌时50%概率减仓)
            if not should_sell and pnl_pct < 0.01:  # 微利或微亏时更积极
                # 使用ML模型或统计预测器判断是否卖出
                ml_sell_signal = False
                if _ml_ready and _ml_inference is not None:
                    ml_pred = _ml_inference.predict(ticker)
                    if ml_pred and ml_pred['direction'] == 0 and ml_pred['confidence'] > 0.55:
                        ml_sell_signal = True
                else:
                    pred_dir, pred_conf, _ = _predictor.predict(ticker, price)
                    if pred_dir == 0 and pred_conf > 0.55:
                        ml_sell_signal = True
                
                if ml_sell_signal:
                    should_sell = True
                    sell_reason = f'ML预测卖出'
            
            # 条件4: 持仓超时强制平仓 (独立检查,不受上面条件阻塞)
            if not should_sell and hold_minutes > MAX_POSITION_HOLD_TIME:
                should_sell = True
                sell_reason = f'超时平仓({hold_minutes:.0f}分钟)'
            
            if should_sell:
                trade_val = pos.quantity * price
                comm = calculate_commission(trade_val, pos.quantity, 'SELL')
                _portfolio.execute_sell(ticker, pos.quantity, price,
                                        commission=comm, reason=sell_reason)
                # 记录交易到杠杆引擎(绩效反馈)
                _leverage_engine.record_trade(
                    win=(pos.unrealized_pnl > 0),
                    pnl_pct=pos.unrealized_pnl_pct
                )
                _position_sell_cooldown[ticker] = _iteration_count
                # 记录交易指令
                _recent_signals.append({
                    'time': datetime.now().strftime('%H:%M:%S'),
                    'action': 'SELL',
                    'ticker': ticker,
                    'qty': pos.quantity,
                    'price': round(price, 2),
                    'reason': sell_reason,
                })
                if len(_recent_signals) > 50:
                    _recent_signals = _recent_signals[-50:]
                break  # 每轮最多一笔卖出
        
        # 买入逻辑: 在卖出后或现金充裕时寻找新机会
        if _portfolio.cash > 5000:
            # 获取不在冷却期且未持仓的候选股票
            candidates = []
            for t in TRACKED_TICKERS:
                if t in _portfolio.positions:
                    continue
                cooldown_iter = _position_sell_cooldown.get(t, 0)
                if (_iteration_count - cooldown_iter) < REENTRY_COOLDOWN * 60:
                    continue  # 还在冷却期
                candidates.append(t)
            
            if candidates:
                # 根据"ML信号"选择最佳候选 (模拟:偏向于选择价格波动的)
                best_ticker = None
                best_signal = -1
                
                for t in candidates:
                    price = _current_prices.get(t, 0)
                    if price <= 0:
                        continue
                    
                    # 优先使用Transformer模型，降级到统计预测器
                    if _ml_ready and _ml_inference is not None:
                        ml_pred = _ml_inference.predict(t)
                        if ml_pred and ml_pred['direction'] == 1:
                            signal = ml_pred['confidence']
                        else:
                            signal = -1
                    else:
                        pred_direction, pred_conf, _ = _predictor.predict(t, price)
                        signal = pred_conf if pred_direction == 1 else -pred_conf
                    
                    if signal > best_signal:
                        best_signal = signal
                        best_ticker = t
                
                if best_ticker and best_signal > 0.005:
                    price = _current_prices[best_ticker]
                    
                    # ── 动态杠杆引擎: 凯利公式+波动率+绩效+热度 ──
                    leverage, lev_detail = _leverage_engine.calculate(
                        confidence=best_signal if best_signal > 0.5 else 0.52,
                        ticker=best_ticker,
                        current_prices=_current_prices,
                        prev_prices=_previous_prices,
                        portfolio=_portfolio,
                        accuracy=_accuracy,
                    )
                    # 如果杠杆低于0.5x, 机会不够好, 跳过
                    if leverage < 0.5:
                        best_ticker = None  # 放弃这次买入
                    
                    max_pct = MAX_POSITION_PCT_LEVERAGED if leverage > 1.0 else POSITION_MAX_PCT
                    max_position_value = current_equity * max_pct
                    available = _portfolio.get_available_cash()
                    max_qty_by_cash = int(available * 0.20 / price)
                    max_qty_by_limit = int(max_position_value / price)
                    qty = min(max_qty_by_cash, max_qty_by_limit, 200 if leverage > 1.0 else 100)
                    
                    if qty > 0:
                        # 风控检查
                        if not _risk_mgr.in_trade.is_paused:
                            trade_val = qty * price
                            comm = calculate_commission(trade_val, qty, 'BUY')
                            buy_reason = f'ML{leverage}x' if leverage > 1.0 else 'ML短线买入'
                            _portfolio.execute_buy(best_ticker, qty, price,
                                                   commission=comm, reason=buy_reason)
                            _position_entry_time[best_ticker] = _iteration_count
                        # 记录交易指令
                            _recent_signals.append({
                                'time': datetime.now().strftime('%H:%M:%S'),
                                'action': 'BUY',
                                'ticker': best_ticker,
                                'qty': qty,
                                'price': round(price, 2),
                                'reason': buy_reason,
                            })
                            if len(_recent_signals) > 50:
                                _recent_signals = _recent_signals[-50:]
    
    # ═══════════════════════════════════════════════════════════
    # 真实统计预测（每30次约30秒生成一次预测）
    # 使用多因子模型: 动量+均值回归+成交量+波动率
    # ═══════════════════════════════════════════════════════════
    if _iteration_count % 30 == 0:
        # 只从有价格数据的股票中选取
        available_tickers = [t for t in TRACKED_TICKERS if t in _current_prices and _current_prices[t] > 0]
        if not available_tickers:
            available_tickers = list(TRACKED_TICKERS)
        ticker = available_tickers[_iteration_count % len(available_tickers)]
        price = _current_prices.get(ticker, 0)
        if price > 0:
            # 使用统计预测器生成真实信号
            direction, confidence, factors = _predictor.predict(ticker, price)
            predicted_return = (price / _previous_prices.get(ticker, price) - 1) if ticker in _previous_prices else 0.001
            
            pred_id = _accuracy.record_prediction(
                ticker, predicted_return, direction, confidence
            )
            _prediction_iters[pred_id] = _iteration_count
            
            # 确认该ticker的所有旧预测（已过30秒）
            # 查找30秒前同一ticker的未确认预测来确认
            candidates_to_confirm = []
            for pid, pr in _accuracy.predictions.items():
                if pr.ticker == ticker and pr.status == 'pending' and pr.id < pred_id:
                    candidates_to_confirm.append((pid, pr))
            
            if candidates_to_confirm:
                # 确认最早的预测
                candidates_to_confirm.sort(key=lambda x: x[0])
                oldest_pid, oldest_pr = candidates_to_confirm[0]
                old_price = _previous_prices.get(ticker, price)
                actual_return = (price / old_price - 1) if old_price > 0 else 0
                actual_direction = 1 if actual_return > 0 else 0
                _accuracy.confirm_prediction(oldest_pid, actual_return, actual_direction)
            
            # 额外：确认超过2分钟(120迭代)的未确认预测，使用当前价
            for pid, pr in list(_accuracy.predictions.items()):
                if pr.status == 'pending' and (_iteration_count - _prediction_iters.get(pid, 0)) > 120:
                    actual_return = (price / _previous_prices.get(pr.ticker, price) - 1) if pr.ticker in _previous_prices else 0
                    actual_direction = 1 if actual_return > 0 else 0
                    _accuracy.confirm_prediction(pid, actual_return, actual_direction)
    
    # 超时确认：每60秒确认超过90秒未确认的预测
    if _iteration_count % 60 == 0:
        for pid, pr in list(_accuracy.predictions.items()):
            if pr.status == 'pending':
                # 用预测时的迭代数判断是否超时（90秒 ≈ 90迭代）
                pred_iter = _prediction_iters.get(pid, 0)
                if (_iteration_count - pred_iter) > 90:
                    ticker = pr.ticker
                    current_p = _current_prices.get(ticker, 0)
                    if current_p > 0 and ticker in _previous_prices:
                        actual_return = (current_p / _previous_prices[ticker] - 1)
                        actual_direction = 1 if actual_return > 0 else 0
                        _accuracy.confirm_prediction(pid, actual_return, actual_direction)



def _collect_globals_dict():
    """收集全局运行时状态，避免重复代码"""
    return {
        'current_prices': _current_prices,
        'previous_prices': _previous_prices,
        'iteration_count': _iteration_count,
        'positions_initialized': _positions_initialized,
        'market_opened': _market_opened,
        'position_entry_time': dict(_position_entry_time),
        'position_sell_cooldown': dict(_position_sell_cooldown),
        'recent_signals': _recent_signals[-20:],
        'predictor': _predictor.to_dict(),
        'ml_ready': _ml_ready,
        'prediction_iters': dict(_prediction_iters),
        'leverage_engine': _leverage_engine.to_dict(),
    }


def init_ml_model():
    """后台加载ML模型（异步，不阻塞启动）"""
    global _ml_inference, _ml_ready
    try:
        _ml_inference = ModelInference()
        if _ml_inference.load():
            _ml_ready = True
            print(f'[{datetime.now().strftime("%H:%M:%S")}] 🧠 Transformer模型加载成功', flush=True)
            return True
        else:
            print(f'[{datetime.now().strftime("%H:%M:%S")}] ⚠️ Transformer模型文件不存在，使用统计预测器', flush=True)
    except Exception as e:
        print(f'[{datetime.now().strftime("%H:%M:%S")}] ⚠️ ML模型加载失败: {e}', flush=True)
    return False

def engine_loop():
    """后台引擎循环（每秒1次）"""
    global _engine_running, _iteration_count, _ml_ready
    _offline_training_done = False
    _ml_model_loaded = _ml_ready  # 从持久化状态恢复
    
    def trigger_offline_training():
        """在休市时离线训练Transformer"""
        nonlocal _offline_training_done
        if _offline_training_done:
            return
        status, _ = _clock.get_status()
        if status != MarketStatus.REGULAR_HOURS and _iteration_count > 300:
            _offline_training_done = True  # 防止重复触发
            print(f'[{datetime.now().strftime("%H:%M:%S")}] 🧠 休市中，启动离线Transformer训练（后台）...', flush=True)
            # 在后台线程中训练，避免阻塞引擎循环
            def _train_thread():
                try:
                    from live_trading.predictor import train_offline_transformer
                    acc = train_offline_transformer()
                    if acc:
                        print(f'[{datetime.now().strftime("%H:%M:%S")}] ✅ 离线训练完成，准确率: {acc:.2%}', flush=True)
                except Exception as e:
                    print(f'[{datetime.now().strftime("%H:%M:%S")}] ⚠️ 离线训练失败: {e}', flush=True)
            threading.Thread(target=_train_thread, daemon=True, name='offline-train').start()
    
    # 启动时立即加载ML模型
    print(f'[{datetime.now().strftime("%H:%M:%S")}] 🔄 加载ML模型...', flush=True)
    if init_ml_model():
        _ml_model_loaded = True
        _ml_ready = True
        print(f'[{datetime.now().strftime("%H:%M:%S")}] ✅ ML模型就绪', flush=True)
    else:
        print(f'[{datetime.now().strftime("%H:%M:%S")}] ⚠️ ML模型未加载', flush=True)
    
    while _engine_running:
        try:
            # 延迟加载ML模型（每60秒重试一次，避免频繁创建对象）
            if not _ml_model_loaded and _iteration_count % 60 == 0:
                if init_ml_model():
                    _ml_model_loaded = True
                    _ml_ready = True  # 同步全局状态（已在函数顶部声明global）
            tick_engine()
            # 休市时每10分钟检查是否需要离线训练
            if _iteration_count % 600 == 0:
                trigger_offline_training()
            # 每60次迭代（约60秒）自动保存状态
            if _iteration_count > 0 and _iteration_count % 60 == 0:
                save_state(_portfolio, _accuracy, _benchmark, _collect_globals_dict())
        except Exception as e:
            print(f"引擎异常: {e}", flush=True)
        time.sleep(1)
    
    # 引擎停止时保存最终状态（包含全部运行时状态）
    save_state(_portfolio, _accuracy, _benchmark, _collect_globals_dict())
    print('💾 最终状态已保存', flush=True)


# ============================================================================
# API 路由
# ============================================================================

@app.route('/api/health')
def api_health():
    """健康检查端点（供watchdog使用）"""
    status, desc = _clock.get_status()
    return jsonify({
        'status': 'ok',
        'uptime_iterations': _iteration_count,
        'positions_initialized': _positions_initialized,
        'market_status': status.value,
        'timestamp': datetime.now().isoformat(),
    })


@app.route('/api/tickers')
def api_tickers():
    """返回所有追踪的股票代码列表"""
    return jsonify({
        'tickers': list(TRACKED_TICKERS),
        'count': len(TRACKED_TICKERS),
    })

@app.route('/api/signals')
def api_signals():
    """交易指令端点 - 返回Markdown格式的交易信号"""
    if not _recent_signals:
        return jsonify({'signals_md': '*(暂无交易指令)*', 'signals': []})
    
    lines = []
    for s in _recent_signals[-20:]:  # 最近20条
        emoji = '🟢' if s['action'] == 'BUY' else '🔴'
        lines.append(
            f"**{emoji} {s['action']}** {s['ticker']} "
            f"{s['qty']}股 @ ${s['price']:.2f} "
            f"| {s['time']} | {s['reason']}"
        )
    
    return jsonify({
        'signals_md': '\n'.join(lines),
        'signals': _recent_signals[-20:],
    })

@app.route('/')
def index():
    return render_template('dashboard.html')



def build_status_data():
    """构建状态数据字典（供API和WebSocket共用）"""
    status, desc = _clock.get_status()
    market_info = _clock.get_trading_session_info()
    waiting_for_open = not _positions_initialized
    
    positions_data = []
    for ticker, pos in sorted(_portfolio.positions.items(),
                               key=lambda x: abs(x[1].market_value), reverse=True):
        positions_data.append({
            'ticker': ticker,
            'quantity': pos.quantity,
            'avg_cost': round(pos.avg_cost, 2),
            'current_price': round(pos.current_price, 2),
            'market_value': round(pos.market_value, 2),
            'cost_basis': round(pos.cost_basis, 2),
            'unrealized_pnl': round(pos.unrealized_pnl, 2),
            'unrealized_pnl_pct': round(pos.unrealized_pnl_pct * 100, 2),
            'day_change_pct': round(pos.day_change_pct * 100, 2),
            'weight': round(pos.weight * 100, 1),
        })
    
    total_equity = _portfolio.get_total_equity()
    total_market_value = _portfolio.get_total_market_value()
    net_pnl = total_equity - _portfolio.initial_capital
    net_pnl_pct = (net_pnl / _portfolio.initial_capital) if _portfolio.initial_capital > 0 else 0
    
    _benchmark._ensure_curves_synced()  # 确保 API 读取时曲线已同步
    bench = _benchmark.get_snapshot()
    acc = _accuracy.get_snapshot()
    
    recent_trades = []
    trade_df = _portfolio.get_trade_summary(last_n=10)
    if not trade_df.empty:
        for _, row in trade_df.iterrows():
            trade_time = row.get('Time', '')
            if trade_time and ' ' in str(trade_time):
                trade_time = str(trade_time).split(' ')[-1][:8]
            elif trade_time and 'T' in str(trade_time):
                trade_time = str(trade_time).split('T')[-1][:8]
            recent_trades.append({
                'id': row['ID'],
                'ticker': row['Ticker'],
                'side': row['Side'],
                'qty': int(row['Qty']),
                'price': round(float(row['Price']), 2),
                'value': round(float(row['Value']), 2),
                'pnl': round(float(row.get('PnL', 0)), 2) if row.get('PnL') else 0,
                'reason': row.get('Reason', ''),
                'time': trade_time,
            })
    
    countdown = ''
    if status == MarketStatus.REGULAR_HOURS:
        cd = market_info.get('countdown_to_close')
        if cd:
            countdown = f"距闭市 {cd[0]}h {cd[1]:02d}m {cd[2]:02d}s"
    else:
        cd = market_info.get('countdown_to_open')
        if cd:
            h, m, s = cd
            countdown = f"{h}h {m:02d}m {s:02d}s"
    
    return {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'iteration': _iteration_count,
        'waiting_for_open': waiting_for_open,
        'positions_initialized': _positions_initialized,
        'market': {
            'status': status.value,
            'status_desc': desc,
            'is_open': status == MarketStatus.REGULAR_HOURS,
            'is_active': market_info.get('is_active', False),
            'countdown': countdown,
            'current_et': market_info.get('current_et', ''),
            'next_trading_day': market_info.get('next_trading_day', ''),
        },
        'account': {
            'borrowed': round(_portfolio.borrowed, 2),
            'margin_ratio': round(_portfolio.get_margin_ratio() * 100, 1),
            'leverage_ratio': round(_portfolio.get_leverage_ratio(), 2),
            'margin_call_risk': _portfolio.is_margin_call_risk(),
            'total_interest': round(_portfolio.total_interest, 2),
            'initial_capital': _portfolio.initial_capital,
            'cash': round(_portfolio.cash, 2),
            'total_equity': round(total_equity, 2),
            'total_market_value': round(total_market_value, 2),
            'position_count': len(_portfolio.positions),
            'leverage': round(_portfolio.get_leverage(), 2),
            'leverage_avg': round(_leverage_engine.avg_leverage, 2),
            'leverage_calcs': _leverage_engine.total_calculations,
        },
        'pnl': {
            'net_pnl': round(net_pnl, 2),
            'net_pnl_pct': round(net_pnl_pct * 100, 2),
            'realized_pnl': round(_portfolio.realized_pnl, 2),
            'unrealized_pnl': round(sum(p.unrealized_pnl for p in _portfolio.positions.values()), 2),
            'total_commission': round(_portfolio.total_commission, 2),
            'day_pnl': round(_portfolio.get_day_pnl(), 2),
            'day_pnl_pct': round(_portfolio.get_day_pnl_pct() * 100, 2),
            'max_drawdown_pct': round(_portfolio.get_max_drawdown_pct() * 100, 2),
        },
        'positions': positions_data,
        'accuracy': {
            'direction_accuracy': round(acc.direction_accuracy * 100, 1),
            'recent_50_accuracy': round(acc.recent_accuracy_50 * 100, 1),
            'rmse': round(acc.rmse, 6),
            'mae': round(acc.mae, 6),
            'total_predictions': acc.total_predictions,
            'confirmed_predictions': acc.confirmed_predictions,
            'correct_long': acc.correct_long,
            'total_long': acc.total_long,
            'correct_short': acc.correct_short,
            'total_short': acc.total_short,
            'is_acceptable': acc.is_acceptable,
            'trend': acc.accuracy_trend,
        },
        'benchmark': {
            'nasdaq_price': round(bench.nasdaq_price, 2),
            'strategy_return_pct': round(bench.strategy_return_pct * 100, 2),
            'nasdaq_return_pct': round(bench.nasdaq_return_pct * 100, 2),
            'excess_return': round(bench.excess_return * 100, 2),
            'strategy_annual_return': round(bench.strategy_annual_return * 100, 2),
            'nasdaq_annual_return': round(bench.nasdaq_annual_return * 100, 2),
            'strategy_sharpe': round(bench.strategy_sharpe, 3),
            'nasdaq_sharpe': round(bench.nasdaq_sharpe, 3),
            'strategy_max_drawdown': round(bench.strategy_max_drawdown * 100, 2),
            'nasdaq_max_drawdown': round(bench.nasdaq_max_drawdown * 100, 2),
            'alpha': round(bench.alpha, 4),
            'beta': round(bench.beta, 3),
            'information_ratio': round(bench.information_ratio, 3),
            'outperformance_pct': round(bench.outperformance_pct * 100, 2),
        },
        'pending_signals': _recent_signals[-15:],
        'recent_trades': recent_trades,
        'data_quality': {
            'source': _data_source,
            'last_update': _last_price_update,
            'latency_ms': round(_data_latency_ms, 2),
            'update_count': _price_update_count,
            'is_real_time': _data_latency_ms < 5000 and not _price_is_stale,
            'is_stale': _price_is_stale,
            'data_age_s': round(_price_data_age_s, 1) if _price_data_age_s < 999 else 0,
        },
        'ml_ready': _ml_ready,
        'model_info': _ml_inference.get_model_info() if _ml_inference and _ml_inference._loaded else {
            'loaded': False, 'feature_count': 0, 'features': [], 'has_sentiment': False,
        },
    }


@app.route('/api/kline/<ticker>')
def api_kline(ticker):
    """K线数据API"""
    period = request.args.get('period', '1d')
    interval = request.args.get('interval', '5m')
    candles = fetch_kline_data(ticker.upper(), period, interval)
    return jsonify({'ticker': ticker.upper(), 'candles': candles, 'count': len(candles)})

@app.route('/api/kline/multi')
def api_kline_multi():
    """批量K线数据API"""
    tickers_str = request.args.get('tickers', '')
    period = request.args.get('period', '1d')
    interval = request.args.get('interval', '15m')
    tickers = [t.strip().upper() for t in tickers_str.split(',') if t.strip()]
    if not tickers:
        tickers = ['AAPL', 'NVDA', 'MSFT', 'GOOGL']
    
    result = {}
    for t in tickers[:6]:  # 最多6只
        candles = fetch_kline_data(t, period, interval)
        if candles:
            result[t] = candles
    
    return jsonify({'data': result, 'count': len(result)})

@app.route('/api/status')
def api_status():
    """获取完整系统状态"""
    return jsonify(build_status_data())

def start_server(host: str = '0.0.0.0', port: int = 8080, debug: bool = False):
    """启动Web服务器"""
    global _engine_running, _engine_thread
    
    # 不在此处初始化持仓 —— 等开盘后由 tick_engine 执行
    
    _engine_running = True
    _engine_thread = threading.Thread(target=engine_loop, daemon=True)
    _engine_thread.start()
    
    # 检查当前市场状态
    status, desc = _clock.get_status()
    if status == MarketStatus.REGULAR_HOURS:
        print(f"\n  🟢 当前市场已开盘，立即开始交易!")
    else:
        cd = _clock.countdown_to_next_open()
        print(f"\n  ⏳ 等待开盘... 距离开市还有约 {cd[0]}h {cd[1]:02d}m {cd[2]:02d}s")
        print(f"  系统将在美东 09:30 自动建仓并开始交易")
    
    print(f"\n{'='*60}")
    print(f"  美股量化交易系统 - Web仪表盘")
    print(f"  地址: http://localhost:{port}")
    print(f"  刷新频率: 每秒")
    print(f"  按 Ctrl+C 停止")
    print(f"{'='*60}\n", flush=True)
    
    app.run(host=host, port=port, debug=debug, use_reloader=False, threaded=True)



@app.route('/api/benchmark_curve')
def api_benchmark_curve():
    """策略 vs 纳指 权益曲线数据（返回百分比收益率）"""
    try:
        _benchmark._ensure_curves_synced()
        nasdaq_curve = _benchmark.nasdaq_equity_curve
        strategy_curve = _benchmark.strategy_equity_curve
        initial = _benchmark.initial_capital
        
        # 合并两个 Series 的时间戳，取最新的300个（避免全量排序）
        all_indices = nasdaq_curve.index.union(strategy_curve.index)
        latest_indices = all_indices[-300:] if len(all_indices) > 300 else all_indices
        points = []
        
        for t in latest_indices:
            ts = int(pd.Timestamp(t).timestamp())
            n_val = float(nasdaq_curve.get(t, initial))
            s_val = float(strategy_curve.get(t, initial))
            n_pct = round((n_val - initial) / initial * 100, 4) if n_val > 0 else 0
            s_pct = round((s_val - initial) / initial * 100, 4) if s_val > 0 else 0
            points.append({
                'time': ts,
                'nasdaq': n_pct,
                'strategy': s_pct,
            })
        
        nasdaq_last = float(nasdaq_curve.iloc[-1]) if len(nasdaq_curve) > 0 else initial
        strategy_last = float(strategy_curve.iloc[-1]) if len(strategy_curve) > 0 else initial
        
        return jsonify({
            'points': points,
            'count': len(points),
            'nasdaq_current': round((nasdaq_last - initial) / initial * 100, 4),
            'strategy_current': round((strategy_last - initial) / initial * 100, 4),
            'initial_capital': initial,
        })
    except Exception as e:
        return jsonify({'points': [], 'count': 0, 'error': str(e)})

@app.route('/api/backtest_summary')
def api_backtest_summary():
    """回测结果摘要"""
    try:
        from backtesting.performance import BacktestPerformance
        from config.settings import PROCESSED_DATA_DIR
        # 尝试加载已有回测结果或返回空
        return jsonify({
            'available': False,
            'message': '运行 python main.py backtest 生成回测结果',
            'summary': {}
        })
    except Exception as e:
        return jsonify({'available': False, 'message': str(e)})


def _shutdown_handler(signum=None, frame=None):
    """进程退出时保存状态"""
    global _engine_running
    print('\n🛑 正在关闭...', flush=True)
    _engine_running = False
    if _engine_thread and _engine_thread.is_alive():
        _engine_thread.join(timeout=3)
    # 保存状态
    save_state(_portfolio, _accuracy, _benchmark, _collect_globals_dict())
    print('💾 状态已保存，安全退出', flush=True)
    sys.exit(0)

signal.signal(signal.SIGTERM, _shutdown_handler)
signal.signal(signal.SIGINT, _shutdown_handler)


if __name__ == '__main__':
    start_server(debug=False)
