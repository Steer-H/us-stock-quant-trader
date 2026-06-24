"""Live trading web server. Flask app serving dashboard at localhost:8080."""

import sys
import time
import threading
import hashlib
from pathlib import Path
from datetime import datetime
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# macOS Python 3.9 werkzeug selectors bug fix
# werkzeug 在 macOS + Python 3.9 上報 "changelist must be an iterable of select.kevent objects"
# 用 PollSelector 替換 DefaultSelector 以消除此錯誤
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

# 全局引擎實例
_clock = MarketClock()
_portfolio = PortfolioManager(initial_capital=100_000.0)
_benchmark = BenchmarkTracker(initial_capital=100_000.0)
_accuracy = AccuracyTracker(rolling_window=50)
_predictor = RealtimePredictor(window_size=120)
# ML模型推理（加載Transformer）
_ml_inference: ModelInference = None  # 延遲加載，避免阻塞啟動
_ml_ready: bool = False
_risk_mgr = RiskManager(trading_config)
_leverage_engine = LeverageEngine()  # 動態槓桿引擎
_risk_checks_passed: int = 0
_risk_checks_failed: int = 0

_current_prices: dict = {}
_previous_prices: dict = {}
_iteration_count: int = 0
_engine_running: bool = False
_engine_thread: threading.Thread = None

# 是否已開盤（開盤後才開始交易）
_market_opened: bool = False
# 是否已建倉
_positions_initialized: bool = False

# ═══════════════════════════════════════════════════════════════
# 數據延遲與精度監控
# ═══════════════════════════════════════════════════════════════
_last_price_update: str = ''       # 最後一次價格更新時間
_data_source: str = 'Yahoo Finance (啟動中)'    # 數據源: simulated / yahoo / akshare
_data_latency_ms: float = 0.0      # 數據延遲(毫秒)
_price_update_count: int = 0       # 價格更新總次數
_price_is_stale: bool = True       # 價格數據是否過期
_price_data_age_s: float = 999.0  # 數據已存在秒數(上次更新距今)
# ═══════════════════════════════════════════════════════════
# K線緩存
# ═══════════════════════════════════════════════════════════
_kline_cache: dict = {}         # {ticker: [{time,open,high,low,close,volume}]}
_last_kline_fetch: float = 0.0  # 上次K線抓取時間
_kline_fetch_interval: int = 300  # 每5分鐘更新一次K線

# ═══════════════════════════════════════════════════════════════
# 交易指令追蹤
# ═══════════════════════════════════════════════════════════════
_recent_signals: list = []         # 最近交易指令 [{time, action, ticker, qty, price, reason}]
_prediction_iters: dict = {}       # {prediction_id: iteration} 用於超時確認

TRACKED_TICKERS = [
    # 科技七巨頭
    'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'META', 'TSLA',
    # 軟體/SaaS
    'NFLX', 'ADBE', 'CRM', 'NOW', 'ORCL',
    # 金融
    'JPM', 'V', 'MA', 'BAC',
    # 消費
    'WMT', 'HD', 'NKE', 'SBUX', 'UBER',
    # 晶片/半導體
    'AVGO', 'AMD', 'INTC', 'QCOM', 'TXN',
    # 光模塊/光通信 (高波動)
    'AAOI', 'COHR', 'LITE', 'FN',
    # 存儲 (高波動)
    'WDC', 'STX', 'NTAP',
    # 數據中心晶片 (高波動)
    'MRVL', 'MU',
    # 半導體設備 (高波動)
    'LRCX', 'AMAT', 'KLAC',
    # EDA軟體 (高波動)
    'SNPS', 'CDNS',
]

# ═══════════════════════════════════════════════════════════════
# Yahoo Finance 實時數據抓取
# ═══════════════════════════════════════════════════════════════
_yahoo_fetch_interval: int = 30      # 基礎抓取間隔（秒）

# 按市場時段分層抓取間隔（優化版）
YAHOO_INTERVALS = {
    'REGULAR': 12,       # 正常交易：12秒（快速刷新）
    'PRE_MARKET': 60,    # 盤前：1分鐘
    'AFTER_HOURS': 60,   # 盤後：1分鐘
    'CLOSED': 60,        # 閉市：1分鐘
}
_last_yahoo_fetch: float = 0.0       # 上次抓取時間戳
_first_yahoo_fetch_done: bool = False  # 啟動後首次抓取標誌
_yahoo_fetch_success: bool = False   # 上次抓取是否成功
_yahoo_error_count: int = 0          # 連續失敗次數
_yahoo_session = None               # 持久HTTP會話（v7 API用）
_yahoo_crumb: str = None            # Yahoo API crumb
_yahoo_crumb_ts: float = 0.0        # crumb獲取時間戳
_v7_active: bool = False            # v7 API是否可用


def fetch_kline_data(ticker: str, period: str = "1d", interval: str = "5m"):
    """
    獲取單只股票的K線數據
    
    參數:
        ticker: 股票代碼
        period: 時間範圍 (1d/5d/1mo/3mo)
        interval: K線周期 (1m/5m/15m/30m/1h/1d)
    
    返回:
        [{time, open, high, low, close, volume}, ...] 或空列表
    """

    
    cache_key = f"{ticker}_{period}_{interval}"
    
    # 檢查緩存
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
        logger.debug(f"K線數據獲取失敗 {ticker}: {e}")
        return []

def fetch_yahoo_prices():
    """從 Yahoo Finance 抓取實時價格（v7批量API，單次請求獲取全部股票）"""
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

    # ── 方案1: v7 批量報價API（單請求，<0.2s） ──
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

    # ── 方案2: fast_info 並行回退（6 workers, 8s超時） ──
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
    螞蟻銀行澳門 美股佣金計算
    
    費用組成:
    - 佣金: 0.05% × 成交金額, 最低 USD 2.00
    - SEC費(僅賣出): 0.0008% × 成交金額
    - TAF費(僅賣出): USD 0.00013 × 股數
    
    參數:
        trade_value: 成交金額 (price × quantity)
        quantity: 股數
        side: 'BUY' 或 'SELL'
    
    返回:
        總佣金(USD)
    """
    # 基礎佣金: 0.05%, 最低$2.00
    base_commission = max(trade_value * 0.0005, 2.00)
    
    total = base_commission
    
    if side == 'SELL':
        # SEC費用: 0.0008% × 成交金額
        sec_fee = max(trade_value * 0.000008, 0.01)
        # TAF費用: $0.00013 × 股數
        taf_fee = max(quantity * 0.00013, 0.01)
        total += sec_fee + taf_fee
    
    return round(total, 2)


# ═══════════════════════════════════════════════════════════════
# 激進短線交易參數
# ═══════════════════════════════════════════════════════════════
TRADE_CHECK_INTERVAL = 60      # 每60秒檢查一次交易信號(原300秒)

# Yahoo Finance 抓取配置

PROFIT_TAKE_THRESHOLD = 0.03   # 止盈閾值3%(原10%)
STOP_LOSS_THRESHOLD = -0.04    # 止損閾值-4%(原-8%)
MAX_POSITION_HOLD_TIME = 30    # 最大持倉時間(分鐘),超時強制平倉
REENTRY_COOLDOWN = 5           # 賣出後冷卻時間(分鐘)
POSITION_MAX_PCT = 0.08        # 單只股票最大倉位8%
PREDICTIVE_SELL_THRESHOLD = 0.55  # 模型預測準確率>55%時觸發預測賣出

# Leverage Trading Parameters
MAX_LEVERAGE = 2.0
LEVERAGE_TIERS = {0.55: 1.0, 0.60: 1.25, 0.65: 1.5, 0.70: 1.75, 0.75: 2.0}
MAX_POSITION_PCT_LEVERAGED = 0.12
LEVERAGE_STOP_LOSS = -0.025
DRAWDOWN_DELEVERAGE = 0.10

# 持倉計時器: {ticker: buy_iteration}
_position_entry_time: dict = {}
# 賣出冷卻: {ticker: sell_iteration}
_position_sell_cooldown: dict = {}

# 狀態持久化: 啟動時嘗試恢復上次保存的狀態
_saved_state = load_state()
if _saved_state:
    try:
        deserialize_portfolio(_portfolio, _saved_state['portfolio'])
        deserialize_accuracy(_accuracy, _saved_state['accuracy'])
        deserialize_benchmark(_benchmark, _saved_state['benchmark'])
        # 恢復統計預測器狀態
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
        # 恢復持倉計時器和冷卻
        _position_entry_time.update(g.get('position_entry_time', {}))
        # 恢復預測迭代映射
        _prediction_iters.update(g.get('prediction_iters', {}))
        _position_sell_cooldown.update(g.get('position_sell_cooldown', {}))
        # 恢復槓桿引擎狀態
        if g.get('leverage_engine'):
            _leverage_engine = LeverageEngine.from_dict(g['leverage_engine'])
        # 恢復交易信號
        sigs = g.get('recent_signals', [])
        if sigs:
            _recent_signals.extend(sigs)
        saved_at = _saved_state.get('saved_at', '?')
        print(f'  ✅ 已恢復交易狀態 (保存於 {saved_at})')
        print(f'     現金: ${_portfolio.cash:,.2f} | 持倉: {len(_portfolio.positions)}只')
        print(f'     預測: {len(_accuracy.predictions)}條 | 交易: {len(_portfolio.trade_history)}筆')
        print(f'     迭代: {_iteration_count} | 建倉: {_positions_initialized}')
    except Exception as e:
        print(f'  ⚠️ 狀態恢復失敗: {e}，使用全新狀態')
else:
    print('  🆕 使用全新交易狀態')

# ═══════════════════════════════════════════════════════════════
# 基準初始化：拉取納指歷史數據用於曲線對比
# ═══════════════════════════════════════════════════════════════
if _benchmark.nasdaq_equity_curve.empty:
    print('  📊 拉取納指歷史數據用於基準曲線...')
    ok = _benchmark.fetch_nasdaq_history('6mo')
    if ok:
        print(f'  ✅ 納指基準已初始化 ({len(_benchmark.nasdaq_equity_curve)} 數據點)')
    else:
        print('  ⚠️ 納指歷史數據拉取失敗，基準曲線將在交易開始後逐步建立')

# 初始建倉配置
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
    # 光模塊/光通信
    'AAOI': {'qty': 30, 'cost': 15},
    'COHR': {'qty': 8, 'cost': 100},
    'LITE': {'qty': 15, 'cost': 85},
    'FN':   {'qty': 7, 'cost': 220},
    # 存儲
    'WDC':  {'qty': 10, 'cost': 70},
    'STX':  {'qty': 11, 'cost': 105},
    'NTAP': {'qty': 9, 'cost': 120},
    # 數據中心晶片
    'MRVL': {'qty': 13, 'cost': 110},
    'MU':   {'qty': 10, 'cost': 140},
    # 半導體設備
    'LRCX': {'qty': 6, 'cost': 95},
    'AMAT': {'qty': 8, 'cost': 230},
    'KLAC': {'qty': 5, 'cost': 800},
    # EDA軟體
    'SNPS': {'qty': 5, 'cost': 560},
    'CDNS': {'qty': 5, 'cost': 310},
}


def init_positions():
    """初始化建倉 - 使用Yahoo實時價格作為成本價"""
    global _positions_initialized, _portfolio, _current_prices, _data_source, _last_price_update
    
    # 已有持倉(從持久化恢復): 跳過建倉
    if len(_portfolio.positions) > 0:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ 恢復持倉: {len(_portfolio.positions)}只, 跳過建倉")
        for ticker in list(_portfolio.positions.keys()):
            if ticker not in _current_prices:
                _current_prices[ticker] = _portfolio.positions[ticker].avg_cost
        if '^IXIC' not in _current_prices:
            _current_prices['^IXIC'] = 18200.0
        _positions_initialized = True
        return
    
    # 先從Yahoo獲取實時價格作為建倉成本（批量下載避免限流）
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 📡 獲取Yahoo實時價格用於建倉...", flush=True)
    
    real_prices = {}
    
    # 方式1: 嘗試批量下載（單次請求）
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
                print(f"  批量獲取: {len(real_prices)}/{len(batch_tickers)} 只", flush=True)
    except Exception as e:
        print(f"  批量下載失敗: {e}", flush=True)
    
    # 方式2: 對失敗的逐個重試
    for ticker in INITIAL_POSITIONS:
        if ticker in real_prices:
            continue
        try:
            tk = yf.Ticker(ticker)
            # 多數據源fallback
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
    
    # 獲取納指價格
    try:
        ix = yf.Ticker('^IXIC')
        ix_p = ix.fast_info.get('lastPrice') or ix.fast_info.get('regularMarketPrice') or 18000
        if ix_p:
            _current_prices['^IXIC'] = float(ix_p)
    except Exception:
        _current_prices['^IXIC'] = 18000.0
    
    if len(real_prices) < 5:
        # Yahoo不可用: 回退到預設價格
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚠️ Yahoo價格不足, 使用預設價格")
        for ticker, info in INITIAL_POSITIONS.items():
            real_prices[ticker] = info['cost']
        _data_source = 'fallback (預設價格)'
    else:
        _data_source = 'Yahoo Finance (實時)'
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✓ 獲取到 {len(real_prices)} 只實時價格")
    
    # 用真實價格建倉, 調整數量以匹配初始資金分配
    # 總資金80%用於建倉，20%留作短線交易現金
    trading_cash = 100000.0 * 0.20
    investable = 100000.0 - trading_cash
    cash_per_stock = investable / len(INITIAL_POSITIONS)  # 每隻約$10,000
    total_commission = 0
    
    for ticker, info in INITIAL_POSITIONS.items():
        price = real_prices.get(ticker, info['cost'])
        qty = max(1, int(cash_per_stock / price))  # 按價格計算股數
        trade_val = qty * price
        comm = calculate_commission(trade_val, qty, 'BUY')
        total_commission += comm
        
        _portfolio.execute_buy(ticker, qty, price, commission=comm, reason='開盤建倉')
        _current_prices[ticker] = price
        _recent_signals.append({
            'time': datetime.now().strftime('%H:%M:%S'),
            'action': 'BUY', 'ticker': ticker,
            'qty': qty, 'price': round(price, 2),
            'reason': '開盤建倉(Yahoo價格)',
        })
        print(f"  {ticker}: {qty}股 @ ${price:.2f} (佣金${comm:.2f})", flush=True)
    
    _last_price_update = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    if '^IXIC' not in _current_prices:
        _current_prices['^IXIC'] = 18000.0
    
    # 設置納指基準（如果 fetch_nasdaq_history 還未設置）
    if _benchmark.nasdaq_start_price == 0:
        nasdaq_price = _current_prices.get('^IXIC', 18000.0)
        _benchmark.nasdaq_start_price = nasdaq_price
        _benchmark.nasdaq_shares = 100000.0 / nasdaq_price if nasdaq_price > 0 else 5.5
    
    _positions_initialized = True
    total_invested = 100000.0 - _portfolio.cash - total_commission
    print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ 建倉完成: {len(_portfolio.positions)}只, " +
          f"投入${total_invested:,.0f}, 佣金${total_commission:.2f}", flush=True)


def tick_engine():
    """引擎一個tick（模擬價格波動+交易決策）"""
    global _current_prices, _previous_prices, _iteration_count
    global _market_opened, _positions_initialized, _recent_signals
    
    _iteration_count += 1
    
    # 檢查市場是否開盤
    status, _ = _clock.get_status()
    is_open_now = (status == MarketStatus.REGULAR_HOURS)
    # 是否為正常交易時段（僅此時段允許買賣交易）
    is_trading_session = is_open_now
    
    # 開盤瞬間執行建倉
    if is_trading_session and not _positions_initialized:
        _market_opened = True
        init_positions()
    
    # ═══════════════════════════════════════════════════════════
    # ═══════════════════════════════════════════════════════════
    # 價格更新：始終嘗試獲取Yahoo價格（即使未建倉），確保數據新鮮度
    # ═══════════════════════════════════════════════════════════
    global _last_price_update, _price_update_count, _data_latency_ms
    global _price_is_stale, _data_source, _price_data_age_s
    _previous_prices = dict(_current_prices)
    
    # 嘗試從 Yahoo Finance 抓取真實價格
    # 啟動後首次無條件嘗試（_last_yahoo_fetch=0 保證首次通過）
    yahoo_prices, yahoo_latency = fetch_yahoo_prices()
    
    if yahoo_prices:
        # 使用Yahoo真實價格
        for ticker, price in yahoo_prices.items():
            _current_prices[ticker] = price
        _data_latency_ms = round(yahoo_latency, 2)
        _data_source = 'Yahoo Finance (實時)'
        _last_price_update = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        _price_update_count += 1
        _price_is_stale = False; _data_is_stale = False
        _price_data_age_s = 0  # 數據新鮮
    else:
        # Yahoo未返回新數據：可能是間隔未到，或請求失敗
        # 更新數據年齡（距上次更新秒數）
        if _last_price_update:
            try:
                last_dt = datetime.strptime(_last_price_update[:19], '%Y-%m-%d %H:%M:%S')
                _price_data_age_s = (datetime.now() - last_dt).total_seconds()
            except Exception:
                pass
        # 更新 _data_latency_ms 反映數據新鮮度（距上次成功更新的毫秒數）
        if _price_data_age_s > 0 and _price_data_age_s < 999:
            _data_latency_ms = _price_data_age_s * 1000
        
        # 僅當連續失敗超過閾值時才標記數據為過期
        if _yahoo_error_count > 3:
            _price_is_stale = True; _data_is_stale = True
            _data_source = f'離線 (Yahoo不可用, 數據已過期){" [v7回退]" if _v7_active else ""}'
        elif _yahoo_error_count > 0:
            # 有失敗但未超過閾值：保持當前狀態，來源標記為等待重連
            _price_is_stale = False  # 上次數據仍有效，不標記過期
            _data_source = f'Yahoo Finance (部分失敗, 上次數據仍有效){" [v7]" if _v7_active else ""}'
        else:
            # 完全正常：僅因間隔未到，保持現有數據
            _price_is_stale = False  # 上次數據仍然有效
            # 計算數據年齡用於前端展示
            age_s = int(_price_data_age_s) if _price_data_age_s < 999 else 0
            _data_source = f'Yahoo Finance (緩存{" [v7]" if _v7_active else ""}, {age_s}s前)'
    
    # 未開盤且未建倉：不執行交易操作（但價格已更新供前端展示）
    if not _positions_initialized:
        return
    
    # 更新持倉市值 & 餵入統計預測器
    prices_for_portfolio = {t: p for t, p in _current_prices.items() if t != '^IXIC'}
    _portfolio.update_prices(prices_for_portfolio)
    # 餵入價格到統計預測器
    for ticker, price in _current_prices.items():
        if ticker != '^IXIC':
            _predictor.update_price(ticker, price)
    
    # 更新基準
    _benchmark.update(_current_prices['^IXIC'], _portfolio.get_total_equity())
    
    # Daily interest accrual for borrowed funds
    _portfolio.accrue_interest()
    
    # ═══════════════════════════════════════════════════════════
    # 交易會話門控：僅在正常交易時段執行買賣
    # ═══════════════════════════════════════════════════════════
    if not is_trading_session:
        return  # 休市時不執行任何交易，僅更新價格供展示
    
    # 價格數據過期時跳過交易決策（避免基於偽造數據交易）
    if _price_is_stale:
        return
    
    # 激進短線交易邏輯 (每60秒檢查)
    if _iteration_count % TRADE_CHECK_INTERVAL == 0 and _portfolio.cash > 1000:
        current_equity = _portfolio.get_total_equity()
        
        for ticker, pos in list(_portfolio.positions.items()):
            if ticker not in _current_prices:
                continue
            
            price = _current_prices[ticker]
            pnl_pct = pos.unrealized_pnl_pct
            hold_minutes = (_iteration_count - _position_entry_time.get(ticker, _iteration_count)) / 60
            
            # 使用已更新的持倉數據
            
            should_sell = False
            sell_reason = ''
            
            # 條件1: 止盈 (盈利≥3%時80%概率賣出,鎖定利潤)
            # 使用確定性哈希避免隨機數濫用，同時保持概率分散效果
            _tp_seed = int(hashlib.md5(f'{_iteration_count}:{ticker}:tp'.encode()).hexdigest()[:8], 16)
            if pnl_pct >= PROFIT_TAKE_THRESHOLD and (_tp_seed % 100) < 80:
                should_sell = True
                sell_reason = f'短線止盈({pnl_pct:.1%})'
            
            # 條件2: 止損 (槓桿-2.5%, 普通-4%)
            effective_sl = LEVERAGE_STOP_LOSS if _portfolio.get_leverage_ratio() > 1.2 else STOP_LOSS_THRESHOLD
            if pnl_pct <= effective_sl:
                _sl_seed = int(hashlib.md5(f"{_iteration_count}:{ticker}:sl".encode()).hexdigest()[:8], 16)
                if (_sl_seed % 100) < 90:
                    should_sell = True
                    lev_tag = "[L]" if _portfolio.get_leverage_ratio() > 1.2 else ""
                    sell_reason = f"止損({pnl_pct:.1%}){lev_tag}"
            
            # 條件3: 預測性賣出 (模型預測下跌時50%概率減倉)
            if not should_sell and pnl_pct < 0.01:  # 微利或微虧時更積極
                # 使用ML模型或統計預測器判斷是否賣出
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
                    sell_reason = f'ML預測賣出'
            
            # 條件4: 持倉超時強制平倉 (獨立檢查,不受上麵條件阻塞)
            if not should_sell and hold_minutes > MAX_POSITION_HOLD_TIME:
                should_sell = True
                sell_reason = f'超時平倉({hold_minutes:.0f}分鐘)'
            
            if should_sell:
                trade_val = pos.quantity * price
                comm = calculate_commission(trade_val, pos.quantity, 'SELL')
                _portfolio.execute_sell(ticker, pos.quantity, price,
                                        commission=comm, reason=sell_reason)
                # 記錄交易到槓桿引擎(績效反饋)
                _leverage_engine.record_trade(
                    win=(pos.unrealized_pnl > 0),
                    pnl_pct=pos.unrealized_pnl_pct
                )
                _position_sell_cooldown[ticker] = _iteration_count
                # 記錄交易指令
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
                break  # 每輪最多一筆賣出
        
        # 買入邏輯: 在賣出後或現金充裕時尋找新機會
        if _portfolio.cash > 5000:
            # 獲取不在冷卻期且未持倉的候選股票
            candidates = []
            for t in TRACKED_TICKERS:
                if t in _portfolio.positions:
                    continue
                cooldown_iter = _position_sell_cooldown.get(t, 0)
                if (_iteration_count - cooldown_iter) < REENTRY_COOLDOWN * 60:
                    continue  # 還在冷卻期
                candidates.append(t)
            
            if candidates:
                # 根據"ML信號"選擇最佳候選 (模擬:偏向於選擇價格波動的)
                best_ticker = None
                best_signal = -1
                
                for t in candidates:
                    price = _current_prices.get(t, 0)
                    if price <= 0:
                        continue
                    
                    # 優先使用Transformer模型，降級到統計預測器
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
                    
                    # ── 動態槓桿引擎: 凱利公式+波動率+績效+熱度 ──
                    leverage, lev_detail = _leverage_engine.calculate(
                        confidence=best_signal if best_signal > 0.5 else 0.52,
                        ticker=best_ticker,
                        current_prices=_current_prices,
                        prev_prices=_previous_prices,
                        portfolio=_portfolio,
                        accuracy=_accuracy,
                    )
                    # 如果槓桿低於0.5x, 機會不夠好, 跳過
                    if leverage < 0.5:
                        best_ticker = None  # 放棄這次買入
                    
                    max_pct = MAX_POSITION_PCT_LEVERAGED if leverage > 1.0 else POSITION_MAX_PCT
                    max_position_value = current_equity * max_pct
                    available = _portfolio.get_available_cash()
                    max_qty_by_cash = int(available * 0.20 / price)
                    max_qty_by_limit = int(max_position_value / price)
                    qty = min(max_qty_by_cash, max_qty_by_limit, 200 if leverage > 1.0 else 100)
                    
                    if qty > 0:
                        # 風控檢查
                        if not _risk_mgr.in_trade.is_paused:
                            trade_val = qty * price
                            comm = calculate_commission(trade_val, qty, 'BUY')
                            buy_reason = f'ML{leverage}x' if leverage > 1.0 else 'ML短線買入'
                            _portfolio.execute_buy(best_ticker, qty, price,
                                                   commission=comm, reason=buy_reason)
                            _position_entry_time[best_ticker] = _iteration_count
                        # 記錄交易指令
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
    # 真實統計預測（每30次約30秒生成一次預測）
    # 使用多因子模型: 動量+均值回歸+成交量+波動率
    # ═══════════════════════════════════════════════════════════
    if _iteration_count % 30 == 0:
        # 只從有價格數據的股票中選取
        available_tickers = [t for t in TRACKED_TICKERS if t in _current_prices and _current_prices[t] > 0]
        if not available_tickers:
            available_tickers = list(TRACKED_TICKERS)
        ticker = available_tickers[_iteration_count % len(available_tickers)]
        price = _current_prices.get(ticker, 0)
        if price > 0:
            # 使用統計預測器生成真實信號
            direction, confidence, factors = _predictor.predict(ticker, price)
            predicted_return = (price / _previous_prices.get(ticker, price) - 1) if ticker in _previous_prices else 0.001
            
            pred_id = _accuracy.record_prediction(
                ticker, predicted_return, direction, confidence
            )
            _prediction_iters[pred_id] = _iteration_count
            
            # 確認該ticker的所有舊預測（已過30秒）
            # 查找30秒前同一ticker的未確認預測來確認
            candidates_to_confirm = []
            for pid, pr in _accuracy.predictions.items():
                if pr.ticker == ticker and pr.status == 'pending' and pr.id < pred_id:
                    candidates_to_confirm.append((pid, pr))
            
            if candidates_to_confirm:
                # 確認最早的預測
                candidates_to_confirm.sort(key=lambda x: x[0])
                oldest_pid, oldest_pr = candidates_to_confirm[0]
                old_price = _previous_prices.get(ticker, price)
                actual_return = (price / old_price - 1) if old_price > 0 else 0
                actual_direction = 1 if actual_return > 0 else 0
                _accuracy.confirm_prediction(oldest_pid, actual_return, actual_direction)
            
            # 額外：確認超過2分鐘(120迭代)的未確認預測，使用當前價
            for pid, pr in list(_accuracy.predictions.items()):
                if pr.status == 'pending' and (_iteration_count - _prediction_iters.get(pid, 0)) > 120:
                    actual_return = (price / _previous_prices.get(pr.ticker, price) - 1) if pr.ticker in _previous_prices else 0
                    actual_direction = 1 if actual_return > 0 else 0
                    _accuracy.confirm_prediction(pid, actual_return, actual_direction)
    
    # 超時確認：每60秒確認超過90秒未確認的預測
    if _iteration_count % 60 == 0:
        for pid, pr in list(_accuracy.predictions.items()):
            if pr.status == 'pending':
                # 用預測時的迭代數判斷是否超時（90秒 ≈ 90迭代）
                pred_iter = _prediction_iters.get(pid, 0)
                if (_iteration_count - pred_iter) > 90:
                    ticker = pr.ticker
                    current_p = _current_prices.get(ticker, 0)
                    if current_p > 0 and ticker in _previous_prices:
                        actual_return = (current_p / _previous_prices[ticker] - 1)
                        actual_direction = 1 if actual_return > 0 else 0
                        _accuracy.confirm_prediction(pid, actual_return, actual_direction)



def _collect_globals_dict():
    """收集全局運行時狀態，避免重複代碼"""
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
    """後臺加載ML模型（異步，不阻塞啟動）"""
    global _ml_inference, _ml_ready
    try:
        _ml_inference = ModelInference()
        if _ml_inference.load():
            _ml_ready = True
            print(f'[{datetime.now().strftime("%H:%M:%S")}] 🧠 Transformer模型加載成功', flush=True)
            return True
        else:
            print(f'[{datetime.now().strftime("%H:%M:%S")}] ⚠️ Transformer模型文件不存在，使用統計預測器', flush=True)
    except Exception as e:
        print(f'[{datetime.now().strftime("%H:%M:%S")}] ⚠️ ML模型加載失敗: {e}', flush=True)
    return False

def engine_loop():
    """後臺引擎循環（每秒1次）"""
    global _engine_running, _iteration_count, _ml_ready
    _offline_training_done = False
    _ml_model_loaded = _ml_ready  # 從持久化狀態恢復
    
    def trigger_offline_training():
        """在休市時離線訓練Transformer"""
        nonlocal _offline_training_done
        if _offline_training_done:
            return
        status, _ = _clock.get_status()
        if status != MarketStatus.REGULAR_HOURS and _iteration_count > 300:
            _offline_training_done = True  # 防止重複觸發
            print(f'[{datetime.now().strftime("%H:%M:%S")}] 🧠 休市中，啟動離線Transformer訓練（後臺）...', flush=True)
            # 在後臺線程中訓練，避免阻塞引擎循環
            def _train_thread():
                try:
                    from live_trading.predictor import train_offline_transformer
                    acc = train_offline_transformer()
                    if acc:
                        print(f'[{datetime.now().strftime("%H:%M:%S")}] ✅ 離線訓練完成，準確率: {acc:.2%}', flush=True)
                except Exception as e:
                    print(f'[{datetime.now().strftime("%H:%M:%S")}] ⚠️ 離線訓練失敗: {e}', flush=True)
            threading.Thread(target=_train_thread, daemon=True, name='offline-train').start()
    
    # 啟動時立即加載ML模型
    print(f'[{datetime.now().strftime("%H:%M:%S")}] 🔄 加載ML模型...', flush=True)
    if init_ml_model():
        _ml_model_loaded = True
        _ml_ready = True
        print(f'[{datetime.now().strftime("%H:%M:%S")}] ✅ ML模型就緒', flush=True)
    else:
        print(f'[{datetime.now().strftime("%H:%M:%S")}] ⚠️ ML模型未加載', flush=True)
    
    while _engine_running:
        try:
            # 延遲加載ML模型（每60秒重試一次，避免頻繁創建對象）
            if not _ml_model_loaded and _iteration_count % 60 == 0:
                if init_ml_model():
                    _ml_model_loaded = True
                    _ml_ready = True  # 同步全局狀態（已在函數頂部聲明global）
            tick_engine()
            # 休市時每10分鐘檢查是否需要離線訓練
            if _iteration_count % 600 == 0:
                trigger_offline_training()
            # 每60次迭代（約60秒）自動保存狀態
            if _iteration_count > 0 and _iteration_count % 60 == 0:
                save_state(_portfolio, _accuracy, _benchmark, _collect_globals_dict())
        except Exception as e:
            print(f"引擎異常: {e}", flush=True)
        time.sleep(1)
    
    # 引擎停止時保存最終狀態（包含全部運行時狀態）
    save_state(_portfolio, _accuracy, _benchmark, _collect_globals_dict())
    print('💾 最終狀態已保存', flush=True)


# ============================================================================
# API 路由
# ============================================================================

@app.route('/api/health')
def api_health():
    """健康檢查端點（供watchdog使用）"""
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
    """返回所有追蹤的股票代碼列表"""
    return jsonify({
        'tickers': list(TRACKED_TICKERS),
        'count': len(TRACKED_TICKERS),
    })

@app.route('/api/signals')
def api_signals():
    """交易指令端點 - 返回Markdown格式的交易信號"""
    if not _recent_signals:
        return jsonify({'signals_md': '*(暫無交易指令)*', 'signals': []})
    
    lines = []
    for s in _recent_signals[-20:]:  # 最近20條
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
    """構建狀態數據字典（供API和WebSocket共用）"""
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
    
    _benchmark._ensure_curves_synced()  # 確保 API 讀取時曲線已同步
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
            countdown = f"距閉市 {cd[0]}h {cd[1]:02d}m {cd[2]:02d}s"
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
    """K線數據API"""
    period = request.args.get('period', '1d')
    interval = request.args.get('interval', '5m')
    candles = fetch_kline_data(ticker.upper(), period, interval)
    return jsonify({'ticker': ticker.upper(), 'candles': candles, 'count': len(candles)})

@app.route('/api/kline/multi')
def api_kline_multi():
    """批量K線數據API"""
    tickers_str = request.args.get('tickers', '')
    period = request.args.get('period', '1d')
    interval = request.args.get('interval', '15m')
    tickers = [t.strip().upper() for t in tickers_str.split(',') if t.strip()]
    if not tickers:
        tickers = ['AAPL', 'NVDA', 'MSFT', 'GOOGL']
    
    result = {}
    for t in tickers[:6]:  # 最多6隻
        candles = fetch_kline_data(t, period, interval)
        if candles:
            result[t] = candles
    
    return jsonify({'data': result, 'count': len(result)})

@app.route('/api/status')
def api_status():
    """獲取完整系統狀態"""
    return jsonify(build_status_data())

def start_server(host: str = '0.0.0.0', port: int = 8080, debug: bool = False):
    """啟動Web伺服器"""
    global _engine_running, _engine_thread
    
    # 不在此處初始化持倉 —— 等開盤後由 tick_engine 執行
    
    _engine_running = True
    _engine_thread = threading.Thread(target=engine_loop, daemon=True)
    _engine_thread.start()
    
    # 檢查當前市場狀態
    status, desc = _clock.get_status()
    if status == MarketStatus.REGULAR_HOURS:
        print(f"\n  🟢 當前市場已開盤，立即開始交易!")
    else:
        cd = _clock.countdown_to_next_open()
        print(f"\n  ⏳ 等待開盤... 距離開市還有約 {cd[0]}h {cd[1]:02d}m {cd[2]:02d}s")
        print(f"  系統將在美東 09:30 自動建倉並開始交易")
    
    print(f"\n{'='*60}")
    print(f"  美股量化交易系統 - Web儀錶盤")
    print(f"  地址: http://localhost:{port}")
    print(f"  刷新頻率: 每秒")
    print(f"  按 Ctrl+C 停止")
    print(f"{'='*60}\n", flush=True)
    
    app.run(host=host, port=port, debug=debug, use_reloader=False, threaded=True)



@app.route('/api/benchmark_curve')
def api_benchmark_curve():
    """策略 vs 納指 權益曲線數據（返回百分比收益率）"""
    try:
        _benchmark._ensure_curves_synced()
        nasdaq_curve = _benchmark.nasdaq_equity_curve
        strategy_curve = _benchmark.strategy_equity_curve
        initial = _benchmark.initial_capital
        
        # 合併兩個 Series 的時間戳，取最新的300個（避免全量排序）
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
    """回測結果摘要"""
    try:
        from backtesting.performance import BacktestPerformance
        from config.settings import PROCESSED_DATA_DIR
        # 嘗試加載已有回測結果或返回空
        return jsonify({
            'available': False,
            'message': '運行 python main.py backtest 生成回測結果',
            'summary': {}
        })
    except Exception as e:
        return jsonify({'available': False, 'message': str(e)})


def _shutdown_handler(signum=None, frame=None):
    """進程退出時保存狀態"""
    global _engine_running
    print('\n🛑 正在關閉...', flush=True)
    _engine_running = False
    if _engine_thread and _engine_thread.is_alive():
        _engine_thread.join(timeout=3)
    # 保存狀態
    save_state(_portfolio, _accuracy, _benchmark, _collect_globals_dict())
    print('💾 狀態已保存，安全退出', flush=True)
    sys.exit(0)

signal.signal(signal.SIGTERM, _shutdown_handler)
signal.signal(signal.SIGINT, _shutdown_handler)


if __name__ == '__main__':
    start_server(debug=False)
