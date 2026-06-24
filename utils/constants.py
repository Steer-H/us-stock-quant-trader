"""Trading and model constants."""

from datetime import time, date
from typing import Dict, List, Tuple
from dataclasses import dataclass, field


# ============================================================================
# 交易時段常量（美東時間 ET）
# ============================================================================
@dataclass(frozen=True)
class MarketHours:
    """
    美股交易時段定義（美東時間）
    
    時段劃分：
    - 盤前(Pre-Market): 04:00 - 09:30 ET  (流動性較低，波動較大)
    - 正常盤(Regular):  09:30 - 16:00 ET  (主力交易時段)
    - 盤後(After-Hours): 16:00 - 20:00 ET (流動性較低，波動較大)
    """
    PRE_MARKET_START: time = time(4, 0)    # 盤前開始
    REGULAR_START: time = time(9, 30)      # 正常盤開始
    REGULAR_END: time = time(16, 0)        # 正常盤結束
    AFTER_HOURS_END: time = time(20, 0)    # 盤後結束
    
    # 美股主要交易所
    PRIMARY_EXCHANGES: Tuple[str, ...] = (
        'NYSE', 'NASDAQ', 'ARCA', 'BATS', 'IEX', 'EDGX'
    )
    
    # 暗池（Dark Pools）標識
    DARK_POOL_VENUES: Tuple[str, ...] = (
        'DRCTEDGE', 'CROSSFINDER', 'SIGMAX2', 'LEVEL'
    )


# ============================================================================
# SEC 監管規則常量
# ============================================================================
@dataclass(frozen=True)
class RegSHO:
    """
    Regulation SHO - 做空規則相關常量
    
    Reg SHO (Regulation of Short Sales) 是SEC對做空行為的主要監管框架，
    包含以下核心規則：
    - Rule 201 (Uptick Rule/Alternative Uptick Rule):
      當某股票日內跌幅超過前收盤價10%時，觸發斷路器，
      剩餘交易日及下一個交易日內，做空只能在價格高於當前全國最優報價(ask)時執行。
    - Locate Requirement (Rule 203(b)(1)):
      做空前必須確認可以借到股票（locate），禁止裸賣空(naked short selling)。
    """
    # Uptick Rule觸發閾值（跌幅百分比）
    CIRCUIT_BREAKER_THRESHOLD: float = 0.10  # 10%
    
    # 斷路器有效期限（交易日）
    CIRCUIT_BREAKER_DURATION_DAYS: int = 1
    
    # 做空借券確認有效期（Reg SHO Rule 204: T+3結算）
    LOCATE_VALIDITY_DAYS: int = 3


@dataclass(frozen=True)
class WashSaleRule:
    """
    洗售規則 (Wash Sale Rule) 相關常量
    
    IRS規定：如果在賣出虧損股票的前後30天內（共61天窗口）
    買入"實質上相同"(substantially identical)的證券，
    則該虧損不可抵稅，需調整新買入股票的成本基礎。
    """
    # 洗售窗口（前後各30天，共61天）
    WINDOW_DAYS: int = 30
    
    # 完整的洗售檢查窗口大小
    FULL_WINDOW_DAYS: int = 61


@dataclass(frozen=True)
class PDT_RULES:
    """
    Pattern Day Trader 規則相關常量
    
    FINRA規定：任何在5個交易日內執行4次或以上日內交易
    （同日開倉並平倉同一證券）的帳戶，將被標記為PDT。
    PDT帳戶必須維持至少$25,000的帳戶淨值（現金+證券市值），
    否則將受到交易限制。
    """
    # PDT標記閾值：5個交易日內N次日內交易
    DAY_TRADE_COUNT: int = 4
    # PDT規則的滾動窗口（交易日）
    ROLLING_WINDOW: int = 5
    # PDT最低帳戶資金要求
    MIN_EQUITY: float = 25_000.0


# ============================================================================
# 熔斷機制常量 (LULD - Limit Up-Limit Down)
# ============================================================================
@dataclass(frozen=True)
class LULD:
    """
    LULD (Limit Up-Limit Down) 熔斷機制
    
    防止個股價格在短時間內出現極端波動。
    根據股價和交易量將股票分為Tier 1和Tier 2，
    不同層級的漲跌停幅度不同。
    """
    # Tier 1股票（S&P 500, Russell 1000, 部分ETF）
    TIER1_BANDS: Dict[str, float] = field(default_factory=lambda: {
        'open':    0.05, 
        '09:45':   0.05,
        '15:30':   0.05, 
        'regular': 0.10,
    })
    
    # Tier 2股票（其餘NMS股票）
    TIER2_BANDS: Dict[str, float] = field(default_factory=lambda: {
        'open':    0.10, 
        '09:45':   0.10,
        '15:30':   0.10, 
        'regular': 0.20,
    })
    
    # 全市場熔斷閾值（標普500指數）
    MARKET_CIRCUIT_BREAKERS: Dict[int, str] = field(default_factory=lambda: {
        -7:  'Level 1: 交易暫停15分鐘',
        -13: 'Level 2: 交易暫停15分鐘', 
        -20: 'Level 3: 當日剩餘時間停止交易',
    })


# ============================================================================
# 美股交易所假期（2025-2027）
# 非完整列表，實際應接入交易所API或定期更新
# ============================================================================
EXCHANGE_HOLIDAYS: Dict[int, List[date]] = {
    2025: [
        date(2025, 1, 1),    # 元旦 New Year's Day
        date(2025, 1, 20),   # 馬丁·路德·金紀念日
        date(2025, 2, 17),   # 總統日 Presidents' Day
        date(2025, 4, 18),   # 耶穌受難日 Good Friday
        date(2025, 5, 26),   # 陣亡將士紀念日 Memorial Day
        date(2025, 6, 19),   # 六月節 Juneteenth
        date(2025, 7, 4),    # 獨立日 Independence Day
        date(2025, 9, 1),    # 勞動節 Labor Day
        date(2025, 11, 27),  # 感恩節 Thanksgiving
        date(2025, 12, 25),  # 聖誕節 Christmas
    ],
    2026: [
        date(2026, 1, 1), date(2026, 1, 19), date(2026, 2, 16),
        date(2026, 4, 3), date(2026, 5, 25), date(2026, 6, 19),
        date(2026, 7, 3), date(2026, 9, 7), date(2026, 11, 26), date(2026, 12, 25),
    ],
    2027: [
        date(2027, 1, 1), date(2027, 1, 18), date(2027, 2, 15),
        date(2027, 3, 26), date(2027, 5, 31), date(2027, 6, 18),
        date(2027, 7, 5), date(2027, 9, 6), date(2027, 11, 25), date(2027, 12, 24),
    ],
}


# ============================================================================
# 券商代碼映射
# ============================================================================
BROKER_CODES: Dict[str, str] = {
    'ibkr':         'Interactive Brokers',
    'alpaca':       'Alpaca Markets',
    'td_ameritrade': 'TD Ameritrade (Schwab)',
    'tradier':      'Tradier',
    'robinhood':    'Robinhood',
}


# ============================================================================
# 技術指標默認參數
# ============================================================================
TECHNICAL_INDICATORS_DEFAULTS: Dict[str, int] = {
    'sma_fast':        5,     # 快線均線周期
    'sma_medium':      20,    # 中線均線周期
    'sma_slow':        60,    # 慢線均線周期
    'ema_fast':        12,    # 快線EMA
    'ema_slow':        26,    # 慢線EMA
    'macd_signal':     9,     # MACD信號線周期
    'rsi_period':      14,    # RSI計算周期
    'bb_period':       20,    # 布林帶計算周期
    'bb_std':          2,     # 布林帶標準差倍數
    'atr_period':      14,    # ATR計算周期
    'volume_ma_period': 20,   # 成交量均線周期
}


