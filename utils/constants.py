"""
美股量化交易系统 - 常量定义模块

集中管理系统中所有硬编码的常量值，包括：
- 交易时段定义（美东时间）
- SEC监管规则常量
- 美股交易所假期日历
- 券商代码映射
- 风险阈值常量
"""

from datetime import time, date
from typing import Dict, List, Tuple
from dataclasses import dataclass, field


# ============================================================================
# 交易时段常量（美东时间 ET）
# ============================================================================
@dataclass(frozen=True)
class MarketHours:
    """
    美股交易时段定义（美东时间）
    
    时段划分：
    - 盘前(Pre-Market): 04:00 - 09:30 ET  (流动性较低，波动较大)
    - 正常盘(Regular):  09:30 - 16:00 ET  (主力交易时段)
    - 盘后(After-Hours): 16:00 - 20:00 ET (流动性较低，波动较大)
    """
    PRE_MARKET_START: time = time(4, 0)    # 盘前开始
    REGULAR_START: time = time(9, 30)      # 正常盘开始
    REGULAR_END: time = time(16, 0)        # 正常盘结束
    AFTER_HOURS_END: time = time(20, 0)    # 盘后结束
    
    # 美股主要交易所
    PRIMARY_EXCHANGES: Tuple[str, ...] = (
        'NYSE', 'NASDAQ', 'ARCA', 'BATS', 'IEX', 'EDGX'
    )
    
    # 暗池（Dark Pools）标识
    DARK_POOL_VENUES: Tuple[str, ...] = (
        'DRCTEDGE', 'CROSSFINDER', 'SIGMAX2', 'LEVEL'
    )


# ============================================================================
# SEC 监管规则常量
# ============================================================================
@dataclass(frozen=True)
class RegSHO:
    """
    Regulation SHO - 做空规则相关常量
    
    Reg SHO (Regulation of Short Sales) 是SEC对做空行为的主要监管框架，
    包含以下核心规则：
    - Rule 201 (Uptick Rule/Alternative Uptick Rule):
      当某股票日内跌幅超过前收盘价10%时，触发断路器，
      剩余交易日及下一个交易日内，做空只能在价格高于当前全国最优报价(ask)时执行。
    - Locate Requirement (Rule 203(b)(1)):
      做空前必须确认可以借到股票（locate），禁止裸卖空(naked short selling)。
    """
    # Uptick Rule触发阈值（跌幅百分比）
    CIRCUIT_BREAKER_THRESHOLD: float = 0.10  # 10%
    
    # 断路器有效期限（交易日）
    CIRCUIT_BREAKER_DURATION_DAYS: int = 1
    
    # 做空借券确认有效期（Reg SHO Rule 204: T+3结算）
    LOCATE_VALIDITY_DAYS: int = 3


@dataclass(frozen=True)
class WashSaleRule:
    """
    洗售规则 (Wash Sale Rule) 相关常量
    
    IRS规定：如果在卖出亏损股票的前后30天内（共61天窗口）
    买入"实质上相同"(substantially identical)的证券，
    则该亏损不可抵税，需调整新买入股票的成本基础。
    """
    # 洗售窗口（前后各30天，共61天）
    WINDOW_DAYS: int = 30
    
    # 完整的洗售检查窗口大小
    FULL_WINDOW_DAYS: int = 61


@dataclass(frozen=True)
class PDT_RULES:
    """
    Pattern Day Trader 规则相关常量
    
    FINRA规定：任何在5个交易日内执行4次或以上日内交易
    （同日开仓并平仓同一证券）的账户，将被标记为PDT。
    PDT账户必须维持至少$25,000的账户净值（现金+证券市值），
    否则将受到交易限制。
    """
    # PDT标记阈值：5个交易日内N次日内交易
    DAY_TRADE_COUNT: int = 4
    # PDT规则的滚动窗口（交易日）
    ROLLING_WINDOW: int = 5
    # PDT最低账户资金要求
    MIN_EQUITY: float = 25_000.0


# ============================================================================
# 熔断机制常量 (LULD - Limit Up-Limit Down)
# ============================================================================
@dataclass(frozen=True)
class LULD:
    """
    LULD (Limit Up-Limit Down) 熔断机制
    
    防止个股价格在短时间内出现极端波动。
    根据股价和交易量将股票分为Tier 1和Tier 2，
    不同层级的涨跌停幅度不同。
    """
    # Tier 1股票（S&P 500, Russell 1000, 部分ETF）
    TIER1_BANDS: Dict[str, float] = field(default_factory=lambda: {
        'open':    0.05, 
        '09:45':   0.05,
        '15:30':   0.05, 
        'regular': 0.10,
    })
    
    # Tier 2股票（其余NMS股票）
    TIER2_BANDS: Dict[str, float] = field(default_factory=lambda: {
        'open':    0.10, 
        '09:45':   0.10,
        '15:30':   0.10, 
        'regular': 0.20,
    })
    
    # 全市场熔断阈值（标普500指数）
    MARKET_CIRCUIT_BREAKERS: Dict[int, str] = field(default_factory=lambda: {
        -7:  'Level 1: 交易暂停15分钟',
        -13: 'Level 2: 交易暂停15分钟', 
        -20: 'Level 3: 当日剩余时间停止交易',
    })


# ============================================================================
# 美股交易所假期（2025-2027）
# 非完整列表，实际应接入交易所API或定期更新
# ============================================================================
EXCHANGE_HOLIDAYS: Dict[int, List[date]] = {
    2025: [
        date(2025, 1, 1),    # 元旦 New Year's Day
        date(2025, 1, 20),   # 马丁·路德·金纪念日
        date(2025, 2, 17),   # 总统日 Presidents' Day
        date(2025, 4, 18),   # 耶稣受难日 Good Friday
        date(2025, 5, 26),   # 阵亡将士纪念日 Memorial Day
        date(2025, 6, 19),   # 六月节 Juneteenth
        date(2025, 7, 4),    # 独立日 Independence Day
        date(2025, 9, 1),    # 劳动节 Labor Day
        date(2025, 11, 27),  # 感恩节 Thanksgiving
        date(2025, 12, 25),  # 圣诞节 Christmas
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
# 券商代码映射
# ============================================================================
BROKER_CODES: Dict[str, str] = {
    'ibkr':         'Interactive Brokers',
    'alpaca':       'Alpaca Markets',
    'td_ameritrade': 'TD Ameritrade (Schwab)',
    'tradier':      'Tradier',
    'robinhood':    'Robinhood',
}


# ============================================================================
# 技术指标默认参数
# ============================================================================
TECHNICAL_INDICATORS_DEFAULTS: Dict[str, int] = {
    'sma_fast':        5,     # 快线均线周期
    'sma_medium':      20,    # 中线均线周期
    'sma_slow':        60,    # 慢线均线周期
    'ema_fast':        12,    # 快线EMA
    'ema_slow':        26,    # 慢线EMA
    'macd_signal':     9,     # MACD信号线周期
    'rsi_period':      14,    # RSI计算周期
    'bb_period':       20,    # 布林带计算周期
    'bb_std':          2,     # 布林带标准差倍数
    'atr_period':      14,    # ATR计算周期
    'volume_ma_period': 20,   # 成交量均线周期
}


